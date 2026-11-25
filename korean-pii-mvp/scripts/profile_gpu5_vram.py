"""Measure Stage A FP32 AdamW VRAM with full-length synthetic token batches.

GPU 5 only. Includes initial optimizer allocation, steady steps, and eval while
retaining the optimizer and last gradients, matching stage_a.train(). No model
checkpoint is written. Results describe shape-dependent memory, not accuracy.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

# Set visibility before importing any ML library, including on direct invocation.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu5_uuid = subprocess.check_output([
    "nvidia-smi", "-i", "5", "--query-gpu=uuid", "--format=csv,noheader",
], text=True).strip()
if not gpu5_uuid.startswith("GPU-") or "\n" in gpu5_uuid:
    raise RuntimeError("Cannot identify physical GPU 5 unambiguously")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu5_uuid
os.environ.setdefault("HF_HOME", "/workspace/korean-pii-mvp/.cache/huggingface")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/workspace/korean-pii-mvp/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/workspace/korean-pii-mvp/.cache/triton")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class ProcessMemorySampler:
    def __init__(self):
        self.peak_mib = 0
        self.samples = 0
        self.matching_process_samples = 0
        self.device_peak_mib = 0
        self.observed_compute_pids = set()
        self.errors = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                text = subprocess.check_output([
                    "nvidia-smi", "-i", "5", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ], text=True, stderr=subprocess.DEVNULL, timeout=5)
                for row in text.splitlines():
                    fields = [field.strip() for field in row.split(",")]
                    if len(fields) == 2 and fields[0].isdigit():
                        self.observed_compute_pids.add(int(fields[0]))
                    if len(fields) == 2 and fields[0] == str(os.getpid()):
                        self.peak_mib = max(self.peak_mib, int(fields[1]))
                        self.matching_process_samples += 1
                device_memory = subprocess.check_output([
                    "nvidia-smi", "-i", "5", "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ], text=True, stderr=subprocess.DEVNULL, timeout=5)
                self.device_peak_mib = max(self.device_peak_mib, int(device_memory.strip()))
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError):
                self.errors += 1
            self.stop.wait(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/kdpii-v1/train.jsonl")
    parser.add_argument("--output", default="logs/gpu5-vram-profile.json")
    parser.add_argument("--model-name-or-path", default="skt/A.X-Encoder-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--steady-steps", type=int, default=3)
    args = parser.parse_args()
    if min(args.max_length, args.train_batch_size, args.eval_batch_size, args.steady_steps) <= 0:
        parser.error("length, batch sizes, and steps must be positive")
    if Path(args.output).resolve() == Path(args.train).resolve():
        parser.error("Profile output must not overwrite the input dataset")

    import torch
    import transformers
    from kpii_mvp.data import read_jsonl
    from kpii_mvp.stage_a import build_label_inventory, _load_training_model, model_forward_inputs

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "GPU 5 must be the only visible CUDA device"
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    gpu_metadata = subprocess.check_output([
        "nvidia-smi", "-i", "5", "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    uuid = gpu_metadata.split(",")[1].strip().removeprefix("GPU-")
    torch_uuid = getattr(properties, "uuid", None)
    if torch_uuid is not None:
        assert str(torch_uuid).removeprefix("GPU-").lower() == uuid.lower(), "Visible CUDA device is not physical GPU 5"

    torch.manual_seed(42)
    label2id = build_label_inventory(read_jsonl(args.train))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model, _ = _load_training_model(transformers, args.model_name_or_path, label2id, False, False)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    sampler = ProcessMemorySampler()
    sampler.thread.start()
    started = time.monotonic()
    phases = []
    try:
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        special_count = tokenizer.num_special_tokens_to_add(pair=False)
        content_length = args.max_length - special_count
        if content_length <= 0:
            raise ValueError("Sequence length cannot fit tokenizer special tokens")
        body = tokenizer("가상 문서의 정보를 검토합니다.", add_special_tokens=False)["input_ids"]
        repeated = (body * math.ceil(content_length / len(body)))[:content_length]
        ids = tokenizer.build_inputs_with_special_tokens(repeated)
        assert len(ids) == args.max_length

        def batch(size, with_labels):
            values = {"input_ids": torch.tensor([ids] * size, device=device),
                      "attention_mask": torch.ones((size, args.max_length), dtype=torch.long, device=device)}
            if with_labels:
                labels = torch.zeros((size, args.max_length), dtype=torch.long, device=device)
                labels[:, 0] = labels[:, -1] = -100
                if args.max_length > 10:
                    labels[:, 8] = 1
                    labels[:, 9] = 2
                values["labels"] = labels
            return values

        train_batch = batch(args.train_batch_size, True)

        def record(name, operation):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            phase_started = time.monotonic()
            operation()
            torch.cuda.synchronize()
            row = {
                "phase": name, "seconds": round(time.monotonic() - phase_started, 3),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "ending_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            }
            phases.append(row)
            print(json.dumps(row), flush=True)

        def train_steps(count):
            model.train()
            for _ in range(count):
                optimizer.zero_grad(set_to_none=True)
                loss = model(**model_forward_inputs(model, train_batch)).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite profile loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        record("first_train_step", lambda: train_steps(1))
        record("steady_train_steps", lambda: train_steps(args.steady_steps))

        def evaluate():
            model.eval()
            eval_batch = batch(args.eval_batch_size, False)
            with torch.no_grad():
                for _ in range(2):
                    logits = model(**model_forward_inputs(model, eval_batch)).logits
                    scores, labels = torch.softmax(logits.float(), dim=-1).max(dim=-1)
            del scores, labels, logits, eval_batch

        record("eval_with_optimizer_and_last_gradients", evaluate)
        # Allow the process sampler to observe final CUDA context/module memory.
        time.sleep(0.3)
    finally:
        sampler.stop.set()
        sampler.thread.join(timeout=6)

    peak_reserved = max(row["peak_reserved_gib"] for row in phases)
    observed_total = sampler.peak_mib / 1024 if sampler.peak_mib else None
    basis = max(peak_reserved, observed_total or 0, sampler.device_peak_mib / 1024)
    recommended = math.ceil((basis + max(2.0, basis * 0.3)) / 2) * 2
    result = {
        "physical_gpu_index": 5, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "logical_device": str(device), "gpu_metadata": gpu_metadata,
        "torch_device_uuid": str(torch_uuid) if torch_uuid is not None else None,
        "python_environment": str(Path(os.sys.prefix).resolve()),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "model": args.model_name_or_path, "parameters": parameter_count,
        "dtype": "float32", "optimizer": "AdamW", "amp": False,
        "max_length": args.max_length, "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size, "training_steps": 1 + args.steady_steps,
        "phases": phases, "nvidia_smi_own_process_peak_mib": sampler.peak_mib or None,
        "nvidia_smi_matching_process_samples": sampler.matching_process_samples,
        "nvidia_smi_gpu5_total_peak_mib": sampler.device_peak_mib or None,
        "nvidia_smi_observed_compute_pids": sorted(sampler.observed_compute_pids),
        "nvidia_smi_samples": sampler.samples, "nvidia_smi_sampling_errors": sampler.errors,
        "recommended_available_vram_gib": recommended,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "notes": ["Fixed full-length synthetic token batches measure memory only.",
                  "nvidia-smi is sampled; short peaks may be missed. PyTorch peaks are exact allocator statistics.",
                  "GPU-total usage includes CUDA context and any concurrent process; container/host PID namespaces can prevent own-process matching.",
                  "Recommendation includes at least 2 GiB or 30 percent headroom; longer sequences/larger batches need remeasurement.",
                  "No full dataset training or checkpoint saving was performed."],
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
