#!/usr/bin/env python3
"""Keep the local Qwen model weights in GPU memory without inference.

Run with the existing environment:
  /workspace/saerok/bin/python /workspace/load_qwen_gpu.py
Stop with Ctrl+C, or send SIGTERM to the printed PID.
"""
import argparse
import json
import os
import signal
import threading
from pathlib import Path

DEFAULT_MODEL = '/workspace/nfs/models/llm/foundation/Qwen3.5-27B-FP8'
DEFAULT_GPU = 'GPU-497f23c6-72ad-1ee4-b9de-6f7a9b7b5fe1'  # Physical GPU 5.
GIB = 1024 ** 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Local model directory')
    parser.add_argument('--gpu', default=DEFAULT_GPU, help='GPU index or UUID; default: GPU 5')
    args = parser.parse_args()

    # Restrict this process before importing CUDA libraries.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from safetensors import safe_open

    model_dir = Path(args.model).resolve(strict=True)
    index = json.loads((model_dir / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    shards = [model_dir / name for name in sorted(set(weight_map.values()))]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing model shards: {missing}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one accessible CUDA GPU is required.')

    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    required_bytes = sum(path.stat().st_size for path in shards)
    if free_bytes < required_bytes + GIB:
        raise RuntimeError(
            f'Insufficient free VRAM: {free_bytes/GIB:.2f} GiB available; '
            f'about {required_bytes/GIB+1:.2f} GiB required.'
        )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    weights = {}
    print(f'PID={os.getpid()} GPU={args.gpu} ({torch.cuda.get_device_name(0)})', flush=True)
    print(f'Model={model_dir}', flush=True)
    try:
        for number, shard in enumerate(shards, 1):
            with safe_open(str(shard), framework='pt', device='cuda:0') as handle:
                for name in handle.keys():
                    if stop.is_set():
                        return
                    if name in weights:
                        raise RuntimeError(f'Duplicate tensor: {name}')
                    # Preserve original FP8/BF16 dtypes and hold strong references.
                    weights[name] = handle.get_tensor(name)
            print(f'Loaded shard {number}/{len(shards)}: {shard.name}', flush=True)
        if set(weights) != set(weight_map):
            raise RuntimeError('Loaded tensor names do not match the model index.')
        if any(tensor.device.type != 'cuda' for tensor in weights.values()):
            raise RuntimeError('Some model tensors were not loaded onto CUDA.')
        torch.cuda.synchronize()
        print(
            f'READY: {len(weights)} tensors; '
            f'VRAM allocated={torch.cuda.memory_allocated()/GIB:.2f} GiB; '
            f'reserved={torch.cuda.memory_reserved()/GIB:.2f} GiB.',
            flush=True,
        )
        print('Weights remain resident. No inference or repeated GPU computation.', flush=True)
        stop.wait()
    finally:
        weights.clear()
        torch.cuda.empty_cache()
        print('Released model weights.', flush=True)


if __name__ == '__main__':
    main()
