# Saerok single-GPU runner

This copy runs the v2 pipeline on one CUDA GPU.  Whisper, pyannote, and Gemma
are loaded, executed, and released sequentially so their VRAM allocations do
not overlap.

## Run

```bash
cd /workspace/saerok-api-server-single-gpu
read -rsp "HF token: " HF_TOKEN && export HF_TOKEN && echo

./run_single_gpu.sh \
  /workspace/data/nfs/datasets/dylee/Saerok/test_files/1hour.webm \
  --meeting-type meeting \
  --num-speakers 2 \
  --output-root /workspace/Result/single_gpu
```

GPU 0 is used by default.  Select another physical GPU before launching:

```bash
export SAEROK_CUDA_DEVICE=3
```

The default LLM context is 131,072 tokens.  Reduce it when using a GPU with less
VRAM:

```bash
export LLM_N_CTX=65536
```

The Gemma Q8 model is about 28 GB.  This setup builds llama-cpp for the server's
H200 (compute capability 9.0); the first run also downloads Whisper, pyannote,
and Silero cache files.

## Container GPU requirement

`nvidia-smi` and the following command must succeed inside the container before
running the pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/saerok/bin/python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
```

The expected result is `True 1`.  `run_single_gpu.sh` performs this check before
loading any model.  If CUDA reports no devices even though `/dev/nvidia*` exists,
restart or recreate the container with one GPU assigned (for Docker, for example,
`--gpus device=0`).
