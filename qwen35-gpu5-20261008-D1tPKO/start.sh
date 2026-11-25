#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=/workspace/qwen35-gpu5-20261008-D1tPKO
export CUDA_VISIBLE_DEVICES=GPU-497f23c6-72ad-1ee4-b9de-6f7a9b7b5fe1
export HF_HOME="$RUN_DIR/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$RUN_DIR/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/cache/torchinductor"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export PYTHONUNBUFFERED=1
cd "$RUN_DIR"
exec "$RUN_DIR/venv/bin/vllm" serve /workspace/nfs/models/llm/foundation/Qwen3.5-27B-FP8 --host 127.0.0.1 --port 8005 --served-model-name Qwen3.5-27B-FP8 --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.55 --kv-cache-dtype auto --reasoning-parser qwen3 --enforce-eager
