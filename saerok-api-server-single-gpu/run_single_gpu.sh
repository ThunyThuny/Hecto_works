#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-/workspace/saerok}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  echo "Virtual environment not found: $VENV_PATH" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${SAEROK_CUDA_DEVICE:-0}"

if ! "$VENV_PATH/bin/python" - <<'PY'
import sys

import torch

device_count = torch.cuda.device_count()
if not torch.cuda.is_available() or device_count != 1:
    print(
        "CUDA preflight failed: expected exactly one visible GPU, "
        f"but torch reports {device_count}.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
then
  echo "Expose one GPU to this container (for Docker: --gpus device=<id>) and retry." >&2
  exit 3
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for the gated pyannote model." >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$VENV_PATH/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$VENV_PATH/cache/torch}"
export LLM_PATH="${LLM_PATH:-/workspace/data/nfs/models/llm/foundation/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf}"
export LLM_N_CTX="${LLM_N_CTX:-131072}"

mkdir -p "$HF_HOME" "$TORCH_HOME"
exec "$VENV_PATH/bin/python" "$SCRIPT_DIR/main_gemma4_v2.py" "$@"
