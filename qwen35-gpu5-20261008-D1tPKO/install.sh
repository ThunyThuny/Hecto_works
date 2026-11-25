#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=/workspace/qwen35-gpu5-20261008-D1tPKO
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off
"$RUN_DIR/venv/bin/python" -m pip install --upgrade pip
"$RUN_DIR/venv/bin/python" -m pip install 'vllm==0.17.0' --extra-index-url https://download.pytorch.org/whl/cu129
"$RUN_DIR/venv/bin/python" -m pip check
"$RUN_DIR/venv/bin/python" -m pip freeze > "$RUN_DIR/requirements.freeze.txt"
"$RUN_DIR/venv/bin/python" -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda, "vllm", vllm.__version__)'
touch "$RUN_DIR/INSTALL_OK"
