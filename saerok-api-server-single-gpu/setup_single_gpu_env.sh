#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-/workspace/saerok}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python3 -m venv "$VENV_PATH"
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_PATH/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0
"$VENV_PATH/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-single-gpu.txt"

export CUDACXX="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
export CMAKE_ARGS="${CMAKE_ARGS:--DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
"$VENV_PATH/bin/python" -m pip install --no-cache-dir --force-reinstall \
  llama-cpp-python==0.3.20

# The pipeline passes an in-memory waveform to pyannote and does not use
# torchcodec's decoder.  Removing it avoids CUDA runtime-library conflicts.
"$VENV_PATH/bin/python" -m pip uninstall -y torchcodec || true

"$VENV_PATH/bin/python" -c \
  "import torch, soundfile, faster_whisper, pyannote.audio, llama_cpp; print('environment imports: OK'); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count()); print('llama_cpp:', llama_cpp.__version__)"
