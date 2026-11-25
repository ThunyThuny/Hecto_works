# Source this file from bash: source /workspace/korean-pii-mvp/scripts/activate_gpu5.sh
source /workspace/kpii-env/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID
KPII_GPU_UUID="$(nvidia-smi -i 5 --query-gpu=uuid --format=csv,noheader)" || {
  export CUDA_VISIBLE_DEVICES=""
  return 1
}
if [[ "$KPII_GPU_UUID" != GPU-* || "$KPII_GPU_UUID" == *$'\n'* ]]; then
  export CUDA_VISIBLE_DEVICES=""
  printf '%s\n' 'Cannot identify physical GPU 5 unambiguously.' >&2
  return 1
fi
export CUDA_VISIBLE_DEVICES="$KPII_GPU_UUID"
export HF_HOME=/workspace/korean-pii-mvp/.cache/huggingface
export PIP_CACHE_DIR=/workspace/korean-pii-mvp/.cache/pip
export TORCHINDUCTOR_CACHE_DIR=/workspace/korean-pii-mvp/.cache/torchinductor
export TRITON_CACHE_DIR=/workspace/korean-pii-mvp/.cache/triton
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1
