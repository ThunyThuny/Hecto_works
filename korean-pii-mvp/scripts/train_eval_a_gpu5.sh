#!/usr/bin/env bash
# Full training is started only when this script is explicitly run.
set -euo pipefail
cd /workspace/korean-pii-mvp
source scripts/activate_gpu5.sh

KPII_RUN_DIR="runs/a-native-gpu5-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
mkdir -p runs
printf 'Output: %s/%s\nPhysical GPU: 5 (logical cuda:0)\n' "$PWD" "$KPII_RUN_DIR"

kpii train-a \
  --train data/processed/kdpii-v1/train.jsonl \
  --dev data/processed/kdpii-v1/valid.jsonl \
  --model-name-or-path skt/A.X-Encoder-base \
  --output-dir "$KPII_RUN_DIR" \
  --device cuda --epochs 3 --max-steps 0 \
  --max-length 512 --stride 128 \
  --batch-size 4 --eval-batch-size 8 \
  --learning-rate 3e-5 --weight-decay 0.01 --seed 42 \
  2>&1 | tee "${KPII_RUN_DIR}.train.log"

kpii predict-a \
  --model-dir "$KPII_RUN_DIR/best" \
  --input data/processed/kdpii-v1/test.jsonl \
  --output "$KPII_RUN_DIR/test-predictions.jsonl" \
  --device cuda --batch-size 8 \
  2>&1 | tee "${KPII_RUN_DIR}.predict.log"

kpii evaluate-detection \
  --gold data/processed/kdpii-v1/test.jsonl \
  --pred "$KPII_RUN_DIR/test-predictions.jsonl" \
  --output "$KPII_RUN_DIR/test-detection.json"

printf '\nCompleted: %s/%s\n' "$PWD" "$KPII_RUN_DIR"
