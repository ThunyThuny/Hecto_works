#!/usr/bin/env bash
# Generation and checkpoint binding only. No model training or GPU allocation.
set -euo pipefail
cd /workspace/korean-pii-mvp
source /workspace/kpii-env/bin/activate
KPII_SYNTHETIC_DIR="${1:-/workspace/nfs/datasets/dylee/PII/synthetic_data}"
kpii generate-synthetic --output-dir "$KPII_SYNTHETIC_DIR" \
  --seed 20261020 --train-count 30 --valid-count 10 --test-count 10
kpii lock-a --output "$KPII_SYNTHETIC_DIR/a_checkpoint.lock.json"
printf 'Review: %s/review.html\nNo training executed.\n' "$KPII_SYNTHETIC_DIR"
