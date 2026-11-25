#!/usr/bin/env bash
# Future execution only: reviewed data are required; no corpus is generated.
# Usage: bash scripts/train_eval_b_gpu5.sh DATA_DIR [OUTPUT_DIR] [A_MODEL_DIR]
# DATA_DIR: train/valid/test.jsonl plus train/valid/test.reviews.jsonl.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  printf '%s\n' 'Usage: bash scripts/train_eval_b_gpu5.sh DATA_DIR [OUTPUT_DIR] [A_MODEL_DIR]' >&2
  exit 2
fi

KPII_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Resolve caller-supplied paths before changing directory.
KPII_DATA_DIR="$(realpath -e -- "$1")"
KPII_RUN_DIR="$(realpath -m -- "${2:-$KPII_PROJECT_DIR/runs/b-real-gpu5-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}")"
KPII_A_MODEL_DIR="$(realpath -e -- "${3:-/workspace/korean-pii-mvp/runs/a-native-gpu5-20261019_020722/best}")"

if [[ ! -d "$KPII_DATA_DIR" || ! -d "$KPII_A_MODEL_DIR" ]]; then
  printf '%s\n' 'DATA_DIR and A_MODEL_DIR must be existing directories.' >&2
  exit 2
fi
if [[ -e "$KPII_RUN_DIR" || -L "$KPII_RUN_DIR" ]]; then
  printf 'Output directory already exists; choose a new path: %s\n' "$KPII_RUN_DIR" >&2
  exit 2
fi
if [[ "$KPII_RUN_DIR" == "$KPII_A_MODEL_DIR/"* || "$KPII_RUN_DIR" == "$KPII_DATA_DIR/"* ]]; then
  printf '%s\n' 'Output directory must be outside the source checkpoint and reviewed-data directories.' >&2
  exit 2
fi
for KPII_INPUT_NAME in train.jsonl valid.jsonl test.jsonl train.reviews.jsonl valid.reviews.jsonl test.reviews.jsonl; do
  if [[ ! -r "$KPII_DATA_DIR/$KPII_INPUT_NAME" || ! -s "$KPII_DATA_DIR/$KPII_INPUT_NAME" ]]; then
    printf 'Missing, empty or unreadable reviewed input: %s/%s\n' "$KPII_DATA_DIR" "$KPII_INPUT_NAME" >&2
    exit 2
  fi
done
if [[ ! -s "$KPII_A_MODEL_DIR/training_manifest.json" ]]; then
  printf '%s\n' 'A_MODEL_DIR must be a trained A checkpoint containing training_manifest.json.' >&2
  exit 2
fi

cd -- "$KPII_PROJECT_DIR"
# UUID selection exposes physical GPU 5 only; torch sees it as logical cuda:0.
source scripts/activate_gpu5.sh

# Inspect all three approved splits before any B fitting. Failures create no run.
KPII_PREFLIGHT_JSON="$(kpii preflight-b \
  --train "$KPII_DATA_DIR/train.jsonl" \
  --train-reviews "$KPII_DATA_DIR/train.reviews.jsonl" \
  --dev "$KPII_DATA_DIR/valid.jsonl" \
  --dev-reviews "$KPII_DATA_DIR/valid.reviews.jsonl" \
  --test "$KPII_DATA_DIR/test.jsonl" \
  --test-reviews "$KPII_DATA_DIR/test.reviews.jsonl")"

mkdir -p -- "$(dirname -- "$KPII_RUN_DIR")"
mkdir -- "$KPII_RUN_DIR"
printf '%s\n' "$KPII_PREFLIGHT_JSON" > "$KPII_RUN_DIR/preflight.json"
printf 'Output: %s\nPhysical GPU: 5 (logical cuda:0)\n' "$KPII_RUN_DIR"

# B uses reviewed gold candidates. The A encoder is copied; its head/files stay
# separate. Dev candidate loss selects best/, and test never selects a model.
kpii train-b \
  --train "$KPII_DATA_DIR/train.jsonl" \
  --train-reviews "$KPII_DATA_DIR/train.reviews.jsonl" \
  --dev "$KPII_DATA_DIR/valid.jsonl" \
  --dev-reviews "$KPII_DATA_DIR/valid.reviews.jsonl" \
  --encoder-name-or-path "$KPII_A_MODEL_DIR" \
  --output-dir "$KPII_RUN_DIR/model" \
  --device cuda --epochs 3 --max-steps 0 --max-length 512 \
  --batch-size 4 --eval-batch-size 8 \
  --learning-rate 3e-5 --weight-decay 0.01 --seed 42 \
  2>&1 | tee "$KPII_RUN_DIR/train.log"

KPII_B_MODEL_DIR="$KPII_RUN_DIR/model/best"

kpii_score_split() {
  local KPII_SPLIT_NAME="$1"
  # The scan copy has no gold mentions, actions or reasons. Its annotation_scope
  # lists A's native prediction types, enabling the saved 33-type checkpoint.
  kpii prepare-b-a-input \
    --input "$KPII_DATA_DIR/$KPII_SPLIT_NAME.jsonl" \
    --model-dir "$KPII_A_MODEL_DIR" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-input.jsonl" \
    2>&1 | tee "$KPII_RUN_DIR/$KPII_SPLIT_NAME-prepare.log"

  kpii predict-a \
    --model-dir "$KPII_A_MODEL_DIR" \
    --input "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-input.jsonl" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-candidates.jsonl" \
    --device cuda --batch-size 8 --max-length 512 --stride 128 \
    2>&1 | tee "$KPII_RUN_DIR/$KPII_SPLIT_NAME-predict-a.log"

  kpii score-b \
    --model-dir "$KPII_B_MODEL_DIR" \
    --input "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-candidates.jsonl" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-b-scores.jsonl" \
    --device cuda --batch-size 8 \
    2>&1 | tee "$KPII_RUN_DIR/$KPII_SPLIT_NAME-score-b.log"
}

# Freeze the release threshold using validation gold and actual A predictions.
kpii_score_split valid
kpii calibrate-b \
  --gold "$KPII_DATA_DIR/valid.jsonl" \
  --reviews "$KPII_DATA_DIR/valid.reviews.jsonl" \
  --candidates "$KPII_RUN_DIR/valid-a-candidates.jsonl" \
  --scores "$KPII_RUN_DIR/valid-b-scores.jsonl" \
  --model-manifest "$KPII_B_MODEL_DIR/b_manifest.json" \
  --max-added-leak-rate 0.0 --min-overmask-improvement 0.0 \
  --output "$KPII_RUN_DIR/calibration.json" \
  2>&1 | tee "$KPII_RUN_DIR/calibrate.log"

# Only after calibration: run the held-out test with the same fixed A and B.
kpii_score_split test
kpii evaluate-ab \
  --gold "$KPII_DATA_DIR/test.jsonl" \
  --reviews "$KPII_DATA_DIR/test.reviews.jsonl" \
  --candidates "$KPII_RUN_DIR/test-a-candidates.jsonl" \
  --scores "$KPII_RUN_DIR/test-b-scores.jsonl" \
  --model-manifest "$KPII_B_MODEL_DIR/b_manifest.json" \
  --calibration "$KPII_RUN_DIR/calibration.json" \
  --output-dir "$KPII_RUN_DIR/test-evaluation" \
  2>&1 | tee "$KPII_RUN_DIR/evaluate.log"

printf '\nCompleted: %s\nTest report: %s/test-evaluation/ab-evaluation.json\n' "$KPII_RUN_DIR" "$KPII_RUN_DIR"
