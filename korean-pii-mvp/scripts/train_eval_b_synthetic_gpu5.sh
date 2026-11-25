#!/usr/bin/env bash
# Future execution only. This command never approves reviews or creates data.
# Usage: bash scripts/train_eval_b_synthetic_gpu5.sh [--check-only] DATA_DIR [OUTPUT_DIR]
# DATA_DIR must be the output of synthetic_flow prepare-reviewed, not a pending packet.
set -euo pipefail

KPII_CHECK_ONLY=false
KPII_POSITIONAL=()
for KPII_ARGUMENT in "$@"; do
  case "$KPII_ARGUMENT" in
    --check-only) KPII_CHECK_ONLY=true ;;
    --*) printf 'Unknown option: %s\n' "$KPII_ARGUMENT" >&2; exit 2 ;;
    *) KPII_POSITIONAL+=("$KPII_ARGUMENT") ;;
  esac
done
if [[ ${#KPII_POSITIONAL[@]} -lt 1 || ${#KPII_POSITIONAL[@]} -gt 2 ]]; then
  printf '%s\n' 'Usage: bash scripts/train_eval_b_synthetic_gpu5.sh [--check-only] DATA_DIR [OUTPUT_DIR]' >&2
  exit 2
fi

KPII_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
KPII_DATA_DIR="$(realpath -e -- "${KPII_POSITIONAL[0]}")"
KPII_RUN_DIR="$(realpath -m -- "${KPII_POSITIONAL[1]:-$KPII_PROJECT_DIR/runs/b-synthetic-gpu5-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}")"
KPII_A_MODEL_DIR='/workspace/korean-pii-mvp/runs/a-native-gpu5-20261019_020722/best'
KPII_A_MANIFEST_SHA256='87fd30cf17ee1afe5ea1d995c9f511bc28d11e5935efdc80750c0a30dd56d591'
KPII_PYTHON='/workspace/kpii-env/bin/python'
KPII_A_LOCK="$KPII_DATA_DIR/a_checkpoint.lock.json"

if [[ ! -d "$KPII_DATA_DIR" || ! -x "$KPII_PYTHON" ]]; then
  printf '%s\n' 'An existing reviewed DATA_DIR and /workspace/kpii-env Python are required.' >&2
  exit 2
fi
if [[ -e "$KPII_RUN_DIR" || -L "$KPII_RUN_DIR" ]]; then
  printf 'Output directory already exists: %s\n' "$KPII_RUN_DIR" >&2
  exit 2
fi
if [[ "$KPII_RUN_DIR" == "$KPII_DATA_DIR" || "$KPII_RUN_DIR" == "$KPII_DATA_DIR/"* ||
      "$KPII_RUN_DIR" == "$KPII_A_MODEL_DIR" || "$KPII_RUN_DIR" == "$KPII_A_MODEL_DIR/"* ]]; then
  printf '%s\n' 'Output must be outside the reviewed data and fixed A checkpoint.' >&2
  exit 2
fi
for KPII_INPUT_NAME in train.jsonl valid.jsonl test.jsonl train.reviews.jsonl valid.reviews.jsonl test.reviews.jsonl a_checkpoint.lock.json; do
  if [[ ! -r "$KPII_DATA_DIR/$KPII_INPUT_NAME" || ! -s "$KPII_DATA_DIR/$KPII_INPUT_NAME" ]]; then
    printf 'Missing, empty or unreadable input: %s/%s\n' "$KPII_DATA_DIR" "$KPII_INPUT_NAME" >&2
    exit 2
  fi
done

# All check-only operations are CPU-only validation, without fitting or output creation.
cd -- "$KPII_PROJECT_DIR"
export PYTHONPATH="$KPII_PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ "$(sha256sum -- "$KPII_A_MODEL_DIR/training_manifest.json" | cut -d ' ' -f 1)" != "$KPII_A_MANIFEST_SHA256" ]]; then
  printf '%s\n' 'The fixed A training manifest has changed. Refusing to continue.' >&2
  exit 2
fi
kpii_verify_locked_a() {
  "$KPII_PYTHON" -m kpii_mvp.synthetic_flow verify-a-lock \
    --lock "$KPII_A_LOCK" --a-model-dir "$KPII_A_MODEL_DIR"
}
kpii_verify_locked_a

# Recheck every locked A artifact on success AND failure, preserving a prior error.
# A changed lock is not refreshed automatically, including after a failed run.
kpii_finish() {
  local KPII_ORIGINAL_STATUS=$?
  trap - EXIT
  if ! kpii_verify_locked_a; then
    printf '%s\n' 'Fixed A verification failed after execution; retain artifacts for investigation.' >&2
    if [[ $KPII_ORIGINAL_STATUS -eq 0 ]]; then KPII_ORIGINAL_STATUS=1; fi
  fi
  exit "$KPII_ORIGINAL_STATUS"
}
trap kpii_finish EXIT

"$KPII_PYTHON" - "$KPII_DATA_DIR" <<'PY'
import json
from pathlib import Path
import sys
from kpii_mvp.b_model import reviewed_data_provenance
from kpii_mvp.data import read_jsonl

data_dir = Path(sys.argv[1])
reports = {}
for split in ('train', 'valid', 'test'):
    report = reviewed_data_provenance(read_jsonl(data_dir / (split + '.jsonl')))
    if report['classification'] != 'synthetic':
        raise ValueError(f'{split}: this wrapper requires declared synthetic-only sources; {report}')
    reports[split] = report
print(json.dumps({'source_provenance': reports, 'training_performed': False}, ensure_ascii=False, indent=2))
PY

# Require human APPROVED attestations, complete gold, both classes and split separation.
"$KPII_PYTHON" -m kpii_mvp.b_pipeline preflight \
  --train "$KPII_DATA_DIR/train.jsonl" --train-reviews "$KPII_DATA_DIR/train.reviews.jsonl" \
  --dev "$KPII_DATA_DIR/valid.jsonl" --dev-reviews "$KPII_DATA_DIR/valid.reviews.jsonl" \
  --test "$KPII_DATA_DIR/test.jsonl" --test-reviews "$KPII_DATA_DIR/test.reviews.jsonl"

if [[ "$KPII_CHECK_ONLY" == true ]]; then
  printf '%s\n' 'Check-only completed. No training, inference or output directory creation performed.'
  exit 0
fi

# Delegate only after all gates pass. Existing helper selects physical GPU 5 by UUID;
# A is used solely for encoder initialization and prediction, never A training.
bash scripts/train_eval_b_gpu5.sh "$KPII_DATA_DIR" "$KPII_RUN_DIR" "$KPII_A_MODEL_DIR"
