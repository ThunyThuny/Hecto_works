#!/usr/bin/env bash
# V4 experiment: fixed A + versioned candidate refinement + target-focused B.
# Usage: bash scripts/train_eval_b_v4_gpu5.sh [--check-only] PACKET_DIR [OUTPUT_DIR]
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
  printf '%s\n' 'Usage: bash scripts/train_eval_b_v4_gpu5.sh [--check-only] PACKET_DIR [OUTPUT_DIR]' >&2
  exit 2
fi

KPII_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
KPII_PACKET_DIR="$(realpath -e -- "${KPII_POSITIONAL[0]}")"
KPII_RUN_DIR="$(realpath -m -- "${KPII_POSITIONAL[1]:-$KPII_PROJECT_DIR/runs/b-synthetic-v4-gpu5-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)}")"
KPII_A_MODEL_DIR='/workspace/korean-pii-mvp/runs/a-native-gpu5-20261019_020722/best'
KPII_A_MANIFEST_SHA256='87fd30cf17ee1afe5ea1d995c9f511bc28d11e5935efdc80750c0a30dd56d591'
KPII_PYTHON='/workspace/kpii-env/bin/python'
KPII_A_LOCK="$KPII_PACKET_DIR/a_checkpoint.lock.json"

if [[ ! -d "$KPII_PACKET_DIR" || ! -d "$KPII_A_MODEL_DIR" || ! -x "$KPII_PYTHON" ]]; then
  printf '%s\n' 'Existing PACKET_DIR, fixed A checkpoint and /workspace/kpii-env Python are required.' >&2
  exit 2
fi
if [[ -e "$KPII_RUN_DIR" || -L "$KPII_RUN_DIR" ]]; then
  printf 'Output directory already exists: %s\n' "$KPII_RUN_DIR" >&2
  exit 2
fi
if [[ "$KPII_RUN_DIR" == "$KPII_PACKET_DIR" || "$KPII_RUN_DIR" == "$KPII_PACKET_DIR/"* ||
      "$KPII_RUN_DIR" == "$KPII_A_MODEL_DIR" || "$KPII_RUN_DIR" == "$KPII_A_MODEL_DIR/"* ]]; then
  printf '%s\n' 'Output must be outside the original packet and fixed A checkpoint.' >&2
  exit 2
fi
for KPII_INPUT_NAME in proposals/train.jsonl proposals/valid.jsonl proposals/test.jsonl review/reviews.jsonl generation_manifest.json automated_audit.json a_checkpoint.lock.json; do
  if [[ ! -r "$KPII_PACKET_DIR/$KPII_INPUT_NAME" || ! -s "$KPII_PACKET_DIR/$KPII_INPUT_NAME" ]]; then
    printf 'Missing, empty or unreadable input: %s/%s\n' "$KPII_PACKET_DIR" "$KPII_INPUT_NAME" >&2
    exit 2
  fi
done

# All checks before --check-only exits use CPU data validation only.
cd -- "$KPII_PROJECT_DIR"
export PYTHONPATH="$KPII_PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ "$(sha256sum -- "$KPII_A_MODEL_DIR/training_manifest.json" | cut -d ' ' -f 1)" != "$KPII_A_MANIFEST_SHA256" ]]; then
  printf '%s\n' 'The fixed A training manifest has changed. Refusing to continue.' >&2
  exit 2
fi
KPII_A_LOCK_SHA256="$(sha256sum -- "$KPII_A_LOCK" | cut -d ' ' -f 1)"
kpii_verify_locked_a() {
  if [[ "$(sha256sum -- "$KPII_A_LOCK" | cut -d ' ' -f 1)" != "$KPII_A_LOCK_SHA256" ]]; then
    printf '%s\n' 'The original A lock changed during execution.' >&2
    return 1
  fi
  "$KPII_PYTHON" -m kpii_mvp.synthetic_flow verify-a-lock \
    --lock "$KPII_A_LOCK" --a-model-dir "$KPII_A_MODEL_DIR"
}
kpii_verify_locked_a
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

KPII_PREFLIGHT_JSON="$("$KPII_PYTHON" - "$KPII_PACKET_DIR" "$KPII_A_MODEL_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from kpii_mvp.b_data import load_reviews, validate_split_separation
from kpii_mvp.b_experiment import validate_b_evidence, validate_experiment_packet
from kpii_mvp.b_inputs import keep_eligible
from kpii_mvp.b_input_audit import audit_inputs, load_tokenizer
from kpii_mvp.b_model import validate_train_dev
from kpii_mvp.data import read_jsonl

packet = Path(sys.argv[1])
packet_report = validate_experiment_packet(packet)
generation = json.loads((packet / 'generation_manifest.json').read_text(encoding='utf-8'))
if generation.get('generator_version') != 'synthetic_rule_based_v4_bulk':
    raise ValueError('This wrapper requires the new V4 packet; use the original wrapper for V3')
reviews = load_reviews(packet / 'review' / 'reviews.jsonl')
splits, reports = {}, {}
for split in ('train', 'valid', 'test'):
    documents = read_jsonl(packet / 'proposals' / (split + '.jsonl'))
    subset = {doc.doc_id: reviews[doc.doc_id] for doc in documents}
    reports[split] = validate_b_evidence(documents, subset,
        expected_split='dev' if split == 'valid' else split,
        synthetic_experiment_dir=packet)
    if {m.action for d in documents for m in d.mentions if keep_eligible(m.entity_type)} != {'MASK', 'KEEP'}:
        raise ValueError(f'{split}: contextual MASK and KEEP candidates are required')
    splits[split] = documents
validate_split_separation(splits)
validate_train_dev(splits['train'], splits['valid'])
input_audit = audit_inputs(load_tokenizer(sys.argv[2]), splits,
    input_version='target_focus_v2', max_length=512)
if not input_audit['passed']:
    raise ValueError('V2 tokenizer input audit failed: ' + json.dumps(input_audit['failures'][:10]))
source_files = ['generation_manifest.json', 'automated_audit.json',
                'a_checkpoint.lock.json', 'review/reviews.jsonl']
source_files += ['proposals/' + split + '.jsonl' for split in splits]
hashes = {name: hashlib.sha256((packet / name).read_bytes()).hexdigest() for name in source_files}
print(json.dumps({'experiment_mode': 'unreviewed_synthetic_experiment',
    'human_reviewed': False, 'training_performed': False,
    'packet': packet_report, 'splits': reports, 'tokenizer_input_audit': input_audit, 'source_files_sha256': hashes},
    ensure_ascii=False, indent=2))
PY
)"
printf '%s\n' "$KPII_PREFLIGHT_JSON"
if [[ "$KPII_CHECK_ONLY" == true ]]; then
  printf '%s\n' 'Check-only completed. No training, inference, GPU selection or output directory creation performed.'
  exit 0
fi

# Only explicit real execution reaches GPU selection. Physical GPU 5 is the only
# device exposed, as logical cuda:0. A is read for initialization and inference.
source scripts/activate_gpu5.sh
mkdir -p -- "$(dirname -- "$KPII_RUN_DIR")"
mkdir -- "$KPII_RUN_DIR"
printf '%s\n' "$KPII_PREFLIGHT_JSON" > "$KPII_RUN_DIR/preflight.json"

"$KPII_PYTHON" - "$KPII_PACKET_DIR" "$KPII_RUN_DIR" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from kpii_mvp.b_data import load_reviews
from kpii_mvp.b_experiment import validate_experiment_packet
from kpii_mvp.data import read_jsonl

packet, run = map(Path, sys.argv[1:])
validate_experiment_packet(packet)
preflight = json.loads((run / 'preflight.json').read_text(encoding='utf-8'))
for name, expected in preflight['source_files_sha256'].items():
    if hashlib.sha256((packet / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Packet changed after preflight: {name}')
reviews = load_reviews(packet / 'review' / 'reviews.jsonl')
data_dir = run / 'data'
data_dir.mkdir()
for split in ('train', 'valid', 'test'):
    source = packet / 'proposals' / (split + '.jsonl')
    shutil.copy2(source, data_dir / (split + '.jsonl'))
    documents = read_jsonl(source)
    with (data_dir / (split + '.reviews.jsonl')).open('w', encoding='utf-8', newline='\n') as stream:
        for document in documents:
            record = reviews[document.doc_id]
            if record['review_status'] != 'PENDING':
                raise ValueError('This experiment requires original PENDING declarations')
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
shutil.copy2(packet / 'a_checkpoint.lock.json', data_dir / 'a_checkpoint.lock.json')
manifest = {'experiment_mode': 'unreviewed_synthetic_experiment', 'human_reviewed': False,
    'review_status': 'PENDING', 'review_records_modified': False,
    'synthetic_experiment_dir': str(packet.resolve()),
    'created_at': datetime.now(timezone.utc).isoformat(),
    'source_files_sha256': preflight['source_files_sha256'],
    'a_frozen': True, 'physical_gpu': 5, 'seed': 42, 'epochs': 3,
    'b_input_version': 'target_focus_v2', 'mask_class_weight': 3.0,
    'label_smoothing': 0.02, 'early_stopping_patience': 1,
    'learning_rate': 1e-5, 'candidate_refinement': 'address-position-v1',
    'limitation': 'Synthetic policy labels; scores do not establish real-document performance.'}
(run / 'experiment_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
PY

KPII_DATA_DIR="$KPII_RUN_DIR/data"
kpii preflight-b \
  --synthetic-experiment-dir "$KPII_PACKET_DIR" \
  --train "$KPII_DATA_DIR/train.jsonl" --train-reviews "$KPII_DATA_DIR/train.reviews.jsonl" \
  --dev "$KPII_DATA_DIR/valid.jsonl" --dev-reviews "$KPII_DATA_DIR/valid.reviews.jsonl" \
  --test "$KPII_DATA_DIR/test.jsonl" --test-reviews "$KPII_DATA_DIR/test.reviews.jsonl" \
  > "$KPII_RUN_DIR/staged-preflight.json"

printf 'Output: %s\nMode: unreviewed_synthetic_experiment\nPhysical GPU: 5 (logical cuda:0)\n' "$KPII_RUN_DIR"
kpii train-b \
  --synthetic-experiment-dir "$KPII_PACKET_DIR" \
  --train "$KPII_DATA_DIR/train.jsonl" --train-reviews "$KPII_DATA_DIR/train.reviews.jsonl" \
  --dev "$KPII_DATA_DIR/valid.jsonl" --dev-reviews "$KPII_DATA_DIR/valid.reviews.jsonl" \
  --encoder-name-or-path "$KPII_A_MODEL_DIR" --output-dir "$KPII_RUN_DIR/model" \
  --device cuda --epochs 3 --max-steps 0 --max-length 512 \
  --batch-size 4 --eval-batch-size 8 --learning-rate 1e-5 --weight-decay 0.01 --seed 42 \
  --input-version target_focus_v2 --mask-class-weight 3.0 \
  --label-smoothing 0.02 --early-stopping-patience 1 \
  2>&1 | tee "$KPII_RUN_DIR/train.log"

KPII_B_MODEL_DIR="$KPII_RUN_DIR/model/best"
kpii_score_split() {
  local KPII_SPLIT_NAME="$1"
  kpii prepare-b-a-input \
    --input "$KPII_DATA_DIR/$KPII_SPLIT_NAME.jsonl" --model-dir "$KPII_A_MODEL_DIR" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-input.jsonl" \
    2>&1 | tee "$KPII_RUN_DIR/$KPII_SPLIT_NAME-prepare.log"
  kpii predict-a \
    --model-dir "$KPII_A_MODEL_DIR" --input "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-input.jsonl" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-raw-candidates.jsonl" \
    --device cuda --batch-size 8 --max-length 512 --stride 128 \
    2>&1 | tee "$KPII_RUN_DIR/$KPII_SPLIT_NAME-predict-a.log"
  kpii refine-a-candidates \
    --input "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-raw-candidates.jsonl" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-candidates.jsonl" \
    --report "$KPII_RUN_DIR/$KPII_SPLIT_NAME-refinement-report.json" \
    2>&1 | tee "$KPII_RUN_DIR/$KPII_SPLIT_NAME-refine.log"
  kpii diagnose-b-candidates \
    --gold "$KPII_DATA_DIR/$KPII_SPLIT_NAME.jsonl" \
    --raw-candidates "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-raw-candidates.jsonl" \
    --candidates "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-candidates.jsonl" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-candidate-diagnostics.json"
  kpii score-b \
    --model-dir "$KPII_B_MODEL_DIR" --input "$KPII_RUN_DIR/$KPII_SPLIT_NAME-a-candidates.jsonl" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-b-scores.jsonl" --device cuda --batch-size 8 \
    2>&1 | tee "$KPII_RUN_DIR/$KPII_SPLIT_NAME-score-b.log"
  # Gold candidates isolate B classification from A boundary errors. These
  # diagnostics never select the checkpoint or the release threshold.
  kpii score-b \
    --model-dir "$KPII_B_MODEL_DIR" --input "$KPII_DATA_DIR/$KPII_SPLIT_NAME.jsonl" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-gold-b-scores.jsonl" --device cuda --batch-size 8 \
    2>&1 | tee "$KPII_RUN_DIR/$KPII_SPLIT_NAME-gold-score-b.log"
  kpii diagnose-b-classification \
    --gold "$KPII_DATA_DIR/$KPII_SPLIT_NAME.jsonl" \
    --scores "$KPII_RUN_DIR/$KPII_SPLIT_NAME-gold-b-scores.jsonl" \
    --model-manifest "$KPII_B_MODEL_DIR/b_manifest.json" \
    --output "$KPII_RUN_DIR/$KPII_SPLIT_NAME-b-classification.json"
}

kpii_score_split valid
kpii calibrate-b \
  --synthetic-experiment-dir "$KPII_PACKET_DIR" \
  --gold "$KPII_DATA_DIR/valid.jsonl" --reviews "$KPII_DATA_DIR/valid.reviews.jsonl" \
  --candidates "$KPII_RUN_DIR/valid-a-candidates.jsonl" --scores "$KPII_RUN_DIR/valid-b-scores.jsonl" \
  --model-manifest "$KPII_B_MODEL_DIR/b_manifest.json" \
  --max-added-leak-rate 0.0 --min-overmask-improvement 0.0 --output "$KPII_RUN_DIR/calibration.json" \
  2>&1 | tee "$KPII_RUN_DIR/calibrate.log"

kpii_score_split test
kpii evaluate-ab \
  --synthetic-experiment-dir "$KPII_PACKET_DIR" \
  --gold "$KPII_DATA_DIR/test.jsonl" --reviews "$KPII_DATA_DIR/test.reviews.jsonl" \
  --candidates "$KPII_RUN_DIR/test-a-candidates.jsonl" --scores "$KPII_RUN_DIR/test-b-scores.jsonl" \
  --model-manifest "$KPII_B_MODEL_DIR/b_manifest.json" --calibration "$KPII_RUN_DIR/calibration.json" \
  --output-dir "$KPII_RUN_DIR/test-evaluation" \
  2>&1 | tee "$KPII_RUN_DIR/evaluate.log"

printf '\nCompleted: %s\nTest report: %s/test-evaluation/ab-evaluation.json\n' "$KPII_RUN_DIR" "$KPII_RUN_DIR"
