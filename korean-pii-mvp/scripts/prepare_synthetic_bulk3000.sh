#!/usr/bin/env bash
# Generate and bind a separate PENDING dataset. No model/GPU/training invocation.
set -euo pipefail
cd /workspace/korean-pii-mvp
source /workspace/kpii-env/bin/activate
export CUDA_VISIBLE_DEVICES=''
KPII_BULK_OUTPUT="${1:-/workspace/nfs/datasets/dylee/PII/synthetic_data_bulk3000_v3}"
python - "$KPII_BULK_OUTPUT" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
from kpii_mvp.synthetic_bulk import generate
from kpii_mvp.synthetic_flow import FIXED_A_DIR,verify_a_lock
original=Path('/workspace/nfs/datasets/dylee/PII/synthetic_data').resolve()
destination=Path(sys.argv[1]).absolute()
archive=original.parent/'synthetic_data_archive'
if destination.resolve().is_relative_to(original) or destination.resolve().is_relative_to(archive):
    raise ValueError('Use a separate dataset directory outside existing pilot/archive')
if destination.resolve().is_relative_to(Path(FIXED_A_DIR).resolve()):
    raise ValueError('Data output cannot be inside the fixed checkpoint')
def snapshot(directory):
    return {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}
before=snapshot(original)
before_a=verify_a_lock(original/'a_checkpoint.lock.json',FIXED_A_DIR)
before_runs=sorted(str(p) for p in Path('runs').glob('b-*'))
summary=generate(destination,seed=20261021)
shutil.copyfile(original/'a_checkpoint.lock.json',destination/'a_checkpoint.lock.json')
assert snapshot(original)==before
assert verify_a_lock(destination/'a_checkpoint.lock.json',FIXED_A_DIR)==before_a
assert sorted(str(p) for p in Path('runs').glob('b-*'))==before_runs
preservation={'existing_50_dir':str(original),'original_files_sha256':before,
              'original_files_unchanged':True,'fixed_a':before_a,
              'b_runs_unchanged':True,'training_executed':False}
(destination/'preservation_record.json').write_text(json.dumps(preservation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('Original pilot and A preserved. All reviews PENDING. No training executed.')
PY
