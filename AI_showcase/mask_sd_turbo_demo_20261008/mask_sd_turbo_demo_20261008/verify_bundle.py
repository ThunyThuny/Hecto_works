#!/usr/bin/env python3
"""배포 파일 SHA256 검사. Python 표준 라이브러리만 사용합니다."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root/'SHA256.json').read_text())
failures = []
for name, expected in manifest['files'].items():
    p = root/name
    if not p.is_file():
        failures.append(name+': missing')
        continue
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    if h.hexdigest() != expected['sha256'] or p.stat().st_size != expected['bytes']:
        failures.append(name+': mismatch')
print(json.dumps({'checked_files':len(manifest['files']), 'failures':failures,
                  'verified':not failures},ensure_ascii=False,indent=2))
raise SystemExit(1 if failures else 0)
