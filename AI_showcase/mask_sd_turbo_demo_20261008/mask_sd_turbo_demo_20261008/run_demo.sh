#!/usr/bin/env bash
set -euo pipefail
DEMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DEMO_PYTHON:-}" ]]; then
  DEMO_PYTHON=python3
fi
exec "$DEMO_PYTHON" "$DEMO_ROOT/demo.py" "$@"
