#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[[ "$SCRIPT_DIR" != "${BASH_SOURCE[0]}" ]] || SCRIPT_DIR=.
cd -- "$SCRIPT_DIR"
PYTHON="${PYTHON:-python}"

case "${1:---help}" in
  --help|-h)
    printf '%s\n' 'Prepare the frozen-feature archives used by the main experiments.

  bash datasets.sh --from /path/to/prepared/features [datasets]
  bash datasets.sh --check [datasets]

The source directory must contain retrieval/ and classification/ archives in
the schema documented in README.md. Existing destination files are preserved;
conflicting files cause an error. No raw datasets or model weights are stored
in this repository. A public feature-download URL has not been released yet.'
    ;;
  --from)
    [[ $# -ge 2 && $# -le 3 ]] || { echo "Usage: bash datasets.sh --from SOURCE [DEST]" >&2; exit 2; }
    "$PYTHON" - "$2" "${3:-datasets}" <<'PY'
import hashlib
import shutil
import sys
from pathlib import Path
from run import inventory, load_features

source, destination = (Path(value).expanduser().resolve() for value in sys.argv[1:])
pending = []
for task, backbone, relation in inventory():
    relative = Path(task) / backbone / (relation + '.npz')
    src, dst = source / relative, destination / relative
    if not src.is_file():
        raise SystemExit(f'Missing source archive: {src}')
    load_features(src, task)
    if dst.exists() and src != dst:
        def digest(path):
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                    h.update(chunk)
            return h.digest()
        if digest(src) != digest(dst):
            raise SystemExit(f'Conflicting destination file: {dst}')
    pending.append((src, dst))
for src, dst in pending:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
print(f'Prepared {len(pending)} main-experiment archives in {destination}')
PY
    ;;
  --check)
    [[ $# -le 2 ]] || { echo "Usage: bash datasets.sh --check [DEST]" >&2; exit 2; }
    "$PYTHON" run.py --check-data --data-root "${2:-datasets}"
    ;;
  *) echo "Unknown option: $1. Run bash datasets.sh --help." >&2; exit 2 ;;
esac
