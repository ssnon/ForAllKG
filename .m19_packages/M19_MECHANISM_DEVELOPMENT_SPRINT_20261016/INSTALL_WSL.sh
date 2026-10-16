#!/usr/bin/env bash
set -euo pipefail
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$PWD"
if [[ ! -d "$REPO/pipeline_core/discovery/scientific_research_control" || ! -f "$REPO/pipeline_core/discovery/scientific_research_control/research_work_v12_stable_runner.py" || ! -f "$REPO/pipeline_core/discovery/scientific_research_control/research_work_v12_contracts.py" ]]; then
  echo "ERROR: Run from ForAllKG repository root containing the local stable Research Work runner" >&2; exit 1
fi
python - "$PKG_DIR" "$REPO" <<'PYINSTALL'
import json,hashlib,sys,shutil
from pathlib import Path
p,r=map(Path,sys.argv[1:])
m=json.loads((p/'MANIFEST.json').read_text())
assert m['schema']=='m19-mechanism-science-only-add-only-v1'
for rel,h in m['files'].items():
    src=p/rel
    if not src.is_file() or src.is_symlink() or hashlib.sha256(src.read_bytes()).hexdigest()!=h:
        raise RuntimeError('MISSING_OR_TAMPERED_PACKAGE_SOURCE '+rel)
    dst=r/rel
    if dst.exists() or dst.is_symlink():
        raise RuntimeError('DESTINATION_EXISTS_REFUSE_OVERWRITE '+rel)
for rel,h in m['files'].items():
    dst=r/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('xb') as f:f.write((p/rel).read_bytes())
    if hashlib.sha256(dst.read_bytes()).hexdigest()!=h:raise RuntimeError('INSTALLED_SHA_MISMATCH '+rel)
print('M19_MECHANISM_SPRINT_ADD_ONLY_INSTALLED')
print('REPO',r)
print('ADDED_FILES',len(m['files']))
print('PAID_CALLS',0)
print('OLD_EPISODE_FILES_CHANGED',0)
PYINSTALL
