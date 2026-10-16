#!/usr/bin/env bash
# Add-only source-locked offline reference-quarantine patch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FAKG_REPO_ROOT:-/home/rldnr7959/FAKG/ForAllKG}"
python - "$HERE" "$REPO" <<'PY'
import hashlib,json,sys
from pathlib import Path
pkg=Path(sys.argv[1]).resolve();repo=Path(sys.argv[2]).resolve()
manifest=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text())
if not repo.is_dir() or not (repo/'pipeline_core/discovery/scientific_research_control/m19_deep_work_e1.py').is_file():
    raise SystemExit('REFUSE: not an installed M19 E1 repo: '+str(repo))
for rel,sha in manifest['required_existing_source_sha256'].items():
    p=repo/rel
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
        raise SystemExit('REFUSE: existing code changed: '+rel)
for rel,sha in manifest['add_only_files_sha256'].items():
    src=pkg/'src'/rel;dest=repo/rel
    if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest()!=sha:
        raise SystemExit('REFUSE: package hash mismatch: '+rel)
    if dest.exists():raise SystemExit('REFUSE: destination exists: '+str(dest))
for rel in manifest['add_only_files_sha256']:
    src=pkg/'src'/rel;dest=repo/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(src.read_bytes())
print('M19_E1_REFERENCE_RECOVERY_ADD_ONLY_OK')
print('REPO',repo)
print('ADDED_FILES',len(manifest['add_only_files_sha256']))
print('PAID_CALLS 0')
print('W06_W07_W08_AND_EXISTING_M19_RUNNER_UNCHANGED 1')
PY
