#!/usr/bin/env bash
# Non-destructive, add-only installation under the requested FAKG checkout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FAKG_REPO_ROOT:-/home/rldnr7959/FAKG/ForAllKG}"
python - "$HERE" "$REPO" <<'PY'
import hashlib,json,sys
from pathlib import Path
pkg=Path(sys.argv[1]).resolve();repo=Path(sys.argv[2]).resolve()
manifest=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text())
if not repo.is_dir() or not (repo/'pipeline_core/discovery/scientific_research_control/research_work_v18_feedback_runner.py').is_file():
    raise SystemExit('REFUSE: not an existing ForAllKG v1.8 checkout: '+str(repo))
# Complete preflight, no destination write until every prerequisite has passed.
for rel,sha in manifest['required_existing_source_sha256'].items():
    path=repo/rel
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
        raise SystemExit('REFUSE: installed v1.8 baseline changed: '+rel)
for rel,sha in manifest['add_only_files_sha256'].items():
    source=pkg/'src'/rel;dest=repo/rel
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=sha:
        raise SystemExit('REFUSE: package payload hash mismatch: '+rel)
    if dest.exists():raise SystemExit('REFUSE: add-only destination exists: '+str(dest))
for rel in manifest['add_only_files_sha256']:
    source=pkg/'src'/rel;dest=repo/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as stream:stream.write(source.read_bytes())
print('M19_E1_ADD_ONLY_INSTALL_OK')
print('REPO',repo)
print('ADDED_FILES',len(manifest['add_only_files_sha256']))
print('PAID_CALLS 0')
print('EXISTING_W06_W07_W08_UNCHANGED 1')
print('NEXT: run offline tests and read-only prepare before any paid work')
PY
