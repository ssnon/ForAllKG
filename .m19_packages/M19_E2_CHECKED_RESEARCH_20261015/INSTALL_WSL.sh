#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FAKG_REPO_ROOT:-/home/rldnr7959/FAKG/ForAllKG}"
python - "$HERE" "$REPO" <<'PYINSTALL'
from pathlib import Path
import hashlib,json,sys
pkg=Path(sys.argv[1]).resolve();repo=Path(sys.argv[2]).resolve()
manifest=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
if not (repo/'pipeline_core/discovery/scientific_research_control').is_dir():
    raise SystemExit('REFUSE: not a FAKG checkout')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,digest in manifest['required_existing_source_sha256'].items():
    p=repo/rel
    if not p.is_file() or sha(p)!=digest:raise SystemExit('REFUSE: SOURCE_MISMATCH '+rel)
for rel,digest in manifest['add_only_files_sha256'].items():
    p=pkg/'src'/rel;dest=repo/rel
    if not p.is_file() or sha(p)!=digest:raise SystemExit('REFUSE: PACKAGE_MISMATCH '+rel)
    if dest.exists():raise SystemExit('REFUSE: DESTINATION_EXISTS '+rel)
for rel in manifest['add_only_files_sha256']:
    src=pkg/'src'/rel;dest=repo/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(src.read_bytes())
print('M19_E2_CHECKED_RESEARCH_ADD_ONLY_OK')
print('REPO',repo)
print('ADDED_FILES',len(manifest['add_only_files_sha256']))
print('PAID_CALLS 0')
print('W06_W07_W08_E1_UNCHANGED 1')
PYINSTALL
