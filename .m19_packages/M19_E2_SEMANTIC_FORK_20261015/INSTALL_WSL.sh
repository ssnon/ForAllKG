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
    raise SystemExit('REFUSE: MISSING_FAKG_REPO')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,digest in manifest['required_existing_source_sha256'].items():
    p=repo/rel
    if not p.is_file() or sha(p)!=digest:raise SystemExit('REFUSE: EXISTING_SOURCE_MISMATCH '+rel)
for rel,digest in manifest['add_only_files_sha256'].items():
    src=pkg/'src'/rel;dst=repo/rel
    if not src.is_file() or sha(src)!=digest:raise SystemExit('REFUSE: PACKAGE_MISMATCH '+rel)
    if dst.exists():raise SystemExit('REFUSE: DESTINATION_EXISTS '+rel)
for rel in manifest['add_only_files_sha256']:
    src=pkg/'src'/rel;dst=repo/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('xb') as f:f.write(src.read_bytes())
print('M19_E2_SEMANTIC_FORK_ADD_ONLY_OK')
print('REPO',repo)
print('ADDED_FILES',len(manifest['add_only_files_sha256']))
print('PAID_CALLS 0')
print('ORIGINAL_E2_E1_W06_W07_W08_UNCHANGED 1')
PYINSTALL
