#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FAKG_REPO_ROOT:-/home/rldnr7959/FAKG/ForAllKG}"
python - "$HERE" "$REPO" <<'PYINSTALL'
from pathlib import Path
import hashlib,json,sys
pkg=Path(sys.argv[1]).resolve(); repo=Path(sys.argv[2]).resolve()
manifest=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
if str(repo)!='/home/rldnr7959/FAKG/ForAllKG' and not (repo/'pipeline_core/discovery/scientific_research_control').is_dir():
    raise SystemExit('REFUSE: INVALID_FAKG_REPO')
if not (repo/'pipeline_core/discovery/scientific_research_control').is_dir():
    raise SystemExit('REFUSE: MISSING_FAKG_DIR')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for rel,digest in manifest['required_existing_source_sha256'].items():
    p=repo/rel
    if not p.is_file() or sha(p)!=digest: raise SystemExit('REFUSE: SOURCE_MISMATCH '+rel)
for rel,digest in manifest['add_only_files_sha256'].items():
    p=pkg/'src'/rel; dest=repo/rel
    if not p.is_file() or sha(p)!=digest: raise SystemExit('REFUSE: PACKAGE_MISMATCH '+rel)
    if dest.exists():raise SystemExit('REFUSE: DESTINATION_EXISTS '+rel)
for rel in manifest['add_only_files_sha256']:
    src=pkg/'src'/rel; dest=repo/rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(src.read_bytes())
print('M19_E2_JSON_V2_ADD_ONLY_INSTALL_OK')
print('REPO',repo)
print('ADDED_FILES',len(manifest['add_only_files_sha256']))
print('PAID_CALLS 0')
print('OLDER_E1_E2_AND_W06_W07_W08_UNCHANGED 1')
PYINSTALL
