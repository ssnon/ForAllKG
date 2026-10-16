#!/usr/bin/env bash
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FAKG_M19_INSTALL_REPO:-/home/rldnr7959/FAKG/ForAllKG}"
python - "$PKG" "$REPO" <<'PYINSTALL'
import json,hashlib,sys,os
from pathlib import Path
pkg=Path(sys.argv[1]).resolve()
repo=Path(sys.argv[2]).resolve()
m=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text())
if not (repo/'pipeline_core/discovery/scientific_research_control').is_dir():
 raise SystemExit('EXPECTED_FAKG_REPO_MISSING')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for rel,h in m['expected_existing_sha256'].items():
 p=repo/rel
 if not p.is_file() or sha(p)!=h:raise SystemExit('BASE_SHA_MISMATCH: '+rel)
for rel,h in m['new_files_sha256'].items():
 src=pkg/'src'/rel;dst=repo/rel
 if not src.is_file() or sha(src)!=h:raise SystemExit('PACKAGE_HASH_MISMATCH: '+rel)
 if dst.exists() or dst.is_symlink():raise SystemExit('ADD_ONLY_REFUSES_EXISTING_FILE: '+rel)
for rel in m['new_paths']:
 src=pkg/'src'/rel;dst=repo/rel
 dst.parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'wb') as writer:writer.write(src.read_bytes())
 if sha(dst)!=m['new_files_sha256'][rel]:raise SystemExit('POST_WRITE_MISMATCH: '+rel)
print('M19_E4B_TARGET_ONLY_ADD_ONLY_OK')
print('REPO',repo)
print('ADDED_FILES',len(m['new_paths']))
print('INSTALL_PAID_CALLS 0')
print('E3_RETURN_AND_FINALIZED_E4_UNCHANGED 1')
print('TARGET_MODEL_RECEIVES_AUDIT_PROOF 0')
PYINSTALL
