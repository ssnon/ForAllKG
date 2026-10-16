#!/usr/bin/env bash
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FAKG_M19_INSTALL_REPO:-/home/rldnr7959/FAKG/ForAllKG}"
python - "$PKG" "$REPO" <<'PYINSTALL'
import json,hashlib,sys,os
from pathlib import Path
pkg=Path(sys.argv[1]).resolve(); repo=Path(sys.argv[2]).resolve()
m=json.loads((pkg/'PACKAGE_MANIFEST.json').read_text())
if not (repo/'pipeline_core/discovery/scientific_research_control').is_dir():
 raise SystemExit('EXPECTED_FAKG_REPO_MISSING')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for rel,h in m['expected_existing_sha256'].items():
 p=repo/rel
 if not p.is_file() or sha(p)!=h:raise SystemExit('BASE_SHA_MISMATCH: '+rel)
for rel,h in m['new_files_sha256'].items():
 src=pkg/'src'/rel;dest=repo/rel
 if not src.is_file() or sha(src)!=h:raise SystemExit('PACKAGE_HASH_MISMATCH: '+rel)
 if dest.exists() or dest.is_symlink():raise SystemExit('ADD_ONLY_REFUSES_EXISTING_FILE: '+rel)
# All checks completed before adding anything; never alter existing files.
for rel in m['new_paths']:
 src=pkg/'src'/rel;dest=repo/rel
 dest.parent.mkdir(parents=True,exist_ok=True)
 fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
 with os.fdopen(fd,'wb') as writer:writer.write(src.read_bytes())
 if sha(dest)!=m['new_files_sha256'][rel]:raise SystemExit('POST_WRITE_MISMATCH: '+rel)
print('M19_E3_SELECTED_WORK_DEEP_ADD_ONLY_OK')
print('REPO',repo)
print('ADDED_FILES',len(m['new_paths']))
print('PAID_CALLS 0')
print('E3_D4_W08_ORIGINALS_UNCHANGED 1')
print('NATIVE_WORK_EVENT_AUTO_COMMIT 0')
PYINSTALL
