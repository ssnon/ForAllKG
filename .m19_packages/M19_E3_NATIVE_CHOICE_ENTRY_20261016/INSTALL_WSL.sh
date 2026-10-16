#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FAKG_M19_INSTALL_REPO:-/home/rldnr7959/FAKG/ForAllKG}"
python - "$PACKAGE" "$REPO" <<'PYINSTALL'
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);repo=Path(sys.argv[2]).resolve()
m=json.loads((p/'PACKAGE_MANIFEST.json').read_text())
if not (repo/'pipeline_core/discovery/scientific_research_control').is_dir():
 raise SystemExit('INVALID_REPO_ROOT')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
for rel,expected in m['expected_existing_shas'].items():
 f=repo/rel
 if not f.is_file() or sha(f)!=expected:
  raise SystemExit('PINNED_DEPENDENCY_MISMATCH: '+rel)
for rel,expected in m['source_shas'].items():
 f=p/'src'/rel; dest=repo/rel
 if not f.is_file() or sha(f)!=expected:
  raise SystemExit('PACKAGE_SOURCE_HASH_MISMATCH: '+rel)
 if dest.exists() or dest.is_symlink():
  raise SystemExit('ADD_ONLY_REFUSES_EXISTING_DESTINATION: '+rel)
for rel in m['new_paths']:
 src=p/'src'/rel;dest=repo/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
 fd=os.open(dest, flags, 0o644)
 with os.fdopen(fd,'wb') as w:w.write(src.read_bytes())
 if sha(dest)!=m['source_shas'][rel]:
  raise SystemExit('WRITTEN_HASH_MISMATCH: '+rel)
print('M19_E3_NATIVE_CHOICE_ENTRY_ADD_ONLY_OK')
print('REPO',repo)
print('ADDED_FILES',len(m['new_paths']))
print('PAID_CALLS 0')
print('W06_W07_W08_E1_E2_UNCHANGED 1')
PYINSTALL
