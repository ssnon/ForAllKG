"""Read-only offline M4-C4 B03/B06 operational estimand demonstration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.conditional_covariance_confrontation_m4c4 import (
    render_report, run,
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def execute(*, m4a1: Path, m4c2: Path, m4c3: Path,
            expected_m4a1_sha256: str, output_dir: Path) -> dict:
    sources = [p.expanduser().resolve() for p in (m4a1, m4c2, m4c3)]
    for p in sources:
        if not p.is_file():
            raise FileNotFoundError(f'M4C4 source not found: {p}')
    root = Path(__file__).resolve().parents[2]
    dest = output_dir.expanduser().resolve()
    if dest == root or root in dest.parents:
        raise ValueError('M4C4_INTEGRITY_FAILURE: output must stay outside repo')
    if dest.exists():
        raise FileExistsError('refusing to overwrite: ' + str(dest))
    if any(dest == p or dest in p.parents or p in dest.parents for p in sources):
        raise ValueError('M4C4_INTEGRITY_FAILURE: source/output path overlap')
    hashes = [_sha(p) for p in sources]
    if hashes[0] != expected_m4a1_sha256:
        raise ValueError('M4C4_INTEGRITY_FAILURE: M4-A1 SHA mismatch')
    documents = []
    for p in sources:
        data = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('M4C4_INTEGRITY_FAILURE: source root not JSON object')
        documents.append(data)
    result = run(a1=documents[0], c2=documents[1], c3=documents[2],
                 sha_a1=hashes[0], sha_c2=hashes[1], sha_c3=hashes[2])
    dest.mkdir(parents=True, exist_ok=False)
    (dest / 'M4C4_OPERATIONAL_CONFRONTATION_PRIVATE.json').write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (dest / 'M4C4_REPORT_PRIVATE.md').write_text(render_report(result), encoding='utf-8')
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m4a1-report', type=Path, required=True)
    p.add_argument('--m4c2-report', type=Path, required=True)
    p.add_argument('--m4c3-audit', type=Path, required=True)
    p.add_argument('--expected-m4a1-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args=p.parse_args()
    try:
        r=execute(m4a1=args.m4a1_report,m4c2=args.m4c2_report,
                  m4c3=args.m4c3_audit,expected_m4a1_sha256=args.expected_m4a1_sha256,
                  output_dir=args.output_dir)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, TypeError) as e:
        p.exit(2, f'M4C4 STOPPED (source unmodified): {e}\n')
    print('M4C4:',r['status'])
    print('case:',r['case_id'],'synthetic fixtures:',len(r['synthetic_fixtures']))
    print('empirical scientific improvement certified:', r['scientific_improvement_certified'])
    print('report:', args.output_dir / 'M4C4_REPORT_PRIVATE.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
