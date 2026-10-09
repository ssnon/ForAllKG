"""Build M4-C2 hypothetical, non-authoritative kernel drafts from frozen Q-A."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.research_idea_revision_m4c2 import build_revision_replay, render_report


def load(path: Path):
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("M4C2_INTEGRITY_FAILURE: expected JSON object: "+str(path))
    return value


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, a1: Path, c0: Path, c1: Path, trajectories: Path,
        expected_trajectories_sha256: str, output_dir: Path):
    paths=[a1,c0,c1,trajectories]
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(str(p))
    sha=[digest(p) for p in paths]
    if sha[3] != expected_trajectories_sha256:
        raise ValueError("M4C2_INTEGRITY_FAILURE: M3C1 pinned SHA-256 mismatch")
    root=Path(__file__).resolve().parents[2]
    dest=output_dir.expanduser().resolve()
    if dest == root or root in dest.parents:
        raise ValueError("M4C2_INTEGRITY_FAILURE: output must remain outside repository")
    if dest.exists():
        raise FileExistsError("refusing to overwrite output: "+str(dest))
    if any(p.expanduser().resolve()==dest or dest in p.expanduser().resolve().parents for p in paths):
        raise ValueError("M4C2_INTEGRITY_FAILURE: output overlaps input")
    report=build_revision_replay(a1=load(a1),c0=load(c0),c1=load(c1),
        trajectories=load(trajectories),sha_a1=sha[0],sha_c0=sha[1],sha_c1=sha[2],sha_trajectories=sha[3])
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'M4C2_KERNEL_DRAFTS_PRIVATE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    (dest/'M4C2_REPORT_PRIVATE.md').write_text(render_report(report),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m4a1-report',type=Path,required=True)
    p.add_argument('--m4c0-replay',type=Path,required=True)
    p.add_argument('--m4c1-audit',type=Path,required=True)
    p.add_argument('--m3c1-trajectories',type=Path,required=True)
    p.add_argument('--expected-m3c1-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    try:
        r=run(a1=args.m4a1_report,c0=args.m4c0_replay,c1=args.m4c1_audit,
            trajectories=args.m3c1_trajectories,expected_trajectories_sha256=args.expected_m3c1_sha256,output_dir=args.output_dir)
    except (ValueError,FileNotFoundError,FileExistsError,KeyError) as exc:
        p.exit(2,'M4C2 STOPPED (no source modified): '+str(exc)+'\n')
    print('M4C2:',r['status'])
    print('synthetic scenarios:',r['scenario_count'],'draft kernels:',r['draft_kernel_count'],'no revision:',r['no_revision_count'])
    print('scientific learning certified: false')

if __name__=='__main__':
    main()
