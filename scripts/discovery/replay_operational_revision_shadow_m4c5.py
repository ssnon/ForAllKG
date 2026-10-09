"""M4-C5 offline counterfactual ResearchIdeaKernel draft replay (no mutation/API)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.operational_revision_shadow_m4c5 import execute_shadow, render_report


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, m4c4: Path, m3c1: Path, expected_m4c4_sha256: str,
        expected_m3c1_sha256: str, output_dir: Path) -> dict:
    paths = [p.expanduser().resolve() for p in (m4c4, m3c1)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    sha4, sha3 = [digest(p) for p in paths]
    if sha4 != expected_m4c4_sha256 or sha3 != expected_m3c1_sha256:
        raise ValueError("M4C5_INTEGRITY_FAILURE: pinned source SHA mismatch")
    root = Path(__file__).resolve().parents[2]
    target = output_dir.expanduser().resolve()
    if target == root or root in target.parents:
        raise ValueError("M4C5_INTEGRITY_FAILURE: output directory inside repository")
    if target.exists():
        raise FileExistsError("refusing to overwrite existing output: " + str(target))
    if any(target == p or target in p.parents or p in target.parents for p in paths):
        raise ValueError("M4C5_INTEGRITY_FAILURE: source/output overlap")
    source = []
    for path in paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("M4C5_INTEGRITY_FAILURE: source must be JSON object")
        source.append(obj)
    result = execute_shadow(m4c4=source[0], trajectories=source[1],sha_m4c4=sha4,sha_traces=sha3)
    target.mkdir(parents=True, exist_ok=False)
    (target / "M4C5_OPERATIONAL_REVISION_SHADOW_PRIVATE.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    (target / "M4C5_REPORT_PRIVATE.md").write_text(render_report(result), encoding="utf-8")
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4c4-report",type=Path,required=True)
    parser.add_argument("--m3c1-trajectories",type=Path,required=True)
    parser.add_argument("--expected-m4c4-sha256",required=True)
    parser.add_argument("--expected-m3c1-sha256",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    try:
        report=run(m4c4=args.m4c4_report,m3c1=args.m3c1_trajectories,
                   expected_m4c4_sha256=args.expected_m4c4_sha256,
                   expected_m3c1_sha256=args.expected_m3c1_sha256,output_dir=args.output_dir)
    except (ValueError,FileNotFoundError,FileExistsError,KeyError,TypeError) as error:
        parser.exit(2,"M4C5 STOPPED (no source modified): "+str(error)+"\n")
    print("M4C5:",report["status"])
    print("kernel drafts:",report["kernel_drafts"],"abstentions:",report["abstentions"])
    print("research idea nodes created:",report["new_research_idea_nodes_created"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
