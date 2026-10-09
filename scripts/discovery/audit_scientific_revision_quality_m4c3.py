"""Read-only M4-C3 scientific kernel revision quality review; no API calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from pipeline_core.discovery.scientific_revision_quality_m4c3 import (
    evaluate_quality, render_report,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("M4C3_INTEGRITY_FAILURE: expected JSON object")
    return data


def run(*, m4c2: Path, m4a1: Path, m3c1: Path, output_dir: Path,
        expected_m3c1_sha256: str, expected_m4a1_sha256: str) -> dict:
    paths = [p.expanduser().resolve() for p in (m4c2, m4a1, m3c1)]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(str(p))
    checksums = [digest(p) for p in paths]
    if checksums[1] != expected_m4a1_sha256 or checksums[2] != expected_m3c1_sha256:
        raise ValueError("M4C3_INTEGRITY_FAILURE: source SHA pin mismatch")
    root = Path(__file__).resolve().parents[2]
    target = output_dir.expanduser().resolve()
    if target == root or root in target.parents:
        raise ValueError("M4C3_INTEGRITY_FAILURE: output must stay outside repository")
    if target.exists():
        raise FileExistsError("refusing to overwrite output: " + str(target))
    if any(target == p or target in p.parents or p in target.parents for p in paths):
        raise ValueError("M4C3_INTEGRITY_FAILURE: source/output path overlap")
    result = evaluate_quality(c2=read(paths[0]), a1=read(paths[1]),
                              trajectories=read(paths[2]), sha_c2=checksums[0],
                              sha_a1=checksums[1], sha_traces=checksums[2])
    target.mkdir(parents=True, exist_ok=False)
    (target / "M4C3_QUALITY_AUDIT_PRIVATE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (target / "M4C3_REPORT_PRIVATE.md").write_text(render_report(result), encoding="utf-8")
    with (target / "M4C3_EXPERT_REVIEW_TEMPLATE_PRIVATE.csv").open("w", encoding="utf-8", newline="") as f:
        columns = ["case_id", "synthetic_scenario", "draft_id", "mechanism_specific_delta",
                   "discriminating_estimand", "independent_measurement_path", "confound_controls",
                   "alternative_explanations", "scope_of_possible_falsifier",
                   "separate_parent_programs_preserved", "reviewer", "review_status"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in result["rows"]:
            if row["draft_id"]:
                writer.writerow({"case_id": row["case_id"], "synthetic_scenario": row["scenario"],
                                 "draft_id": row["draft_id"], "review_status": "UNREVIEWED"})
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m4c2-report", type=Path, required=True)
    p.add_argument("--m4a1-report", type=Path, required=True)
    p.add_argument("--m3c1-trajectories", type=Path, required=True)
    p.add_argument("--expected-m3c1-sha256", required=True)
    p.add_argument("--expected-m4a1-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        result = run(m4c2=args.m4c2_report, m4a1=args.m4a1_report,
                     m3c1=args.m3c1_trajectories, output_dir=args.output_dir,
                     expected_m3c1_sha256=args.expected_m3c1_sha256,
                     expected_m4a1_sha256=args.expected_m4a1_sha256)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, TypeError) as exc:
        p.exit(2, "M4C3 STOPPED (no source modified): " + str(exc) + "\n")
    print("M4C3:", result["status"])
    print("drafts needing expert review:", result["review_required_count"])
    print("scientific improvements certified:", result["substantive_revisions_scientifically_confirmed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
