from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dac_her.hypothesis_novelty_status_lattice_v2 import (
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    canonical_json,
    compute_real_synthesis,
    load_frozen_inputs,
    read_json,
    render_human_audit,
    sha256_json,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    args = parser.parse_args()

    spec_root = (
        args.spec_root if args.spec_root.is_absolute()
        else ROOT / args.spec_root
    )
    run_root = (
        args.run_root if args.run_root.is_absolute()
        else ROOT / args.run_root
    )

    spec_issues, spec = verify_spec(
        ROOT,
        spec_root / "status_lattice_v2_spec.json",
    )
    if spec_issues:
        print("status-lattice v2 verification: FAIL")
        for issue in spec_issues:
            print(" -", issue)
        return 2

    report_path = run_root / "status_lattice_v2_report.json"
    audit_path = run_root / "human_status_lattice_v2_audit.md"
    marker_path = run_root / "STRUCTURAL_PASS.json"
    issues = []
    for path in (report_path, audit_path, marker_path):
        if not path.is_file():
            issues.append(f"required artifact missing: {path.name}")
    if issues:
        print("status-lattice v2 verification: FAIL")
        for issue in issues:
            print(" -", issue)
        return 2

    report = read_json(report_path)
    marker = read_json(marker_path)
    body = dict(report)
    run_id = body.pop("run_id", None)
    run_sha = body.pop("run_sha256", None)
    observed = sha256_json(body)
    if run_sha != observed:
        issues.append("run SHA mismatch")
    if run_id != (
        "sers_hypothesis_novelty_status_lattice_v2_run:"
        + observed[:20]
    ):
        issues.append("run ID mismatch")

    plan, packet, reviews, _ = load_frozen_inputs(ROOT)
    recomputed = compute_real_synthesis(
        plan=plan,
        packet=packet,
        reviews=reviews,
    )
    if canonical_json(recomputed) != canonical_json(
        report.get("hypothesis_rows")
    ):
        issues.append("offline synthesis recomputation mismatch")

    counts = dict(
        sorted(Counter(row["status"] for row in recomputed).items())
    )
    if report.get("status_counts") != counts:
        issues.append("status-count mismatch")
    if audit_path.read_text(encoding="utf-8") != (
        render_human_audit(recomputed)
    ):
        issues.append("human audit mismatch")

    if report.get("structural_outcome") != (
        "HYPOTHESIS_NOVELTY_STATUS_LATTICE_V2_STRUCTURAL_PASS"
    ):
        issues.append("structural outcome is not PASS")
    if report.get("scientific_novelty_status_outcome") != (
        "MANUAL_REVIEW_REQUIRED"
    ):
        issues.append("scientific outcome overclaimed")
    if marker.get("status") != "structural_pass":
        issues.append("STRUCTURAL_PASS marker mismatch")
    if marker.get("run_id") != report.get("run_id"):
        issues.append("STRUCTURAL_PASS run ID mismatch")

    for key in (
        "ranker_recomputed",
        "claim_reviewer_recomputed",
        "coverage_policy_changed",
        "case_specific_sers_rules_used",
        "automatic_scientific_status_approval",
        "automatic_next_stage_authorized",
        "fresh_reserve_consumed",
    ):
        if report.get(key) is not False:
            issues.append(f"policy violation: {key}")
    if report.get("llm_calls") != 0:
        issues.append("unexpected LLM calls")
    if report.get("network_calls") != 0:
        issues.append("unexpected network calls")

    if issues:
        print("status-lattice v2 verification: FAIL")
        for issue in sorted(set(issues)):
            print(" -", issue)
        print("LLM calls during verification:", 0)
        print("Network calls during verification:", 0)
        return 2

    print("status-lattice v2 verification: PASS")
    print("Spec ID:", spec["spec_id"])
    print("Run ID:", report["run_id"])
    print("Status counts:", report["status_counts"])
    for index, row in enumerate(recomputed, start=1):
        print(
            f"[{index}/3]",
            row["hypothesis_id"],
            "=>", row["status"],
            "| core=", row["core_claim_statuses"],
            "| coverage_sufficient=",
            row["coverage"][
                "sufficient_for_absence_based_novelty"
            ],
        )
    print("Scientific novelty-status outcome: MANUAL_REVIEW_REQUIRED")
    print("LLM calls during verification:", 0)
    print("Network calls during verification:", 0)
    print("Fresh Reserve consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
