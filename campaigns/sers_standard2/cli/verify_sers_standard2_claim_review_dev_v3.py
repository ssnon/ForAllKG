from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.claim_review_dev_validation import (
    load_inputs,
    render_human_audit,
)
from campaigns.sers_standard2.claim_review_dev_validation_v3 import (
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    canonical_json,
    offline_recompile_from_report,
    read_json,
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
        repo_root=ROOT,
        spec_path=spec_root / "claim_review_spec_v3.json",
    )
    if spec_issues:
        print("claim-review-only DEV v3 verification: FAIL")
        for issue in spec_issues:
            print(" -", issue)
        return 2

    report_path = run_root / "claim_review_report_v3.json"
    audit_path = run_root / "human_relationship_audit_v3.md"
    marker_path = run_root / "STRUCTURAL_PASS.json"
    issues = []
    for path in (report_path, audit_path, marker_path):
        if not path.is_file():
            issues.append(f"required artifact missing: {path.name}")
    if issues:
        print("claim-review-only DEV v3 verification: FAIL")
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
        "sers_standard2_claim_review_only_dev_run_v3:"
        + observed[:20]
    ):
        issues.append("run ID mismatch")

    required_false = {
        "ranker_recomputed": False,
        "compiler_changed_from_v2": False,
        "invalid_id_guess_mapping_used": False,
        "case_specific_expected_statuses_used": False,
        "hypothesis_level_novelty_status_computed": False,
        "automatic_next_stage_authorized": False,
        "fresh_reserve_consumed": False,
    }
    for key, expected in required_false.items():
        if report.get(key) is not expected:
            issues.append(f"{key} policy violation")

    if report.get("structural_outcome") != (
        "CLAIM_REVIEW_V3_STRUCTURAL_DEV_PASS"
    ):
        issues.append("structural outcome is not PASS")
    if report.get("scientific_relationship_outcome") != (
        "MANUAL_REVIEW_REQUIRED"
    ):
        issues.append("scientific relationship outcome overclaimed")
    if report.get("literature_network_calls") != 0:
        issues.append("unexpected literature network calls")
    if report.get("secret_scan_pass") is not True:
        issues.append("secret scan is not PASS")

    checks = report.get("checks", {})
    for key in (
        "reviewer_work_ids_within_frozen_topn",
        "compiler_unknown_work_ids_empty",
        "allowed_work_id_block_verified_in_every_prompt",
        "reviewer_match_ids_unique_per_claim",
        "reviewer_match_count_not_exceed_candidates",
    ):
        if checks.get(key) is not True:
            issues.append(f"required provenance check failed: {key}")

    try:
        reviews, drafts = offline_recompile_from_report(
            repo_root=ROOT,
            spec=spec,
            report=report,
        )
        stored_reviews = report.get("claim_reviews")
        recomputed = [
            review.model_dump(mode="json")
            for review in reviews
        ]
        if canonical_json(stored_reviews) != canonical_json(recomputed):
            issues.append("offline compiler recomputation mismatch")

        _plan, _packet, _ranker_spec, ranker_report = load_inputs(ROOT)
        expected_md = render_human_audit(
            reviews=reviews,
            drafts=drafts,
            ranker_report=ranker_report,
        )
        if audit_path.read_text(encoding="utf-8") != expected_md:
            issues.append("human relationship audit mismatch")
    except Exception as exc:
        issues.append(
            f"offline verification failed: "
            f"{type(exc).__name__}: {exc}"
        )

    if marker.get("status") != "structural_pass":
        issues.append("STRUCTURAL_PASS marker mismatch")
    if marker.get("run_id") != report.get("run_id"):
        issues.append("STRUCTURAL_PASS run ID mismatch")

    if issues:
        print("claim-review-only DEV v3 verification: FAIL")
        for issue in sorted(set(issues)):
            print(" -", issue)
        print("LLM calls during verification:", 0)
        return 2

    print("claim-review-only DEV v3 verification: PASS")
    print("Spec ID:", spec["spec_id"])
    print("Run ID:", report["run_id"])
    print(
        "Compiled status counts:",
        report["diagnostics"]["compiled_status_counts"],
    )
    print(
        "Compiled relationship counts:",
        report["diagnostics"]["compiled_relationship_counts"],
    )
    print("Reviewer work IDs within frozen top-N:", True)
    print("Reviewer work IDs unique per claim:", True)
    print("Allowed-ID prompt block verified:", True)
    print("LLM calls during verification:", 0)
    print("Literature network calls during verification:", 0)
    print("Ranker recomputed:", False)
    print("Compiler changed:", False)
    print("Hypothesis-level novelty verdict:", False)
    print("Fresh Reserve consumed:", False)
    print("Automatic next stage authorized:", False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
