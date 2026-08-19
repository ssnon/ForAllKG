from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.canonicalization_dev_validation import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    verify_run,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_ROOT,
    )
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

    diagnostic_root = (
        args.diagnostic_root
        .expanduser()
        .resolve()
    )
    spec_root = (
        args.spec_root
        if args.spec_root.is_absolute()
        else ROOT / args.spec_root
    )
    run_root = (
        args.run_root
        if args.run_root.is_absolute()
        else ROOT / args.run_root
    )

    spec_issues, spec = verify_spec(
        repo_root=ROOT,
        diagnostic_root=diagnostic_root,
        spec_path=(
            spec_root
            / "canonicalization_spec.json"
        ),
    )
    if spec_issues:
        print(
            "canonicalization DEV verification: FAIL"
        )
        for issue in spec_issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    run_issues, report = verify_run(
        spec=spec,
        raw_packet_path=(
            run_root
            / "raw_prior_art.json"
        ),
        canonical_packet_path=(
            run_root
            / "canonical_prior_art.json"
        ),
        report_path=(
            run_root
            / "canonicalization_report.json"
        ),
    )
    if run_issues:
        print(
            "canonicalization DEV verification: FAIL"
        )
        for issue in run_issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    print(
        "canonicalization DEV verification: PASS"
    )
    print(
        "Spec ID:",
        spec["spec_id"],
    )
    print(
        "Run ID:",
        report["run_id"],
    )
    print(
        "Outcome:",
        report["outcome"],
    )
    print(
        "Checks:",
        report["checks"],
    )
    print(
        "Counts:",
        report["counts"],
    )
    print(
        "Provenance:",
        report[
            "provenance_diagnostics"
        ],
    )
    print(
        "Title/DOI collision groups:",
        report["counts"][
            "title_cross_doi_collision_group_count"
        ],
    )
    print(
        "Secret values persisted:",
        report["secret_scan"][
            "secret_values_persisted"
        ],
    )
    print("Ranker used:", False)
    print("Claim review used:", False)
    print(
        "Scientific interpretation:",
        False,
    )
    print(
        "Fresh reserve consumed:",
        False,
    )
    print(
        "Canonical packet eligible for DEV ranker validation:",
        report[
            "canonical_packet_eligible_for_dev_ranker_validation"
        ],
    )
    print(
        "Network calls during verification:",
        0,
    )
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
