from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.standard2_ranker_dev_validation import (
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

    diagnostic_root = args.diagnostic_root.expanduser().resolve()
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
        spec_path=spec_root / "ranker_spec.json",
    )
    if spec_issues:
        print("ranker-only DEV verification: FAIL")
        for issue in spec_issues:
            print(" -", issue)
        return 2

    run_issues, report = verify_run(
        repo_root=ROOT,
        diagnostic_root=diagnostic_root,
        spec=spec,
        report_path=run_root / "ranker_report.json",
        audit_path=run_root / "human_relevance_audit.md",
    )
    if run_issues:
        print("ranker-only DEV verification: FAIL")
        for issue in run_issues:
            print(" -", issue)
        return 2

    print("ranker-only DEV verification: PASS")
    print("Spec ID:", spec["spec_id"])
    print("Run ID:", report["run_id"])
    print("Mechanical outcome:", report["mechanical_outcome"])
    print(
        "Scientific relevance:",
        report["scientific_relevance_outcome"],
    )
    print("Summary:", report["summary"])
    print("Network calls:", 0)
    print("LLM calls:", 0)
    print("Claim review used:", False)
    print("Novelty verdict used:", False)
    print("Fresh Reserve consumed:", False)
    print(
        "Automatic claim-level review authorized:",
        False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
