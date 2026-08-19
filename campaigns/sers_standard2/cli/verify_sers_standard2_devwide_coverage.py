from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.devwide_coverage_audit import (
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
        diagnostic_root=diagnostic_root,
        spec_path=spec_root / "coverage_spec.json",
    )
    if spec_issues:
        print("standard2 DEV-wide coverage verification: FAIL")
        for issue in spec_issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    run_issues, result = verify_run(
        spec=spec,
        run_path=run_root / "coverage_run.json",
    )
    if run_issues:
        print("standard2 DEV-wide coverage verification: FAIL")
        for issue in run_issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    print("standard2 DEV-wide coverage verification: PASS")
    print("Spec ID:", spec["spec_id"])
    print("Run ID:", result["run_id"])
    print("Operational outcome:", result["operational_outcome"])
    print("Logical successes:",
          result["successful_logical_execution_count"],
          "/",
          result["logical_execution_count"])
    print("Provider summary:", result["provider_summary"])
    print(
        "Cross-provider overlap:",
        result["cross_provider_unique_work_overlap"],
    )
    print(
        "Combined abstract coverage:",
        result["combined_abstract_coverage"],
    )
    print(
        "Abstract complementarity:",
        result["abstract_complementarity"],
    )
    print("Paper titles persisted:", result["paper_titles_persisted"])
    print("Raw abstract text persisted:", result["raw_abstract_text_persisted"])
    print("Ranker used:", result["ranker_used"])
    print("Scientific interpretation:", False)
    print("Full DEV rerun authorized:", False)
    print("Network calls during verification:", 0)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
