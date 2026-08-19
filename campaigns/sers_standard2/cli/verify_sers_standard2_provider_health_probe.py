from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.provider_health_probe import (
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
        spec_path=spec_root / "probe_spec.json",
    )
    if spec_issues:
        print("standard2 provider-health verification: FAIL")
        for issue in spec_issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    run_issues, result = verify_run(
        spec=spec,
        run_path=run_root / "probe_run.json",
    )
    if run_issues:
        print("standard2 provider-health verification: FAIL")
        for issue in run_issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    print("standard2 provider-health verification: PASS")
    print("Spec ID:", spec["spec_id"])
    print("Run ID:", result["run_id"])
    print("Outcome:", result["outcome"])
    print(
        "Logical successes:",
        result["successful_logical_execution_count"],
        "/",
        result["logical_execution_count"],
    )
    print("Provider summary:", result["provider_summary"])
    print(
        "Per-query failures:",
        [
            (
                row["provider"],
                row["query_id"],
                row["failure_category"],
                row["http_status"],
            )
            for row in result["executions"]
            if not row["success"]
        ],
    )
    print("Paper titles persisted:", result["paper_titles_persisted"])
    print("Scientific result interpretation:", False)
    print("Full DEV rerun authorized:", False)
    print("Network calls during verification:", 0)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
