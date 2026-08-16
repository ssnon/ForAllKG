from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.standard2_devwide_coverage_audit import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    read_json,
    run_audit,
    verify_run,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--confirm-one-shot-standard2-devwide-coverage-audit",
        action="store_true",
    )
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
        "--output-root",
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
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    issues, spec = verify_spec(
        diagnostic_root=diagnostic_root,
        spec_path=spec_root / "coverage_spec.json",
    )
    if issues:
        print("standard2 DEV-wide coverage preflight: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    marker_path = spec_root / "SPEC_FREEZE_PASS.json"
    if not marker_path.is_file():
        print("standard2 DEV-wide coverage preflight: FAIL")
        print(" - SPEC_FREEZE_PASS missing")
        print("Network calls:", 0)
        return 2
    marker = read_json(marker_path)
    if (
        marker.get("status") != "spec_freeze_pass"
        or marker.get("spec_id") != spec.get("spec_id")
    ):
        print("standard2 DEV-wide coverage preflight: FAIL")
        print(" - spec freeze marker mismatch")
        print("Network calls:", 0)
        return 2

    if output_root.exists():
        print("standard2 DEV-wide coverage preflight: FAIL")
        print(" - output root already exists:", output_root)
        print(" - automatic rerun refused")
        print("Network calls:", 0)
        return 2

    print("SERS STANDARD_2_PROVIDER DEV-wide One-Shot Coverage Audit")
    print("Spec ID:", spec["spec_id"])
    print("Providers:", spec["provider_plan"]["active_providers"])
    print("Frozen queries:", spec["query_count"])
    print("Result limit/query/provider:", spec["result_limit_per_query_provider"])
    print("Logical executions:", spec["logical_execution_count"])
    print("Scientific interpretation:", False)
    print("Ranker used:", False)
    print("Automatic rerun:", False)
    print("LLM calls:", 0)

    if args.preflight:
        print("Preflight: PASS")
        print("Network calls:", 0)
        print("Write performed:", False)
        return 0

    if not args.confirm_one_shot_standard2_devwide_coverage_audit:
        print("standard2 DEV-wide coverage audit: REFUSED")
        print(
            "Missing --confirm-one-shot-standard2-devwide-coverage-audit"
        )
        print("Network calls:", 0)
        return 2

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        result = run_audit(spec=spec)
        run_path = output_root / "coverage_run.json"
        atomic_json(run_path, result)
        run_issues, verified = verify_run(
            spec=spec,
            run_path=run_path,
        )
        if run_issues:
            raise RuntimeError(
                "run verification failed:\n- " + "\n- ".join(run_issues)
            )
        atomic_json(
            output_root / "AUDIT_COMPLETE.json",
            {
                "status": "audit_complete",
                "run_id": verified["run_id"],
                "run_sha256": verified["run_sha256"],
                "operational_outcome": verified["operational_outcome"],
                "abstract_complementarity":
                    verified["abstract_complementarity"],
                "full_dev_rerun_authorized": False,
                "automatic_provider_policy_change_authorized": False,
                "llm_calls": 0,
            },
        )
    except Exception as exc:
        atomic_json(
            output_root / "RUN_INTERRUPTED.json",
            {
                "status": "audit_interrupted",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "automatic_rerun_authorized": False,
                "llm_calls": 0,
            },
        )
        raise

    print()
    print("standard2 DEV-wide coverage audit: COMPLETE")
    print("Run ID:", result["run_id"])
    print("Operational outcome:", result["operational_outcome"])
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
    print("Full DEV rerun authorized:", False)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
