from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.provider_health_probe import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    read_json,
    run_probe,
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
        "--confirm-one-shot-standard2-health-probe",
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
        spec_path=spec_root / "probe_spec.json",
    )
    if issues:
        print("standard2 provider-health preflight: FAIL")
        for issue in issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    marker = spec_root / "SPEC_FREEZE_PASS.json"
    if not marker.is_file():
        print("standard2 provider-health preflight: FAIL")
        print(" - SPEC_FREEZE_PASS missing")
        print("Network calls:", 0)
        return 2
    marker_value = read_json(marker)
    if (
        marker_value.get("status") != "spec_freeze_pass"
        or marker_value.get("spec_id") != spec.get("spec_id")
    ):
        print("standard2 provider-health preflight: FAIL")
        print(" - frozen spec marker mismatch")
        print("Network calls:", 0)
        return 2

    if output_root.exists():
        print("standard2 provider-health preflight: FAIL")
        print(" - output root already exists:", output_root)
        print(" - automatic rerun refused")
        print("Network calls:", 0)
        return 2

    print("SERS STANDARD_2_PROVIDER One-Shot Health Probe")
    print("Spec ID:", spec["spec_id"])
    print("Providers:", spec["provider_plan"]["active_providers"])
    print("Logical executions:", spec["expected_logical_execution_count"])
    print("Result limit/query/provider:", spec["result_limit_per_query"])
    print("Scientific result interpretation:", False)
    print("Automatic rerun:", False)
    print("LLM calls:", 0)

    if args.preflight:
        print("Preflight: PASS")
        print("Network calls:", 0)
        print("Write performed:", False)
        return 0

    if not args.confirm_one_shot_standard2_health_probe:
        print("standard2 provider-health probe: REFUSED")
        print("Missing --confirm-one-shot-standard2-health-probe")
        print("Network calls:", 0)
        return 2

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        result = run_probe(spec=spec)
        run_path = output_root / "probe_run.json"
        atomic_json(run_path, result)
        run_issues, verified = verify_run(
            spec=spec,
            run_path=run_path,
        )
        if run_issues:
            raise RuntimeError(
                "run verification failed:\n- "
                + "\n- ".join(run_issues)
            )
        atomic_json(
            output_root / "PROBE_COMPLETE.json",
            {
                "status": "probe_complete",
                "run_id": verified["run_id"],
                "run_sha256": verified["run_sha256"],
                "outcome": verified["outcome"],
                "successful_logical_execution_count":
                    verified["successful_logical_execution_count"],
                "logical_execution_count":
                    verified["logical_execution_count"],
                "full_dev_rerun_authorized": False,
                "automatic_rerun_authorized": False,
                "llm_calls": 0,
            },
        )
    except Exception as exc:
        atomic_json(
            output_root / "RUN_INTERRUPTED.json",
            {
                "status": "probe_interrupted",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "automatic_rerun_authorized": False,
                "llm_calls": 0,
            },
        )
        raise

    print()
    print("standard2 provider-health probe: COMPLETE")
    print("Run ID:", result["run_id"])
    print("Outcome:", result["outcome"])
    print(
        "Logical successes:",
        result["successful_logical_execution_count"],
        "/",
        result["logical_execution_count"],
    )
    print("Provider summary:", result["provider_summary"])
    print("Full DEV rerun authorized:", False)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
