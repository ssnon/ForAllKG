from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.standard2_canonicalization_dev_validation import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    read_json,
    run_validation,
    verify_run,
    verify_spec,
)

ROOT = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(
        required=True
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
    )
    mode.add_argument(
        "--run",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-one-shot-canonicalization-dev-validation",
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
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    issues, spec = verify_spec(
        repo_root=ROOT,
        diagnostic_root=diagnostic_root,
        spec_path=(
            spec_root
            / "canonicalization_spec.json"
        ),
    )
    if issues:
        print(
            "canonicalization DEV preflight: FAIL"
        )
        for issue in issues:
            print(" -", issue)
        print("Network calls:", 0)
        return 2

    marker_path = (
        spec_root / "SPEC_FREEZE_PASS.json"
    )
    if not marker_path.is_file():
        print(
            "canonicalization DEV preflight: FAIL"
        )
        print(
            " - SPEC_FREEZE_PASS missing"
        )
        print("Network calls:", 0)
        return 2
    marker = read_json(marker_path)
    if (
        marker.get("status")
        != "spec_freeze_pass"
        or marker.get("spec_id")
        != spec.get("spec_id")
    ):
        print(
            "canonicalization DEV preflight: FAIL"
        )
        print(
            " - spec freeze marker mismatch"
        )
        print("Network calls:", 0)
        return 2

    if output_root.exists():
        print(
            "canonicalization DEV preflight: FAIL"
        )
        print(
            " - output root already exists:",
            output_root,
        )
        print(
            " - automatic rerun refused"
        )
        print("Network calls:", 0)
        return 2

    print(
        "SERS STANDARD_2 Canonicalization-only DEV One-Shot Validation"
    )
    print(
        "Spec ID:",
        spec["spec_id"],
    )
    print(
        "Providers:",
        spec["provider_plan"][
            "active_providers"
        ],
    )
    print(
        "Frozen queries:",
        spec["query_count"],
    )
    print(
        "Logical executions:",
        spec["logical_execution_count"],
    )
    print(
        "Result limit/query/provider:",
        spec[
            "result_limit_per_query_provider"
        ],
    )
    print("Ranker used:", False)
    print("Claim review used:", False)
    print(
        "Scientific interpretation:",
        False,
    )
    print("Automatic rerun:", False)
    print("LLM calls:", 0)

    if args.preflight:
        print("Preflight: PASS")
        print("Network calls:", 0)
        print("Write performed:", False)
        return 0

    if not (
        args
        .confirm_one_shot_canonicalization_dev_validation
    ):
        print(
            "canonicalization DEV validation: REFUSED"
        )
        print(
            "Missing --confirm-one-shot-canonicalization-dev-validation"
        )
        print("Network calls:", 0)
        return 2

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )
    try:
        (
            raw_packet,
            canonical_packet,
            report,
        ) = run_validation(
            repo_root=ROOT,
            diagnostic_root=diagnostic_root,
            spec=spec,
        )

        raw_path = (
            output_root
            / "raw_prior_art.json"
        )
        canonical_path = (
            output_root
            / "canonical_prior_art.json"
        )
        report_path = (
            output_root
            / "canonicalization_report.json"
        )

        # Secret scan is part of the in-memory report and must pass
        # before scientific metadata is persisted.
        if not report[
            "checks"
        ]["secret_scan_pass"]:
            raise RuntimeError(
                "Secret scan failed; refusing to persist prior-art packets."
            )

        atomic_json(
            raw_path,
            raw_packet.model_dump(
                mode="json"
            ),
        )
        atomic_json(
            canonical_path,
            canonical_packet.model_dump(
                mode="json"
            ),
        )
        atomic_json(
            report_path,
            report,
        )

        run_issues, verified = verify_run(
            spec=spec,
            raw_packet_path=raw_path,
            canonical_packet_path=
                canonical_path,
            report_path=report_path,
        )
        if run_issues:
            raise RuntimeError(
                "run verification failed:\n- "
                + "\n- ".join(run_issues)
            )

        atomic_json(
            output_root
            / "VALIDATION_COMPLETE.json",
            {
                "status":
                    "validation_complete",
                "run_id":
                    verified["run_id"],
                "run_sha256":
                    verified["run_sha256"],
                "outcome":
                    verified["outcome"],
                "canonical_packet_id":
                    verified[
                        "canonical_packet_id"
                    ],
                "canonical_packet_sha256":
                    verified[
                        "canonical_packet_sha256"
                    ],
                "canonical_packet_eligible_for_dev_ranker_validation":
                    verified[
                        "canonical_packet_eligible_for_dev_ranker_validation"
                    ],
                "automatic_next_stage_authorized":
                    False,
                "llm_calls": 0,
            },
        )
    except Exception as exc:
        atomic_json(
            output_root
            / "RUN_INTERRUPTED.json",
            {
                "status":
                    "validation_interrupted",
                "error_type":
                    type(exc).__name__,
                "error_message":
                    str(exc),
                "automatic_rerun_authorized":
                    False,
                "llm_calls": 0,
            },
        )
        raise

    print()
    print(
        "canonicalization-only DEV validation: COMPLETE"
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
        "Canonical packet eligible for DEV ranker validation:",
        report[
            "canonical_packet_eligible_for_dev_ranker_validation"
        ],
    )
    print(
        "Automatic next stage authorized:",
        False,
    )
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
