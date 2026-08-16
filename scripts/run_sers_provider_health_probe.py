from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dac_her.provider_health_probe import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    EXPECTED_CLEAN_BRANCH,
    EXPECTED_CLEAN_HEAD,
    atomic_json,
    read_json,
    run_probe,
    verify_probe_spec,
    verify_run,
)


ROOT = Path.cwd()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


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
        "--confirm-one-shot-provider-health-probe",
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

    if git(
        "branch",
        "--show-current",
    ) != EXPECTED_CLEAN_BRANCH:
        raise SystemExit(
            "Unexpected clean branch."
        )
    if git(
        "rev-parse",
        "HEAD",
    ) != EXPECTED_CLEAN_HEAD:
        raise SystemExit(
            "Unexpected clean HEAD."
        )

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
    spec_path = (
        spec_root / "probe_spec.json"
    )
    pass_path = (
        spec_root
        / "SPEC_FREEZE_PASS.json"
    )

    issues, spec = (
        verify_probe_spec(
            root=ROOT,
            diagnostic_root=
                diagnostic_root,
            spec_path=
                spec_path,
        )
    )
    if issues:
        print(
            "provider-health probe preflight: FAIL"
        )
        for issue in issues:
            print(" -", issue)
        print("Network searches:", 0)
        return 2

    if not pass_path.is_file():
        print(
            "provider-health probe preflight: FAIL"
        )
        print(
            " - SPEC_FREEZE_PASS missing"
        )
        print("Network searches:", 0)
        return 2
    passed = read_json(
        pass_path
    )
    if (
        passed.get("status")
        != "spec_freeze_pass"
        or passed.get(
            "spec_id"
        )
        != spec.get("spec_id")
    ):
        print(
            "provider-health probe preflight: FAIL"
        )
        print(
            " - frozen spec pass marker mismatch"
        )
        print("Network searches:", 0)
        return 2

    if output_root.exists():
        print(
            "provider-health probe preflight: FAIL"
        )
        print(
            " - output root already exists:",
            output_root,
        )
        print(
            " - automatic rerun refused"
        )
        print("Network searches:", 0)
        return 2

    print(
        "SERS Semantic Scholar One-Shot Provider Health Probe"
    )
    print(
        "Spec ID:",
        spec["spec_id"],
    )
    print(
        "Frozen logical requests:",
        spec[
            "expected_logical_execution_count"
        ],
    )
    print(
        "Provider:",
        spec["provider"],
    )
    print(
        "Result limit/query:",
        spec[
            "result_limit_per_query"
        ],
    )
    print(
        "Scientific result use:",
        False,
    )
    print(
        "Automatic rerun:",
        False,
    )
    print(
        "LLM calls:",
        0,
    )

    if args.preflight:
        print("Preflight: PASS")
        print("Network searches:", 0)
        print("Write performed:", False)
        return 0

    if not (
        args.
        confirm_one_shot_provider_health_probe
    ):
        print(
            "provider-health probe: REFUSED"
        )
        print(
            "Missing --confirm-one-shot-provider-health-probe"
        )
        print("Network searches:", 0)
        return 2

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )
    try:
        result = run_probe(
            spec=spec
        )
        run_path = (
            output_root
            / "probe_run.json"
        )
        atomic_json(
            run_path,
            result,
        )
        issues, verified = (
            verify_run(
                run_path=run_path,
                spec=spec,
            )
        )
        if issues:
            raise RuntimeError(
                "probe run verification failed:\n- "
                + "\n- ".join(issues)
            )
        atomic_json(
            output_root
            / "PROBE_COMPLETE.json",
            {
                "status":
                    "probe_complete",
                "run_id":
                    verified[
                        "run_id"
                    ],
                "run_sha256":
                    verified[
                        "run_sha256"
                    ],
                "outcome":
                    verified[
                        "outcome"
                    ],
                "successful_logical_execution_count":
                    verified[
                        "successful_logical_execution_count"
                    ],
                "logical_execution_count":
                    verified[
                        "logical_execution_count"
                    ],
                "total_429_event_count":
                    verified[
                        "total_429_event_count"
                    ],
                "full_dev_rerun_authorized":
                    False,
                "automatic_rerun_authorized":
                    False,
                "llm_calls":
                    0,
            },
        )
    except Exception as exc:
        atomic_json(
            output_root
            / "RUN_INTERRUPTED.json",
            {
                "status":
                    "probe_interrupted",
                "error_type":
                    type(exc).__name__,
                "error_message":
                    str(exc),
                "automatic_rerun_authorized":
                    False,
                "llm_calls":
                    0,
            },
        )
        raise

    print()
    print(
        "provider-health probe: COMPLETE"
    )
    print(
        "Run ID:",
        result["run_id"],
    )
    print(
        "Outcome:",
        result["outcome"],
    )
    print(
        "API key configured:",
        result[
            "api_key_configured"
        ],
    )
    print(
        "Logical successes:",
        result[
            "successful_logical_execution_count"
        ],
        "/",
        result[
            "logical_execution_count"
        ],
    )
    print(
        "HTTP attempts:",
        result[
            "total_http_attempt_count"
        ],
    )
    print(
        "429 events:",
        result[
            "total_429_event_count"
        ],
    )
    print(
        "Retry-After honored:",
        result[
            "retry_after_honored_count"
        ],
    )
    print(
        "Terminal cooldowns:",
        result[
            "terminal_cooldown_count"
        ],
    )
    print(
        "Full DEV rerun authorized:",
        False,
    )
    print(
        "LLM calls:",
        0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
