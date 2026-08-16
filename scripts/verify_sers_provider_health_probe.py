from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.provider_health_probe import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_RUN_ROOT,
    DEFAULT_SPEC_ROOT,
    verify_probe_spec,
    verify_run,
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

    spec_issues, spec = (
        verify_probe_spec(
            root=ROOT,
            diagnostic_root=
                diagnostic_root,
            spec_path=
                spec_root
                / "probe_spec.json",
        )
    )
    if spec_issues:
        print(
            "provider-health verification: FAIL"
        )
        for issue in spec_issues:
            print(" -", issue)
        print("Network searches:", 0)
        return 2

    run_issues, result = (
        verify_run(
            run_path=
                run_root
                / "probe_run.json",
            spec=spec,
        )
    )
    if run_issues:
        print(
            "provider-health verification: FAIL"
        )
        for issue in run_issues:
            print(" -", issue)
        print("Network searches:", 0)
        return 2

    print(
        "provider-health verification: PASS"
    )
    print(
        "Spec ID:",
        spec["spec_id"],
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
        "Per-query terminal failures:",
        [
            (
                row["query_id"],
                row[
                    "failure_category"
                ],
            )
            for row in result[
                "executions"
            ]
            if not row["success"]
        ],
    )
    print(
        "Full DEV rerun authorized:",
        False,
    )
    print(
        "Network searches during verification:",
        0,
    )
    print(
        "LLM calls:",
        0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
