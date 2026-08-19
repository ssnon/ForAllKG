from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_standard2.canonicalization_dev_validation import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_SPEC_ROOT,
    atomic_json,
    build_spec,
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
        "--diagnostic-root",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_SPEC_ROOT,
    )
    args = parser.parse_args()

    diagnostic_root = (
        args.diagnostic_root
        .expanduser()
        .resolve()
    )
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else ROOT / args.output_root
    )

    if output_root.exists():
        print(
            "canonicalization DEV spec preflight: FAIL"
        )
        print(
            " - output root exists:",
            output_root,
        )
        print("Write performed: False")
        return 2

    try:
        spec = build_spec(
            repo_root=ROOT,
            diagnostic_root=diagnostic_root,
        )
    except Exception as exc:
        print(
            "canonicalization DEV spec preflight: FAIL"
        )
        print(
            " -",
            f"{type(exc).__name__}: {exc}",
        )
        print("Write performed: False")
        return 2

    print(
        "SERS STANDARD_2 Canonicalization-only DEV Specification"
    )
    print(
        "Prospective spec ID:",
        spec["spec_id"],
    )
    print(
        "Prospective spec SHA256:",
        spec["spec_sha256"],
    )
    print(
        "Provider mode:",
        spec["provider_plan"]["mode"],
    )
    print(
        "Active providers:",
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
    print(
        "Canonicalizer:",
        spec["canonicalizer"],
    )
    print("Ranker used:", False)
    print("LLM calls:", 0)
    print(
        "Scientific interpretation:",
        False,
    )
    print("Network calls:", 0)

    if args.preflight:
        print("Preflight: PASS")
        print("Write performed: False")
        return 0

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )
    spec_path = (
        output_root / "canonicalization_spec.json"
    )
    pass_path = (
        output_root / "SPEC_FREEZE_PASS.json"
    )
    try:
        atomic_json(
            spec_path,
            spec,
        )
        issues, verified = verify_spec(
            repo_root=ROOT,
            diagnostic_root=diagnostic_root,
            spec_path=spec_path,
        )
        if issues:
            raise RuntimeError(
                "spec verification failed:\n- "
                + "\n- ".join(issues)
            )
        atomic_json(
            pass_path,
            {
                "status":
                    "spec_freeze_pass",
                "spec_id":
                    verified["spec_id"],
                "spec_sha256":
                    verified["spec_sha256"],
                "network_calls": 0,
                "llm_calls": 0,
            },
        )
    except Exception:
        for path in (
            pass_path,
            spec_path,
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            output_root.rmdir()
        except OSError:
            pass
        raise

    print(
        "canonicalization-only DEV specification: FROZEN"
    )
    print("Spec ID:", spec["spec_id"])
    print("Network calls:", 0)
    print("LLM calls:", 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
