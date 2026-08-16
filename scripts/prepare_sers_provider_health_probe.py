from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dac_her.provider_health_probe import (
    DEFAULT_DIAGNOSTIC_ROOT,
    DEFAULT_SPEC_ROOT,
    EXPECTED_CLEAN_BRANCH,
    EXPECTED_CLEAN_HEAD,
    EXPECTED_PATCHED_RETRIEVAL_BLOB,
    EXPECTED_PROVIDER_RESILIENCE_SHA256,
    atomic_json,
    build_probe_spec,
    sha256_file,
    verify_probe_spec,
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

    branch = git(
        "branch",
        "--show-current",
    )
    head = git(
        "rev-parse",
        "HEAD",
    )
    blob = git(
        "hash-object",
        "dac_her/literature_retrieval.py",
    )
    if branch != EXPECTED_CLEAN_BRANCH:
        raise SystemExit(
            f"Unexpected branch: {branch}"
        )
    if head != EXPECTED_CLEAN_HEAD:
        raise SystemExit(
            f"Unexpected HEAD: {head}"
        )
    if blob != EXPECTED_PATCHED_RETRIEVAL_BLOB:
        raise SystemExit(
            f"Unexpected patched retrieval blob: {blob}"
        )
    if sha256_file(
        ROOT
        / "dac_her/provider_resilience.py"
    ) != EXPECTED_PROVIDER_RESILIENCE_SHA256:
        raise SystemExit(
            "provider_resilience.py drift"
        )

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
            "provider-health spec preflight: FAIL"
        )
        print(
            " - output root exists:",
            output_root,
        )
        print("Write performed: False")
        return 2

    try:
        spec = build_probe_spec(
            root=ROOT,
            diagnostic_root=
                diagnostic_root,
        )
    except Exception as exc:
        print(
            "provider-health spec preflight: FAIL"
        )
        print(
            " -",
            f"{type(exc).__name__}: {exc}",
        )
        print("Write performed: False")
        return 2

    print(
        "SERS Semantic Scholar Provider Health Probe Specification"
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
        "Selection policy:",
        spec["selection_policy"],
    )
    print(
        "Frozen hypotheses/queries:",
        spec[
            "expected_hypothesis_count"
        ],
        "/",
        spec[
            "expected_logical_execution_count"
        ],
    )
    print(
        "Query kinds:",
        [
            row["query_kind"]
            for row in spec["queries"]
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
        "Network searches:",
        0,
    )
    print(
        "LLM calls:",
        0,
    )

    if args.preflight:
        print("Preflight: PASS")
        print("Write performed: False")
        return 0

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )
    spec_path = (
        output_root
        / "probe_spec.json"
    )
    pass_path = (
        output_root
        / "SPEC_FREEZE_PASS.json"
    )
    try:
        atomic_json(
            spec_path,
            spec,
        )
        issues, verified = (
            verify_probe_spec(
                root=ROOT,
                diagnostic_root=
                    diagnostic_root,
                spec_path=
                    spec_path,
            )
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
                    verified[
                        "spec_id"
                    ],
                "spec_sha256":
                    verified[
                        "spec_sha256"
                    ],
                "network_searches":
                    0,
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
        "provider-health probe specification: FROZEN"
    )
    print(
        "Spec ID:",
        spec["spec_id"],
    )
    print(
        "Network searches:",
        0,
    )
    print(
        "LLM calls:",
        0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
