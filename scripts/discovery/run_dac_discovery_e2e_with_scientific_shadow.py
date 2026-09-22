from __future__ import annotations

import argparse
import os
import subprocess
import sys

from pipeline_core.discovery.reframing.live_reframing_companion import (
    extract_run_dir_from_e2e_args,
    strip_remainder_separator,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Opt-in wrapper: run the unmodified legacy discovery E2E first, then run "
            "the live scientific-reframing shadow companion only after legacy success. "
            "Pass legacy E2E arguments after '--'."
        )
    )
    parser.add_argument("--semantic-root", action="append", required=True)
    parser.add_argument("--max-syntheses", type=int, default=3)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--telemetry", default=None)
    parser.add_argument(
        "e2e_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to scripts.discovery.run_dac_discovery_e2e",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    e2e_args = strip_remainder_separator(list(args.e2e_args))
    if not e2e_args:
        raise SystemExit("legacy E2E arguments are required after '--'")
    run_dir = extract_run_dir_from_e2e_args(e2e_args)

    legacy_cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_dac_discovery_e2e",
        *e2e_args,
    ]
    print("== legacy E2E ==")
    print("$", *legacy_cmd)
    subprocess.run(legacy_cmd, check=True)

    companion_cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_live_scientific_reframing_companion",
        "--run-dir",
        str(run_dir),
        "--max-syntheses",
        str(args.max_syntheses),
    ]
    for root in args.semantic_root:
        companion_cmd += ["--root", root]
    if args.model:
        companion_cmd += ["--model", args.model]
    if args.base_url:
        companion_cmd += ["--base-url", args.base_url]
    if args.api_key_env:
        companion_cmd += ["--api-key-env", args.api_key_env]
    if args.save_prompts:
        companion_cmd += ["--save-prompts"]
    if args.telemetry:
        companion_cmd += ["--telemetry", args.telemetry]

    print()
    print("== scientific shadow companion ==")
    print("$", *companion_cmd)
    try:
        subprocess.run(companion_cmd, check=True)
    except subprocess.CalledProcessError:
        print(
            "Scientific shadow companion failed after a successful legacy E2E. "
            "Legacy E2E artifacts remain authoritative and unchanged.",
            file=sys.stderr,
        )
        raise

    print()
    print("Legacy E2E + scientific shadow companion complete")
    print("Legacy production selection changed by wrapper: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
