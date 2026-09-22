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
            "Opt-in wrapper: run the unmodified legacy E2E, then use the retained "
            "pre-N10 Alpha6 portfolio to generate cross-lane scientific synthesis "
            "candidates, run fresh strict N10 over them, merge only positive-authority "
            "survivors, and run sidecar final semantic/feasibility."
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
    parser.add_argument(
        "--critic-model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("e2e_args", nargs=argparse.REMAINDER)
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
    subprocess.run(legacy_cmd, check=True)

    pre_n10_cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_pre_n10_scientific_synthesis_companion",
        "--run-dir",
        str(run_dir),
        "--max-syntheses",
        str(args.max_syntheses),
    ]
    for root in args.semantic_root:
        pre_n10_cmd += ["--root", root]
    if args.model:
        pre_n10_cmd += ["--model", args.model]
    if args.base_url:
        pre_n10_cmd += ["--base-url", args.base_url]
    pre_n10_cmd += ["--api-key-env", args.api_key_env]
    if args.save_prompts:
        pre_n10_cmd += ["--save-prompts"]

    print()
    print("== pre-N10 scientific synthesis ==")
    subprocess.run(pre_n10_cmd, check=True)

    n10_cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_scientific_synthesis_n10_e2e",
        "--run-dir",
        str(run_dir),
    ]
    if args.model:
        n10_cmd += ["--model", args.model]
    if args.critic_model:
        n10_cmd += ["--critic-model", args.critic_model]
    if args.base_url:
        n10_cmd += ["--base-url", args.base_url]
    n10_cmd += ["--api-key-env", args.api_key_env]
    if args.save_prompts:
        n10_cmd += ["--save-prompts"]

    print()
    print("== scientific synthesis strict-N10 + final sidecar ==")
    subprocess.run(n10_cmd, check=True)

    print()
    print("Legacy E2E + pre-N10 scientific candidate E2E complete")
    print("Legacy production selection mutated by scientific path: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
