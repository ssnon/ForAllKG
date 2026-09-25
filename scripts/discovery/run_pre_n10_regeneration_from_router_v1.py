from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.pre_n10_regeneration_router_runtime_v1 import (
    execute_pre_n10_regeneration_router_runtime_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen one-shot pre-N10 regeneration unit for exactly "
            "the fallback population emitted by the unified primary router. "
            "This stage performs no repair, semantic critic, decomposition, "
            "retrieval, N9, N10, or second regeneration."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--primary-router-report", required=True, type=Path)
    parser.add_argument(
        "--regeneration-unit-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report, _raw = execute_pre_n10_regeneration_router_runtime_v1(
        context_path=args.context,
        primary_router_report_path=args.primary_router_report,
        regeneration_unit_freeze_path=args.regeneration_unit_freeze,
        output_root=args.output_dir,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout_seconds=args.timeout_seconds,
    )

    print("Pre-N10 router one-shot regeneration complete")
    print("Report:", report.report_id)
    print("Source primary router:", report.source_primary_report_id)
    print("Fallback lineages:", report.regeneration_required_count)
    print("Generation calls:", report.generation_call_count)
    print("Repair calls:", report.repair_call_count)
    print(
        "Generated/abstained/compile-rejected/generation-failed:",
        report.generated_and_compiled_count,
        "/",
        report.generated_abstention_count,
        "/",
        report.compile_rejected_count,
        "/",
        report.generation_failed_count,
    )
    print("Semantic critic performed: false")
    print("Claim decomposition performed: false")
    print("External novelty performed: false")
    print("N9 performed: false")
    print("N10 performed: false")
    print("Second regeneration performed: false")
    print(
        "Output:",
        args.output_dir.expanduser().resolve()
        / "regeneration.execution.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
