from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_core.discovery.reframing.cross_lane_synthesis import (
    CrossLaneSynthesisRuntime,
    InstructorOpenAICompatibleCrossLaneSynthesisBackend,
    build_cross_lane_synthesis_prompt,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
)


def _headers(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--header values must use KEY=VALUE")
        key, item = value.split("=", 1)
        if not key.strip():
            raise ValueError("--header key must be non-empty")
        result[key.strip()] = item
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate shadow-only cross-lane synthesized scientific hypotheses from the common "
            "production-facing candidate portfolio. Source candidates are preserved and no final "
            "production selection or novelty judgment is performed."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prompt-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-syntheses", type=int, default=3)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--telemetry", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.max_syntheses < 1:
        raise SystemExit("--max-syntheses must be at least 1")
    portfolio_path = args.portfolio.expanduser().resolve()
    portfolio = ProductionFacingScientificCandidatePortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    output = args.output or portfolio_path.with_name(
        "scientific_cross_lane_synthesis_shadow.json"
    )
    prompt_output = args.prompt_output

    print("Cross-lane scientific synthesis shadow")
    print("Task:", portfolio.source_task_id)
    print("Source candidates:", portfolio.candidate_count)
    print("Relational candidates:", portfolio.relational_candidate_count)
    print("Reframing candidates:", portfolio.reframing_candidate_count)
    print("Max synthesized hypotheses:", args.max_syntheses)

    if args.dry_run:
        prompt = build_cross_lane_synthesis_prompt(
            portfolio,
            max_syntheses=args.max_syntheses,
        )
        if prompt_output is not None:
            prompt_output.parent.mkdir(parents=True, exist_ok=True)
            prompt_output.write_text(
                "SYSTEM\n======\n"
                + prompt.system_prompt
                + "\n\nUSER\n====\n"
                + prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )
        print("LLM calls: 0")
        print("Would execute synthesis call: true")
        print("Scientific quality ranking performed: false")
        print("Production selection changed: false")
        return 0

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or OPENROUTER_AGENT_MODEL is set"
        )
    backend = InstructorOpenAICompatibleCrossLaneSynthesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=_headers(args.header),
        telemetry_path=args.telemetry,
        telemetry_context={
            "task_id": portfolio.source_task_id,
            "source_context_id": portfolio.source_context_id,
            "source_candidate_portfolio_id": portfolio.portfolio_id,
        },
    )
    report, prompt = CrossLaneSynthesisRuntime(backend).run(
        portfolio=portfolio,
        max_syntheses=args.max_syntheses,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if prompt_output is not None and prompt is not None:
        prompt_output.parent.mkdir(parents=True, exist_ok=True)
        prompt_output.write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )

    print("Cross-lane scientific synthesis complete")
    print("LLM calls:", report.llm_calls_performed)
    print("Synthesized hypotheses:", report.synthesized_hypothesis_count)
    for row in report.hypotheses:
        print(f"  {row.hypothesis_id}: {row.title}")
        print("    kind=", row.synthesis_kind, sep="")
        print("    sources=", ",".join(row.source_candidate_ids), sep="")
    if report.abstention_reason:
        print("Abstention:", report.abstention_reason)
    print("Source candidates preserved: true")
    print("New evidence asserted: false")
    print("Novelty assessment performed: false")
    print("Scientific quality ranking performed: false")
    print("Final hypothesis selection performed: false")
    print("Production selection changed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
