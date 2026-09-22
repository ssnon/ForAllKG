from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    InstructorAtomicSynthesisBackend,
    build_atomic_synthesis_prompt,
    run_atomic_synthesis,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate production-side shadow cross-lane hypotheses with atomic "
            "N9/N10 specification attached at synthesis time, then deterministically "
            "materialize both HypothesisPortfolio and LiteratureQueryPlan without "
            "external-novelty LLM re-decomposition."
        )
    )
    parser.add_argument("--candidate-portfolio", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--portfolio-output", required=True, type=Path)
    parser.add_argument("--query-plan-output", required=True, type=Path)
    parser.add_argument("--prompt-output", type=Path, default=None)
    parser.add_argument("--max-syntheses", type=int, default=2)
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
    parser.add_argument("--instructor-mode", default="JSON")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--telemetry", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    candidates = ProductionFacingScientificCandidatePortfolio.model_validate_json(
        args.candidate_portfolio.read_text(encoding="utf-8")
    )
    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )

    print("Atomic cross-lane scientific synthesis")
    print("Source candidates:", candidates.candidate_count)
    print("Relational candidates:", candidates.relational_candidate_count)
    print("Reframing candidates:", candidates.reframing_candidate_count)
    print("Max syntheses:", args.max_syntheses)
    print("External-novelty LLM re-decomposition required: false")

    if args.dry_run:
        prompt = build_atomic_synthesis_prompt(
            candidates,
            max_syntheses=args.max_syntheses,
        )
        if args.prompt_output is not None:
            args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
            args.prompt_output.write_text(
                "SYSTEM\n======\n"
                + prompt.system_prompt
                + "\n\nUSER\n====\n"
                + prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )
        print("LLM calls: 0")
        print("Would execute atomic synthesis call: true")
        print("Production authority: false")
        return 0

    if not args.model:
        raise SystemExit("--model is required")

    backend = InstructorAtomicSynthesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        telemetry_path=args.telemetry,
        telemetry_context={
            "task_id": candidates.source_task_id,
            "source_context_id": candidates.source_context_id,
            "candidate_portfolio_id": candidates.portfolio_id,
        },
    )
    report, portfolio, plan, prompt = run_atomic_synthesis(
        portfolio=candidates,
        context=context,
        backend=backend,
        max_syntheses=args.max_syntheses,
    )

    for path in (
        args.report_output,
        args.portfolio_output,
        args.query_plan_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    args.report_output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.portfolio_output.write_text(
        portfolio.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.query_plan_output.write_text(
        plan.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if args.prompt_output is not None and prompt is not None:
        args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_output.write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )

    print("Atomic synthesis complete")
    print("LLM calls:", report.llm_calls_performed)
    print("Hypotheses:", report.hypothesis_count)
    print("Atomic specifications:", report.atomic_specification_count)
    for hypothesis in report.hypotheses:
        print(" ", hypothesis.hypothesis_id, hypothesis.title)
        for spec in hypothesis.atomic_specifications:
            print(
                "   ",
                spec.claim_id,
                spec.novelty_selection_role,
                repr(spec.prior_art_identity_terms),
            )
    if report.abstention_reason:
        print("Abstention:", report.abstention_reason)
    print("Branch-specific sanitizers reused: true")
    print("Production authority: false")
    print("Report:", args.report_output)
    print("Portfolio:", args.portfolio_output)
    print("Query plan:", args.query_plan_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
