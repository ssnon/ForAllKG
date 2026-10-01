from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.frontier_exploration_audit import FrontierExplorationAudit
from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.idea_evolution import IdeaEvolutionReport
from pipeline_core.discovery.scientific_portfolio_audit import (
    build_scientific_portfolio_audit,
)
from pipeline_core.discovery.scientific_portfolio_runtime import (
    InstructorOpenAICompatibleScientificPortfolioBackend,
    ScientificPortfolioRuntime,
)


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--header values must use KEY=VALUE")
    key, item = value.split("=", 1)
    if not key.strip():
        raise argparse.ArgumentTypeError("header key may not be empty")
    return key.strip(), item


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate Frontier + Idea Evolution candidates, retain a bounded diverse "
            "shadow research portfolio, and materialize retained ideas through the "
            "existing grounded HypothesisCompiler/HypothesisValidator boundary."
        )
    )
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--population", required=True, type=Path)
    p.add_argument("--frontier-audit", required=True, type=Path)
    p.add_argument("--evolution-report", required=True, type=Path)
    p.add_argument("--task-source", required=True)
    p.add_argument("--task-target", required=True)
    p.add_argument("--max-evaluation-candidates", type=int, default=48)
    p.add_argument("--max-retained", type=int, default=8)
    p.add_argument("--max-retained-per-profile", type=int, default=2)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--save-prompts", action="store_true")
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    return p


def main() -> int:
    args = parser().parse_args()
    for name in (
        "max_evaluation_candidates",
        "max_retained",
        "max_retained_per_profile",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    context = _load(args.context.expanduser().resolve(), HypothesisContext)
    population = _load(args.population.expanduser().resolve(), FrontierIdeaPopulation)
    frontier_audit = _load(
        args.frontier_audit.expanduser().resolve(), FrontierExplorationAudit
    )
    evolution_report = _load(
        args.evolution_report.expanduser().resolve(), IdeaEvolutionReport
    )

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    backend = InstructorOpenAICompatibleScientificPortfolioBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
        telemetry_path=out / "telemetry.jsonl",
        telemetry_context={
            "source_population_id": population.population_id,
            "source_context_id": population.source_context_id,
            "source_evolution_report_id": evolution_report.report_id,
        },
    )
    outcome = ScientificPortfolioRuntime(backend).run(
        context=context,
        population=population,
        exploration_audit=frontier_audit,
        evolution_report=evolution_report,
        task_source=args.task_source,
        task_target=args.task_target,
        max_evaluation_candidates=args.max_evaluation_candidates,
        max_retained_candidates=args.max_retained,
        max_retained_per_profile=args.max_retained_per_profile,
    )
    audit = build_scientific_portfolio_audit(
        pool=outcome.pool,
        evaluation=outcome.evaluation,
        selection=outcome.selection,
        materialization=outcome.materialization_report,
    )

    _write(out / "candidate_pool.json", outcome.pool)
    _write(out / "evaluation.json", outcome.evaluation)
    _write(out / "selection.json", outcome.selection)
    _write(out / "materialization.draft.json", outcome.materialization_draft)
    _write(out / "materialization.report.json", outcome.materialization_report)
    _write(out / "materialized.shadow.portfolio.json", outcome.materialized_portfolio)
    _write(out / "audit.json", audit)

    if args.save_prompts:
        prompt_dir = out / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "evaluation.prompt.txt").write_text(
            "SYSTEM\n======\n"
            + outcome.evaluation_prompt.system_prompt
            + "\n\nUSER\n====\n"
            + outcome.evaluation_prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )
        if outcome.materialization_prompt is not None:
            (prompt_dir / "materialization.prompt.txt").write_text(
                "SYSTEM\n======\n"
                + outcome.materialization_prompt.system_prompt
                + "\n\nUSER\n====\n"
                + outcome.materialization_prompt.user_prompt
                + "\n",
                encoding="utf-8",
            )

    print("Scientific Portfolio Selection shadow complete")
    print("projected candidates:", outcome.pool.projected_candidate_count)
    print("retained:", outcome.selection.retained_count)
    print("retained unique families:", outcome.selection.retained_unique_family_count)
    print("profiles:", outcome.selection.retained_count_by_profile)
    print("origins:", outcome.selection.retained_count_by_origin)
    print("materialized hypotheses:", outcome.materialization_report.materialized_hypothesis_count)
    print("materialization status:", outcome.materialization_report.status_counts)
    print("materialization generation error:", outcome.materialization_generation_error)
    print("downstream verification ready:", audit.downstream_verification_ready)
    print("LLM calls:", outcome.llm_calls_performed)
    print("OVERALL_WINNER_SELECTED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
