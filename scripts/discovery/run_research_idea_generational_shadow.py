from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.idea_evolution import IdeaEvolutionReport
from pipeline_core.discovery.research_idea_feedback_resolution import (
    resolve_feedback_artifacts,
)
from pipeline_core.discovery.research_idea_generational_shadow import (
    build_generational_idea_search_shadow,
)
from pipeline_core.discovery.scientific_portfolio_materialization import (
    ScientificPortfolioMaterializationReport,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioEvaluationReport,
    ScientificPortfolioSelectionReport,
)


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _load_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run SIS-v2.1 generational idea-search shadow over existing Frontier, "
            "Idea Evolution, scientific-portfolio, materialization, and optional "
            "post-verification artifacts. Feedback can be resolved by exact "
            "portfolio/context/hypothesis lineage. No new LLM or retrieval calls."
        )
    )
    p.add_argument("--population", required=True, type=Path)
    p.add_argument("--evolution-report", required=True, type=Path)
    p.add_argument("--candidate-pool", required=True, type=Path)
    p.add_argument("--evaluation", required=True, type=Path)
    p.add_argument("--selection", required=True, type=Path)
    p.add_argument("--materialization-report", required=True, type=Path)
    p.add_argument("--residual-state", type=Path, default=None)
    p.add_argument("--prospective", type=Path, default=None)
    p.add_argument(
        "--feedback-search-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Search root for exact-lineage residual/prospective artifacts. "
            "May be repeated. Explicit --residual-state/--prospective override "
            "the corresponding auto-resolved payload."
        ),
    )
    p.add_argument("--feedback-resolution-output", type=Path, default=None)
    p.add_argument("--max-g2-parents", type=int, default=8)
    p.add_argument("--max-g2-per-profile", type=int, default=2)
    p.add_argument("--output", required=True, type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.max_g2_parents < 1 or args.max_g2_per_profile < 1:
        raise ValueError("G2 parent limits must be >= 1")

    population = _load(args.population, FrontierIdeaPopulation)
    evolution = _load(args.evolution_report, IdeaEvolutionReport)
    pool = _load(args.candidate_pool, ScientificPortfolioCandidatePool)
    evaluation = _load(args.evaluation, ScientificPortfolioEvaluationReport)
    selection = _load(args.selection, ScientificPortfolioSelectionReport)
    materialization = _load(
        args.materialization_report,
        ScientificPortfolioMaterializationReport,
    )

    residual_state = _load_json(args.residual_state)
    prospective = _load_json(args.prospective)
    resolution = None
    if args.feedback_search_root:
        resolution = resolve_feedback_artifacts(
            materialization=materialization,
            search_roots=args.feedback_search_root,
        )
        if residual_state is None:
            residual_state = resolution.residual_state
        if prospective is None:
            prospective = resolution.prospective_by_hypothesis
        resolution_output = (
            args.feedback_resolution_output
            if args.feedback_resolution_output is not None
            else args.output.with_name(args.output.stem + ".feedback_resolution.json")
        )
        _write(resolution_output, resolution.report)

    report = build_generational_idea_search_shadow(
        population=population,
        evolution_report=evolution,
        pool=pool,
        evaluation=evaluation,
        selection=selection,
        materialization=materialization,
        residual_state=residual_state,
        prospective_by_hypothesis=prospective,
        max_g2_parents=args.max_g2_parents,
        max_g2_per_profile=args.max_g2_per_profile,
    )
    _write(args.output, report)

    print("SIS-v2.1.1 feedback-aware generational shadow complete")
    print("research ideas:", report.research_idea_count)
    print("G0/G1:", report.generation0_idea_count, "/", report.generation1_idea_count)
    print("identity transitions:", report.identity_relation_counts)
    print("semantic no-ops:", report.semantic_noop_count)
    print("observations:", report.observation_count)
    print("residual states:", report.observation_count_by_residual_state)
    print("prospective states:", report.observation_count_by_prospective_status)
    print(
        "feedback coverage residual/prospective:",
        report.feedback_coverage.residual_matched_hypothesis_count,
        "/",
        report.feedback_coverage.prospective_matched_hypothesis_count,
        "of",
        report.feedback_coverage.materialized_hypothesis_count,
    )
    print("baseline G2 parents:", report.baseline_g2_parent_count)
    print("feedback-aware G2 parents:", report.g2_parent_count)
    print("feedback changed parent set:", report.feedback_changed_parent_set)
    print("feedback added:", report.feedback_parent_added_idea_ids)
    print("feedback dropped:", report.feedback_parent_dropped_idea_ids)
    print("G2 profiles:", report.g2_parent_profile_counts)
    print("legacy portfolio overlap:", report.legacy_portfolio_overlap_count)
    if resolution is not None:
        print("residual auto-resolution ambiguous:", resolution.report.residual_ambiguous)
        print(
            "prospective auto-resolution ambiguous hypotheses:",
            resolution.report.prospective_ambiguous_hypothesis_ids,
        )
    print("NEW_LLM_CALLS=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
