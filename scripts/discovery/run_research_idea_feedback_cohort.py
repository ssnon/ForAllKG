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


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    label, raw = value.split("=", 1)
    label = label.strip()
    raw = raw.strip()
    if not label or not raw:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label, Path(raw).expanduser().resolve()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run SIS-v2.1.1 exact-lineage feedback resolution and feedback-aware "
            "G2 parent allocation across multiple existing scientific-portfolio "
            "cases. No new LLM or retrieval calls are made."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument(
        "--feedback-search-root",
        action="append",
        type=Path,
        required=True,
    )
    p.add_argument("--max-g2-parents", type=int, default=8)
    p.add_argument("--max-g2-per-profile", type=int, default=2)
    p.add_argument("--output", required=True, type=Path)
    return p


def _case_paths(root: Path) -> dict[str, Path]:
    portfolio = root / "scientific_portfolio_shadow"
    return {
        "population": root / "frontier_idea_population.shadow.json",
        "evolution": root / "idea_evolution.shadow.json",
        "pool": portfolio / "candidate_pool.json",
        "evaluation": portfolio / "evaluation.json",
        "selection": portfolio / "selection.json",
        "materialization": portfolio / "materialization.report.json",
        "output": portfolio / "sis_v2_1_1.feedback_aware.generational_shadow.json",
        "resolution": portfolio / "sis_v2_1_1.feedback_resolution.json",
    }


def main() -> int:
    args = parser().parse_args()
    if args.max_g2_parents < 1 or args.max_g2_per_profile < 1:
        raise ValueError("G2 parent limits must be >= 1")

    search_roots = [path.expanduser().resolve() for path in args.feedback_search_root]
    cohort: dict[str, Any] = {
        "schema_version": "research-idea-feedback-cohort-v1",
        "cases": {},
        "case_count": len(args.case),
        "new_llm_calls": False,
        "new_retrieval_calls": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
        "production_selection_authority": False,
    }

    for label, root in args.case:
        paths = _case_paths(root)
        missing = [str(path) for key, path in paths.items() if key not in {"output", "resolution"} and not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{label} missing required artifacts: {missing}")

        population = _load(paths["population"], FrontierIdeaPopulation)
        evolution = _load(paths["evolution"], IdeaEvolutionReport)
        pool = _load(paths["pool"], ScientificPortfolioCandidatePool)
        evaluation = _load(paths["evaluation"], ScientificPortfolioEvaluationReport)
        selection = _load(paths["selection"], ScientificPortfolioSelectionReport)
        materialization = _load(
            paths["materialization"],
            ScientificPortfolioMaterializationReport,
        )

        resolved = resolve_feedback_artifacts(
            materialization=materialization,
            search_roots=search_roots,
        )
        report = build_generational_idea_search_shadow(
            population=population,
            evolution_report=evolution,
            pool=pool,
            evaluation=evaluation,
            selection=selection,
            materialization=materialization,
            residual_state=resolved.residual_state,
            prospective_by_hypothesis=resolved.prospective_by_hypothesis,
            max_g2_parents=args.max_g2_parents,
            max_g2_per_profile=args.max_g2_per_profile,
        )
        _write(paths["resolution"], resolved.report)
        _write(paths["output"], report)

        coverage = report.feedback_coverage
        cohort["cases"][label] = {
            "case_root": str(root),
            "report": str(paths["output"]),
            "feedback_resolution_report": str(paths["resolution"]),
            "research_idea_count": report.research_idea_count,
            "generation0_idea_count": report.generation0_idea_count,
            "generation1_idea_count": report.generation1_idea_count,
            "identity_relation_counts": report.identity_relation_counts,
            "semantic_noop_count": report.semantic_noop_count,
            "operator_expectation_inconsistent_count": (
                report.operator_expectation_inconsistent_count
            ),
            "materialized_hypothesis_count": coverage.materialized_hypothesis_count,
            "residual_matched_hypothesis_count": (
                coverage.residual_matched_hypothesis_count
            ),
            "prospective_matched_hypothesis_count": (
                coverage.prospective_matched_hypothesis_count
            ),
            "residual_coverage_fraction": coverage.residual_coverage_fraction,
            "prospective_coverage_fraction": coverage.prospective_coverage_fraction,
            "residual_ambiguous": resolved.report.residual_ambiguous,
            "prospective_ambiguous_hypothesis_ids": (
                resolved.report.prospective_ambiguous_hypothesis_ids
            ),
            "baseline_priority_band_counts": report.baseline_priority_band_counts,
            "feedback_priority_band_counts": report.feedback_priority_band_counts,
            "baseline_g2_parent_count": report.baseline_g2_parent_count,
            "feedback_g2_parent_count": report.g2_parent_count,
            "feedback_changed_parent_set": report.feedback_changed_parent_set,
            "feedback_parent_added_idea_ids": report.feedback_parent_added_idea_ids,
            "feedback_parent_dropped_idea_ids": report.feedback_parent_dropped_idea_ids,
            "feedback_parent_overlap_count": report.feedback_parent_overlap_count,
            "g2_parent_profile_counts": report.g2_parent_profile_counts,
            "legacy_portfolio_overlap_count": report.legacy_portfolio_overlap_count,
        }

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print(
            "feedback coverage residual/prospective:",
            coverage.residual_matched_hypothesis_count,
            "/",
            coverage.prospective_matched_hypothesis_count,
            "of",
            coverage.materialized_hypothesis_count,
        )
        print("priority bands:", report.feedback_priority_band_counts)
        print("feedback changed G2 parents:", report.feedback_changed_parent_set)
        print("added:", report.feedback_parent_added_idea_ids)
        print("dropped:", report.feedback_parent_dropped_idea_ids)
        print("residual ambiguous:", resolved.report.residual_ambiguous)
        print(
            "prospective ambiguous:",
            resolved.report.prospective_ambiguous_hypothesis_ids,
        )

    _write(args.output.expanduser().resolve(), cohort)
    print()
    print("SIS-v2.1.1 feedback-aware cohort complete")
    print("cases:", len(args.case))
    print("NEW_LLM_CALLS=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
