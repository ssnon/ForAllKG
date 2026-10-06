from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.idea_evolution import IdeaEvolutionReport
from pipeline_core.discovery.research_idea_evolutionary_policy import (
    EvolutionSlotBudget,
    build_evolutionary_idea_search_shadow,
    derive_slot_budget,
)
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
            "Run SIS-v2.2 evolutionary-fertility shadow allocation across existing "
            "scientific-portfolio cases. Verification outcomes remain realization-"
            "scoped soft search observations unless an explicit idea-level hard "
            "constraint is supplied by a future authoritative boundary."
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
    p.add_argument("--exploit-slots", type=int, default=None)
    p.add_argument("--transform-slots", type=int, default=None)
    p.add_argument("--explore-slots", type=int, default=None)
    p.add_argument("--wildcard-slots", type=int, default=None)
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
        "source_v211": portfolio / "sis_v2_2.source_v2_1_1.generational_shadow.json",
        "resolution": portfolio / "sis_v2_2.feedback_resolution.json",
        "output": portfolio / "sis_v2_2.evolutionary_search_shadow.json",
    }


def _slot_budget(args) -> EvolutionSlotBudget:
    values = (
        args.exploit_slots,
        args.transform_slots,
        args.explore_slots,
        args.wildcard_slots,
    )
    if all(value is None for value in values):
        return derive_slot_budget(args.max_g2_parents)
    if any(value is None for value in values):
        raise ValueError(
            "custom slot budget requires all of --exploit-slots, "
            "--transform-slots, --explore-slots, --wildcard-slots"
        )
    return EvolutionSlotBudget(
        max_parent_budget=args.max_g2_parents,
        exploit_slots=args.exploit_slots,
        transform_slots=args.transform_slots,
        explore_slots=args.explore_slots,
        wildcard_slots=args.wildcard_slots,
    )


def main() -> int:
    args = parser().parse_args()
    if args.max_g2_parents < 1 or args.max_g2_per_profile < 1:
        raise ValueError("G2 limits must be >= 1")

    budget = _slot_budget(args)
    search_roots = [path.expanduser().resolve() for path in args.feedback_search_root]
    cohort: dict[str, Any] = {
        "schema_version": "research-idea-evolutionary-cohort-v1",
        "cases": {},
        "case_count": len(args.case),
        "slot_budget": budget.model_dump(mode="json"),
        "verification_is_not_fertility_authority": True,
        "soft_feedback_cannot_terminate_idea": True,
        "new_llm_calls": False,
        "new_retrieval_calls": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
        "production_selection_authority": False,
    }

    for label, root in args.case:
        paths = _case_paths(root)
        required = (
            "population",
            "evolution",
            "pool",
            "evaluation",
            "selection",
            "materialization",
        )
        missing = [str(paths[name]) for name in required if not paths[name].is_file()]
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
        source = build_generational_idea_search_shadow(
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
        report = build_evolutionary_idea_search_shadow(
            source,
            max_parent_budget=args.max_g2_parents,
            slot_budget=budget,
        )

        _write(paths["resolution"], resolved.report)
        _write(paths["source_v211"], source)
        _write(paths["output"], report)

        cohort["cases"][label] = {
            "case_root": str(root),
            "source_v2_1_1_report": str(paths["source_v211"]),
            "v2_2_report": str(paths["output"]),
            "feedback_resolution_report": str(paths["resolution"]),
            "feedback_coverage": source.feedback_coverage.model_dump(mode="json"),
            "v2_1_1_priority_band_counts": source.feedback_priority_band_counts,
            "realization_viability_counts": report.realization_viability_counts,
            "reproductive_value_counts": report.reproductive_value_counts,
            "transformation_pressure_counts": report.transformation_pressure_counts,
            "operator_hint_counts": report.operator_hint_counts,
            "v2_1_1_parent_idea_ids": report.v2_1_1_parent_idea_ids,
            "v2_2_parent_idea_ids": report.selected_parent_idea_ids,
            "v2_2_parent_channel_counts": report.selected_parent_channel_counts,
            "parent_set_changed": report.parent_set_changed,
            "parent_overlap_count": report.parent_overlap_count,
            "parent_added_idea_ids": report.parent_added_idea_ids,
            "parent_dropped_idea_ids": report.parent_dropped_idea_ids,
            "selected_low_viability_transform_count": (
                report.selected_low_viability_transform_count
            ),
            "selected_v2_1_1_low_priority_count": (
                report.selected_v2_1_1_low_priority_count
            ),
            "hard_constraint_termination_count": (
                report.hard_constraint_termination_count
            ),
        }

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print(
            "feedback coverage residual/prospective:",
            source.feedback_coverage.residual_matched_hypothesis_count,
            "/",
            source.feedback_coverage.prospective_matched_hypothesis_count,
            "of",
            source.feedback_coverage.materialized_hypothesis_count,
        )
        print("v2.1.1 priority bands:", source.feedback_priority_band_counts)
        print("realization viability:", report.realization_viability_counts)
        print("reproductive value:", report.reproductive_value_counts)
        print("transformation pressure:", report.transformation_pressure_counts)
        print("target slots:", budget.model_dump(mode="json"))
        print("actual channels:", report.selected_parent_channel_counts)
        print("v2.2 changed parents:", report.parent_set_changed)
        print("added:", report.parent_added_idea_ids)
        print("dropped:", report.parent_dropped_idea_ids)
        print(
            "rescued low-viability transform parents:",
            report.selected_low_viability_transform_count,
        )
        print(
            "selected parents previously LOW in v2.1.1:",
            report.selected_v2_1_1_low_priority_count,
        )
        print("hard idea terminations:", report.hard_constraint_termination_count)

    _write(args.output.expanduser().resolve(), cohort)
    print()
    print("SIS-v2.2 evolutionary-fertility cohort complete")
    print("cases:", len(args.case))
    print("NEW_LLM_CALLS=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
