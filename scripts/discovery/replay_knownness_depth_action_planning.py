from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_causal_edge_graph_v2 import (
    build_hypothesis_causal_edge_graph_novelty_depth_v2,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.knownness_depth_action_planning import (
    build_actionable_refinement_plan,
    build_knownness_depth_fusion,
)
from pipeline_core.discovery.negative_space_planner import (
    build_negative_space_planner_plan,
    evaluate_negative_space_planner_execution,
)
from pipeline_core.discovery.novelty_depth_causal_edge_coverage import (
    build_novelty_depth_causal_edge_coverage,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGapPlan,
    NoveltyRefinementReport,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Replay Knownness-Depth Fusion and Actionable Refinement "
            "Planning on an existing run. No new retrieval or LLM calls."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--conceptual-knownness", type=Path)
    return p.parse_args()


def _exclusive(path: Path, value) -> None:
    if path.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: " + str(path)
        )
    path.write_text(
        value.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    run = args.run_dir.expanduser().resolve()

    context = HypothesisContext.model_validate_json(
        (run / "hypothesis.context.json").read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        (run / "hypothesis_axis_a4.portfolio.json").read_text(
            encoding="utf-8"
        )
    )
    external = ExternalNoveltyReport.model_validate_json(
        (run / "external_novelty_a52.report.json").read_text(
            encoding="utf-8"
        )
    )
    gap_plan = NoveltyGapPlan.model_validate_json(
        (run / "novelty_refinement_a6.gap_plan.json").read_text(
            encoding="utf-8"
        )
    )
    refinement = NoveltyRefinementReport.model_validate_json(
        (run / "novelty_refinement_a6.report.json").read_text(
            encoding="utf-8"
        )
    )

    conceptual = None
    if args.conceptual_knownness is not None:
        conceptual = json.loads(
            args.conceptual_knownness.read_text(encoding="utf-8")
        )

    v1 = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    v2 = build_hypothesis_causal_edge_graph_novelty_depth_v2(
        context=context,
        portfolio=portfolio,
        external_report=external,
        v1_report=v1,
        conceptual_knownness=conceptual,
    )

    fusion = build_knownness_depth_fusion(
        edge_graph_report=v2,
    )

    action_plan = build_actionable_refinement_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        fusion_report=fusion,
        edge_graph_report=v2,
    )

    planner = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=v2,
        actionable_refinement_plan=action_plan,
    )

    execution = evaluate_negative_space_planner_execution(
        plan=planner,
        refinement_report=refinement,
    )

    fusion_path = (
        run
        / "novelty_refinement_a6.knownness_depth_fusion.json"
    )
    action_path = (
        run
        / "novelty_refinement_a6.actionable_refinement_plan.json"
    )
    planner_path = (
        run
        / "novelty_refinement_a6."
        "negative_space_planner.action_v4.plan.json"
    )
    execution_path = (
        run
        / "novelty_refinement_a6."
        "negative_space_planner.action_v4.execution.json"
    )

    _exclusive(fusion_path, fusion)
    _exclusive(action_path, action_plan)
    _exclusive(planner_path, planner)
    _exclusive(execution_path, execution)

    print("=== KNOWNNESS-DEPTH FUSION & ACTIONABLE REFINEMENT ===")
    print("fusion:", fusion.fusion_state_counts)
    print("conceptual:", fusion.conceptual_signal_counts)
    print("actions:", action_plan.action_counts)
    print(
        "operators:",
        action_plan.operator_candidate_counts,
    )
    print(
        "preferred operators:",
        action_plan.preferred_operator_counts,
    )

    print()
    print("per hypothesis:")
    fusion_by_id = {
        row.hypothesis_id: row
        for row in fusion.rows
    }
    for target in action_plan.targets:
        fused = fusion_by_id[target.hypothesis_id]
        print(
            " ",
            target.title,
            "| fusion:", fused.fusion_state,
            "| conceptual:", fused.conceptual_signal,
            "| action:", target.primary_action,
            "| operators:", target.operator_candidates,
            "| preferred:", target.preferred_operator,
            "| reaxis-capacity:",
            target.evidence_reaxis_capacity_present,
        )

    print()
    print("planner actionable:", planner.actionable_refinement_plan_consumed)
    print(
        "planner actionable actions:",
        planner.actionable_refinement_action_counts,
    )
    print("planned:", execution.planned_action_counts)
    print("observed:", execution.observed_action_counts)
    print("execution match:", execution.match_count)
    print(
        "execution match after fallback:",
        execution.match_after_fallback_count,
    )
    print("execution diverged:", execution.diverged_count)
    print(
        "execution not comparable:",
        execution.not_comparable_count,
    )

    print("fusion artifact:", fusion_path)
    print("action plan:", action_path)
    print("planner artifact:", planner_path)
    print("execution artifact:", execution_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
