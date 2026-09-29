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
            "Replay hypothesis relation-coverage graph v2 and the "
            "depth-aware NegativeSpace Planner on an existing run. "
            "No new retrieval or LLM calls."
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

    v1 = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    conceptual = None
    if args.conceptual_knownness is not None:
        conceptual = json.loads(
            args.conceptual_knownness.read_text(encoding="utf-8")
        )

    v2 = build_hypothesis_causal_edge_graph_novelty_depth_v2(
        context=context,
        portfolio=portfolio,
        external_report=external,
        v1_report=v1,
        conceptual_knownness=conceptual,
    )

    planner = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=v2,
    )

    execution = evaluate_negative_space_planner_execution(
        plan=planner,
        refinement_report=refinement,
    )

    v2_path = (
        run
        / "novelty_refinement_a6.hypothesis_edge_graph_v2.json"
    )
    planner_path = (
        run
        / "novelty_refinement_a6."
        "negative_space_planner.depth_v3.plan.json"
    )
    execution_path = (
        run
        / "novelty_refinement_a6."
        "negative_space_planner.depth_v3.execution.json"
    )

    _exclusive(v2_path, v2)
    _exclusive(planner_path, planner)
    _exclusive(execution_path, execution)

    print("=== HYPOTHESIS EDGE-GRAPH V2 REPLAY ===")
    print("depth:", v2.novelty_depth_counts)
    print("advisory:", v2.planner_advisory_counts)
    print(
        "cross-claim backbone:",
        v2.cross_claim_backbone_hypothesis_count,
    )
    print(
        "higher-order gaps:",
        v2.higher_order_gap_hypothesis_count,
    )
    print(
        "shallow local extensions:",
        v2.shallow_local_extension_count,
    )
    print("weak bridges:", v2.weak_bridge_hypothesis_count)
    print(
        "conceptual signals:",
        v2.conceptual_knownness_signal_count,
    )

    print()
    print("per hypothesis:")
    for row in v2.profiles:
        print(
            " ",
            row.title,
            "| v1:", row.v1_novelty_depth_class,
            "| v2:", row.novelty_depth_class,
            "| advisory:", row.planner_advisory,
            "| cross-claim:", row.cross_claim_backbone_used,
            "| backbone:", row.known_backbone_claim_ids,
            "| weak:", row.weak_bridge_claim_ids,
        )

    print()
    print("planner depth-aware:", planner.novelty_depth_profile_consumed)
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
    print("v2 artifact:", v2_path)
    print("planner artifact:", planner_path)
    print("execution artifact:", execution_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
