from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
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
            "Replay novelty-depth/edge coverage and depth-aware "
            "NegativeSpace Planner on an existing E2E run without "
            "new retrieval or LLM calls."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
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

    depth = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    planner = build_negative_space_planner_plan(
        context=context,
        portfolio=portfolio,
        gap_plan=gap_plan,
        novelty_depth_profile=depth,
    )

    execution = evaluate_negative_space_planner_execution(
        plan=planner,
        refinement_report=refinement,
    )

    depth_path = (
        run / "novelty_refinement_a6.novelty_depth_edge_coverage.json"
    )
    planner_path = (
        run / "novelty_refinement_a6.negative_space_planner.depth_v2.plan.json"
    )
    execution_path = (
        run
        / "novelty_refinement_a6.negative_space_planner.depth_v2.execution.json"
    )

    _exclusive(depth_path, depth)
    _exclusive(planner_path, planner)
    _exclusive(execution_path, execution)

    print("=== DEPTH-AWARE NOVELTY PLANNER REPLAY ===")
    print("depth:", depth.novelty_depth_counts)
    print("advisory:", depth.planner_advisory_counts)
    print("weak bridge:", depth.weak_bridge_hypothesis_count)
    print("planner targets:", planner.target_count)
    print("planner depth-aware:", planner.novelty_depth_profile_consumed)
    print("planner advisory:", planner.novelty_depth_advisory_counts)
    print("execution match:", execution.match_count)
    print(
        "execution match after fallback:",
        execution.match_after_fallback_count,
    )
    print("execution diverged:", execution.diverged_count)
    print("execution not comparable:", execution.not_comparable_count)
    print("planned:", execution.planned_action_counts)
    print("observed:", execution.observed_action_counts)
    print("depth artifact:", depth_path)
    print("planner artifact:", planner_path)
    print("execution artifact:", execution_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
