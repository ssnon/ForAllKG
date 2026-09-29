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
from pipeline_core.discovery.novelty_depth_causal_edge_coverage import (
    build_novelty_depth_causal_edge_coverage,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build hypothesis-level novelty depth and causal-edge coverage "
            "from existing grounded premises plus external novelty claim "
            "reviews. No retrieval or LLM calls."
        )
    )
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--external-report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.output.exists():
        raise RuntimeError(
            "output already exists; refusing overwrite: "
            + str(args.output)
        )

    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    external = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(encoding="utf-8")
    )

    if external.source_portfolio_id != portfolio.portfolio_id:
        raise RuntimeError(
            "external-report/source-portfolio mismatch: "
            f"{external.source_portfolio_id} != {portfolio.portfolio_id}"
        )

    report = build_novelty_depth_causal_edge_coverage(
        context=context,
        portfolio=portfolio,
        external_report=external,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== NOVELTY DEPTH & CAUSAL-EDGE COVERAGE ===")
    print("hypotheses:", report.hypothesis_count)
    print("depth:", report.novelty_depth_counts)
    print("planner advisory:", report.planner_advisory_counts)
    print("weak bridge:", report.weak_bridge_hypothesis_count)
    print(
        "higher-order gaps:",
        report.higher_order_gap_hypothesis_count,
    )
    print(
        "shallow local extensions:",
        report.shallow_local_extension_count,
    )
    print("foundational knownness checked: false")
    print("conceptual L1/L2/L3 integrated: false")
    print("ranking computed: false")
    print("novelty authority created: false")
    print("production selection changed: false")
    print("artifact:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
