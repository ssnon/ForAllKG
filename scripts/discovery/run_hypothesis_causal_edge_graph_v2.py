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
from pipeline_core.discovery.novelty_depth_causal_edge_coverage import (
    NoveltyDepthCausalEdgeCoverageReport,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build hypothesis-level relation-coverage graph and novelty "
            "depth v2 by composing core/supporting claim coverage. "
            "No retrieval or LLM calls."
        )
    )
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--external-report", required=True, type=Path)
    p.add_argument("--v1-report", required=True, type=Path)
    p.add_argument("--conceptual-knownness", type=Path)
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
    v1 = NoveltyDepthCausalEdgeCoverageReport.model_validate_json(
        args.v1_report.read_text(encoding="utf-8")
    )

    conceptual = None
    if args.conceptual_knownness is not None:
        conceptual = json.loads(
            args.conceptual_knownness.read_text(encoding="utf-8")
        )

    report = build_hypothesis_causal_edge_graph_novelty_depth_v2(
        context=context,
        portfolio=portfolio,
        external_report=external,
        v1_report=v1,
        conceptual_knownness=conceptual,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== HYPOTHESIS CAUSAL-EDGE GRAPH & NOVELTY DEPTH V2 ===")
    print("hypotheses:", report.hypothesis_count)
    print("depth:", report.novelty_depth_counts)
    print("planner advisory:", report.planner_advisory_counts)
    print(
        "cross-claim backbone:",
        report.cross_claim_backbone_hypothesis_count,
    )
    print(
        "higher-order gaps:",
        report.higher_order_gap_hypothesis_count,
    )
    print(
        "shallow local extensions:",
        report.shallow_local_extension_count,
    )
    print(
        "weak bridges:",
        report.weak_bridge_hypothesis_count,
    )
    print(
        "conceptual signals:",
        report.conceptual_knownness_signal_count,
    )
    print(
        "conceptual sufficient:",
        report.conceptual_knownness_sufficient_count,
    )
    print("conceptual changed depth: 0")
    print("foundational knownness checked: false")
    print("novelty authority created: false")
    print("production selection changed: false")
    print("artifact:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
