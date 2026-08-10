from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.dual_hypothesis_context import DualHypothesisContext
from dac_her.hypothesis_contracts import HypothesisPortfolio
from dac_her.internal_novelty import InternalNoveltyAssessor
from dac_her.node_mapping import NodeMapper


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assess corpus-internal prior-art overlap for a hypothesis portfolio. "
            "This does NOT assess external-literature novelty."
        )
    )
    parser.add_argument("--dual-context", required=True, type=Path)
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.88)
    parser.add_argument("--extension-threshold", type=float, default=0.80)
    parser.add_argument("--route-reconstruction-threshold", type=float, default=0.80)
    parser.add_argument("--route-extension-threshold", type=float, default=0.50)
    parser.add_argument("--discovery-alignment-threshold", type=float, default=0.70)
    parser.add_argument("--top-k-nodes", type=int, default=12)
    args = parser.parse_args()

    dual = DualHypothesisContext.model_validate_json(args.dual_context.read_text(encoding="utf-8"))
    portfolio = HypothesisPortfolio.model_validate_json(args.portfolio.read_text(encoding="utf-8"))
    mapper = NodeMapper.from_directory(args.index_dir, device=args.device)
    report = InternalNoveltyAssessor(
        node_near_duplicate_threshold=args.near_duplicate_threshold,
        node_extension_threshold=args.extension_threshold,
        route_reconstruction_threshold=args.route_reconstruction_threshold,
        route_extension_threshold=args.route_extension_threshold,
        discovery_alignment_threshold=args.discovery_alignment_threshold,
        top_k_nodes=args.top_k_nodes,
    ).assess(dual, portfolio, mapper)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print("Internal novelty assessment complete")
    print("Report ID:", report.report_id)
    print("External novelty:", report.external_novelty_status)
    for card in report.cards:
        route_coverage = (
            card.strongest_route_match.premise_coverage
            if card.strongest_route_match is not None
            else 0.0
        )
        print(
            f"- {card.hypothesis_id}: {card.status}; "
            f"max_node_sim={card.max_node_similarity:.3f}; "
            f"route_coverage={route_coverage:.2f}"
        )
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
