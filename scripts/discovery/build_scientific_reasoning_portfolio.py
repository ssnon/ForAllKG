from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.mode_contrast import ReasoningModeContrastReport
from pipeline_core.discovery.reframing.reasoning_portfolio import (
    build_unified_reasoning_portfolio,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a zero-LLM shadow portfolio across scientific reframing "
            "reasoning modes without ranking, winner selection, or redundancy pruning."
        )
    )
    parser.add_argument("--mode-contrast", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    contrast = ReasoningModeContrastReport.model_validate_json(
        args.mode_contrast.read_text(encoding="utf-8")
    )
    portfolio = build_unified_reasoning_portfolio(contrast)
    output = args.output or args.mode_contrast.with_name(
        "scientific_reframing_reasoning_portfolio.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(portfolio.model_dump_json(indent=2), encoding="utf-8")

    print("Unified scientific reframing reasoning portfolio complete")
    print("LLM calls: 0")
    print(f"Task: {portfolio.source_task_id}")
    print(f"Candidates preserved: {portfolio.candidate_count}")
    print(
        "Reasoning modes occupied: "
        f"{portfolio.occupied_reasoning_mode_count}/{len(portfolio.slots)}"
    )
    for slot in portfolio.slots:
        state = ",".join(slot.candidate_ids) if slot.candidate_ids else "EMPTY"
        print(f"  {slot.slot_id}: {state}")
    print(
        "Scientific-neighborhood clusters (diagnostic heuristic): "
        f"{portfolio.scientific_neighborhood_cluster_count}"
    )
    for cluster in portfolio.scientific_neighborhood_clusters:
        print(
            f"  {cluster.cluster_id}: candidates={len(cluster.candidate_ids)}; "
            f"modes={len(cluster.representation_transforms)}; "
            f"relations={cluster.pair_relations}"
        )
    print("Candidate ranking performed: false")
    print("Redundancy-based candidate dropping performed: false")
    print("Relational discovery lane integrated: false")
    print("No scientific quality ranking or production selection was performed.")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
