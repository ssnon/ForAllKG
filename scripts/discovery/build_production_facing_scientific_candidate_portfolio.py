from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
)
from pipeline_core.discovery.reframing.cross_lane_portfolio import (
    CrossLaneScientificReasoningShadowPortfolio,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    build_production_facing_candidate_portfolio,
)
from pipeline_core.discovery.reframing.proxy_challenge import ProxyChallengeRunReport
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframingShadowReport,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project relational and scientific-reframing candidates into a common "
            "production-facing shadow contract without ranking, synthesis, pruning, or selection."
        )
    )
    parser.add_argument("--cross-lane-portfolio", required=True, type=Path)
    parser.add_argument("--relational-portfolio", required=True, type=Path)
    parser.add_argument("--reframe-shadow", required=True, type=Path)
    parser.add_argument("--proxy-shadow", type=Path, default=None)
    parser.add_argument("--contradiction-shadow", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cross_lane_path = args.cross_lane_portfolio.expanduser().resolve()
    relational_path = args.relational_portfolio.expanduser().resolve()
    reframe_path = args.reframe_shadow.expanduser().resolve()

    cross_lane = CrossLaneScientificReasoningShadowPortfolio.model_validate_json(
        cross_lane_path.read_text(encoding="utf-8")
    )
    relational = HypothesisPortfolio.model_validate_json(
        relational_path.read_text(encoding="utf-8")
    )
    reframe = ScientificReframingShadowReport.model_validate_json(
        reframe_path.read_text(encoding="utf-8")
    )
    proxy = (
        ProxyChallengeRunReport.model_validate_json(
            args.proxy_shadow.expanduser().resolve().read_text(encoding="utf-8")
        )
        if args.proxy_shadow is not None
        else None
    )
    contradiction = (
        ContradictionResolutionRunReport.model_validate_json(
            args.contradiction_shadow.expanduser().resolve().read_text(encoding="utf-8")
        )
        if args.contradiction_shadow is not None
        else None
    )

    portfolio = build_production_facing_candidate_portfolio(
        cross_lane_portfolio=cross_lane,
        relational_portfolio=relational,
        reframe_shadow=reframe,
        proxy_shadow=proxy,
        contradiction_shadow=contradiction,
        source_cross_lane_portfolio_file_sha256=_sha256(cross_lane_path),
    )
    output = args.output or cross_lane_path.with_name(
        "scientific_production_facing_candidate_portfolio.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(portfolio.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Production-facing scientific candidate portfolio complete")
    print("LLM calls: 0")
    print(f"Task: {portfolio.source_task_id}")
    print(f"Candidates: {portfolio.candidate_count}")
    print(f"Relational: {portfolio.relational_candidate_count}")
    print(f"Reframing: {portfolio.reframing_candidate_count}")
    print(f"Source kinds: {portfolio.source_kind_counts}")
    print("Common downstream contract materialized: true")
    print("Cross-lane synthesis performed: false")
    print("Scientific quality ranking performed: false")
    print("Final hypothesis selection performed: false")
    print("Production selection changed: false")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
