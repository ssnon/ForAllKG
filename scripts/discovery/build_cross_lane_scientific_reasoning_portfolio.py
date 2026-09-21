from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
)
from pipeline_core.discovery.reframing.cross_lane_portfolio import (
    build_cross_lane_scientific_reasoning_portfolio,
)
from pipeline_core.discovery.reframing.proxy_challenge import ProxyChallengeRunReport
from pipeline_core.discovery.reframing.reasoning_portfolio import (
    UnifiedScientificReasoningShadowPortfolio,
)
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
            "Assemble a zero-LLM provenance-safe envelope over the existing "
            "relational discovery portfolio and scientific reframing shadow "
            "portfolio without retyping, ranking, or cross-lane pruning."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--relational-portfolio", required=True, type=Path)
    parser.add_argument("--reframing-portfolio", required=True, type=Path)
    parser.add_argument("--reframe-shadow", required=True, type=Path)
    parser.add_argument("--proxy-shadow", type=Path, default=None)
    parser.add_argument("--contradiction-shadow", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    context = HypothesisContext.model_validate_json(args.context.read_text(encoding="utf-8"))
    relational = HypothesisPortfolio.model_validate_json(
        args.relational_portfolio.read_text(encoding="utf-8")
    )
    reframing = UnifiedScientificReasoningShadowPortfolio.model_validate_json(
        args.reframing_portfolio.read_text(encoding="utf-8")
    )
    reframe_shadow = ScientificReframingShadowReport.model_validate_json(
        args.reframe_shadow.read_text(encoding="utf-8")
    )
    proxy_shadow = (
        ProxyChallengeRunReport.model_validate_json(
            args.proxy_shadow.read_text(encoding="utf-8")
        )
        if args.proxy_shadow is not None
        else None
    )
    contradiction_shadow = (
        ContradictionResolutionRunReport.model_validate_json(
            args.contradiction_shadow.read_text(encoding="utf-8")
        )
        if args.contradiction_shadow is not None
        else None
    )

    portfolio = build_cross_lane_scientific_reasoning_portfolio(
        context=context,
        relational_portfolio=relational,
        reframing_portfolio=reframing,
        reframe_shadow=reframe_shadow,
        proxy_shadow=proxy_shadow,
        contradiction_shadow=contradiction_shadow,
        source_context_file_sha256=_sha256(args.context),
        relational_portfolio_file_sha256=_sha256(args.relational_portfolio),
        reframing_portfolio_file_sha256=_sha256(args.reframing_portfolio),
    )
    output = args.output or args.reframing_portfolio.with_name(
        "scientific_cross_lane_reasoning_portfolio.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(portfolio.model_dump_json(indent=2), encoding="utf-8")

    print("Cross-lane scientific reasoning portfolio complete")
    print("LLM calls: 0")
    print(f"Task: {portfolio.source_task_id}")
    print(f"Context: {portfolio.source_context_id}")
    print(f"Lanes: {portfolio.lane_count}")
    print(
        "Relational discovery entries: "
        f"{portfolio.relational_entry_count} "
        f"(source={portfolio.relational_lane.source_portfolio_id})"
    )
    print(
        "Scientific reframing entries: "
        f"{portfolio.reframing_entry_count} "
        f"(source={portfolio.reframing_lane.source_portfolio_id})"
    )
    print(f"Total entries: {portfolio.entry_count}")
    print("Source lane schemas preserved: true")
    print("Cross-lane ranking performed: false")
    print("Cross-lane redundancy pruning performed: false")
    print("Integrated portfolio is production selection: false")
    print("Production selection changed: false")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
