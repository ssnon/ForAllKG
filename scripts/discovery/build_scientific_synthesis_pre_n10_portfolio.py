from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.reframing.cross_lane_synthesis import (
    CrossLaneSynthesisShadowReport,
)
from pipeline_core.discovery.reframing.pre_n10_synthesis_bridge import (
    build_pre_n10_synthesis_portfolio,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project cross-lane synthesized hypotheses into a conservative "
            "HypothesisPortfolio suitable for fresh external novelty/N10 evaluation. "
            "This projection grants no novelty or production authority."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--alpha6-portfolio", required=True, type=Path)
    parser.add_argument("--synthesis-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lineage-output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    alpha6 = HypothesisPortfolio.model_validate_json(
        args.alpha6_portfolio.read_text(encoding="utf-8")
    )
    synthesis = CrossLaneSynthesisShadowReport.model_validate_json(
        args.synthesis_report.read_text(encoding="utf-8")
    )
    portfolio, report = build_pre_n10_synthesis_portfolio(
        context=context,
        alpha6_portfolio=alpha6,
        synthesis_report=synthesis,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        portfolio.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.lineage_output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Scientific synthesis pre-N10 projection complete")
    print("Projected hypotheses:", len(portfolio.hypotheses))
    print("Novelty assessment performed: false")
    print("N10 authority created: false")
    print("Production selection changed: false")
    print("Portfolio:", args.output)
    print("Lineage:", args.lineage_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
