from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.scientific_portfolio_production import (
    bind_scientific_portfolio_production,
    certify_scientific_portfolio_novelty,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authority-mode",
        choices=("certification_only", "hard_filter"),
        default="certification_only",
    )
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--n10-production-gate", type=Path, required=True)

    parser.add_argument("--output-candidate-portfolio", type=Path)
    parser.add_argument("--output-certification-report", type=Path)
    parser.add_argument("--output-certified-portfolio", type=Path)

    parser.add_argument("--output-portfolio", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()

    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    gate = _load(args.n10_production_gate)

    if args.authority_mode == "certification_only":
        required = (
            args.output_candidate_portfolio,
            args.output_certification_report,
            args.output_certified_portfolio,
        )
        if any(value is None for value in required):
            raise ValueError(
                "certification_only requires candidate/report/certified outputs"
            )

        report, candidate, certified = certify_scientific_portfolio_novelty(
            portfolio=portfolio,
            n10_production_gate=gate,
        )
        _write(args.output_candidate_portfolio, candidate)
        _write(args.output_certification_report, report)
        _write(args.output_certified_portfolio, certified)

        print("Scientific Portfolio N10 certification complete")
        print("Scientific candidates:", report.hypothesis_count)
        print("Novelty certified:", report.certified_count)
        print("Conditional retained:", report.conditional_count)
        print("Ineligible retained:", report.ineligible_count)
        print(
            "N10 candidate survival authority:",
            report.n10_candidate_survival_authority,
        )
        return 0

    if args.output_portfolio is None or args.output_report is None:
        raise ValueError(
            "hard_filter requires --output-portfolio and --output-report"
        )

    report, output = bind_scientific_portfolio_production(
        portfolio=portfolio,
        n10_production_gate=gate,
    )
    _write(args.output_portfolio, output)
    _write(args.output_report, report)
    print("Scientific Portfolio hard-filter binding complete")
    print("Selected:", report.production_selected_count)
    print("Blocked:", report.production_blocked_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
