from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    filter_scientific_synthesis_portfolio_by_n10,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only scientific synthesis candidates with strict first-pass "
            "role-aware N10 positive authority. CONDITIONAL candidates are rejected."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--production-gate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()

    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    gate = json.loads(args.production_gate.read_text(encoding="utf-8"))
    if not isinstance(gate, dict):
        raise ValueError("production gate must be a JSON object")

    survivors, report = filter_scientific_synthesis_portfolio_by_n10(
        portfolio=portfolio,
        production_gate=gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        survivors.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Scientific synthesis strict N10 filtering complete")
    print("Candidates:", report.candidate_count)
    print("First-pass positive survivors:", report.survivor_count)
    print("Conditional rejected:", report.conditional_rejected_count)
    print("Ineligible rejected:", report.ineligible_rejected_count)
    print("Bounded continuation offered: false")
    print("Survivors:", args.output)
    print("Report:", args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
