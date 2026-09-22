from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    split_scientific_synthesis_portfolio_by_n10_certification,
)


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve the scientific synthesis candidate portfolio while "
            "emitting an N10 novelty-certification report and a separate "
            "positive-authority certified subset."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--production-gate", required=True, type=Path)
    parser.add_argument("--output-candidate-portfolio", required=True, type=Path)
    parser.add_argument("--output-certification-report", required=True, type=Path)
    parser.add_argument("--output-certified-portfolio", required=True, type=Path)
    args = parser.parse_args()

    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    candidate, certified, report = (
        split_scientific_synthesis_portfolio_by_n10_certification(
            portfolio=portfolio,
            production_gate=_load_object(args.production_gate),
        )
    )

    _write(args.output_candidate_portfolio, candidate)
    _write(args.output_certification_report, report)
    _write(args.output_certified_portfolio, certified)

    print("Scientific synthesis N10 certification-only assessment complete")
    print("Scientific candidates preserved:", report.candidate_count)
    print("Novelty-certified:", report.certified_count)
    print("Novelty-unresolved:", report.unresolved_count)
    print("Novelty-rejected:", report.rejected_count)
    print("Candidate retention is novelty authority: false")
    print("Candidate portfolio:", args.output_candidate_portfolio)
    print("Certification report:", args.output_certification_report)
    print("Certified subset:", args.output_certified_portfolio)


if __name__ == "__main__":
    main()
