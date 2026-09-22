from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    ScientificSynthesisNoveltyCertificationReport,
    merge_legacy_strict_with_scientific_certification,
    resolve_strict_legacy_n10_portfolio,
)


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
            "Merge strict legacy N10 authority with certification-only "
            "scientific synthesis outputs into two views: all discovery "
            "candidates and positive-authority novelty-certified hypotheses."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--scientific-candidates", required=True, type=Path)
    parser.add_argument("--scientific-certified", required=True, type=Path)
    parser.add_argument("--certification-report", required=True, type=Path)
    parser.add_argument("--output-candidate-portfolio", required=True, type=Path)
    parser.add_argument("--output-certified-portfolio", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()

    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    candidates = HypothesisPortfolio.model_validate_json(
        args.scientific_candidates.read_text(encoding="utf-8")
    )
    certified = HypothesisPortfolio.model_validate_json(
        args.scientific_certified.read_text(encoding="utf-8")
    )
    report = ScientificSynthesisNoveltyCertificationReport.model_validate_json(
        args.certification_report.read_text(encoding="utf-8")
    )

    legacy = resolve_strict_legacy_n10_portfolio(run_dir=args.run_dir)
    merged_candidates, merged_certified, merged_report = (
        merge_legacy_strict_with_scientific_certification(
            context=context,
            legacy_resolution=legacy,
            scientific_candidates=candidates,
            scientific_certified=certified,
            certification_report=report,
        )
    )

    _write(args.output_candidate_portfolio, merged_candidates)
    _write(args.output_certified_portfolio, merged_certified)
    _write(args.report_output, merged_report)

    print("Legacy + scientific certification-only merge complete")
    print("Legacy strict authority:", legacy.authority_kind)
    print("Legacy strict hypotheses:", merged_report.legacy_strict_authority_count)
    print("Scientific candidates:", merged_report.scientific_candidate_count)
    print("Scientific certified:", merged_report.scientific_certified_count)
    print("Scientific unresolved:", merged_report.scientific_unresolved_count)
    print("Scientific rejected:", merged_report.scientific_rejected_count)
    print("Merged discovery candidates:", merged_report.merged_candidate_count)
    print("Merged novelty-certified:", merged_report.merged_certified_count)
    print("Candidate retention is novelty authority: false")
    print("Candidate portfolio:", args.output_candidate_portfolio)
    print("Certified portfolio:", args.output_certified_portfolio)
    print("Authority report:", args.report_output)


if __name__ == "__main__":
    main()
