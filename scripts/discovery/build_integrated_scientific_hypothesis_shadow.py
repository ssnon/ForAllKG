from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.cross_lane_synthesis import (
    CrossLaneSynthesisShadowReport,
)
from pipeline_core.discovery.reframing.integrated_hypothesis_shadow import (
    build_integrated_scientific_hypothesis_shadow,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a zero-LLM integrated final-hypothesis shadow view containing the legacy "
            "relational hypotheses plus every grounded cross-lane synthesis. Raw reframing "
            "candidates remain preserved in the source portfolio and production selection is unchanged."
        )
    )
    parser.add_argument("--candidate-portfolio", required=True, type=Path)
    parser.add_argument("--synthesis-report", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    candidate_path = args.candidate_portfolio.expanduser().resolve()
    synthesis_path = args.synthesis_report.expanduser().resolve()
    candidate_portfolio = ProductionFacingScientificCandidatePortfolio.model_validate_json(
        candidate_path.read_text(encoding="utf-8")
    )
    synthesis_report = CrossLaneSynthesisShadowReport.model_validate_json(
        synthesis_path.read_text(encoding="utf-8")
    )
    portfolio = build_integrated_scientific_hypothesis_shadow(
        candidate_portfolio=candidate_portfolio,
        synthesis_report=synthesis_report,
    )
    output = args.output or synthesis_path.with_name(
        "scientific_integrated_hypothesis_shadow.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(portfolio.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Integrated scientific hypothesis shadow complete")
    print("LLM calls: 0")
    print("Task:", portfolio.source_task_id)
    print("Legacy relational hypotheses:", portfolio.legacy_hypothesis_count)
    print("Cross-lane synthesized additions:", portfolio.synthesized_addition_count)
    print("Integrated shadow hypotheses:", portfolio.integrated_hypothesis_count)
    print("Raw reframing source candidates:", portfolio.raw_reframing_source_candidate_count)
    print(
        "Synthesis coverage: relational="
        f"{len(portfolio.synthesis_covered_relational_candidate_ids)}/"
        f"{portfolio.legacy_hypothesis_count}; reframing="
        f"{len(portfolio.synthesis_covered_reframing_candidate_ids)}/"
        f"{portfolio.raw_reframing_source_candidate_count}"
    )
    print(
        "Shadow output differs from legacy:",
        str(portfolio.shadow_output_differs_from_legacy).lower(),
    )
    for row in portfolio.synthesized_additions:
        print(f"  + {row.entry_id}: {row.title}")
        print(f"    kind={row.synthesis_kind}")
        print(f"    sources={','.join(row.source_candidate_ids)}")
    print("Raw reframing candidates discarded: false")
    print("Scientific quality ranking performed: false")
    print("Final production selection performed: false")
    print("Production selection changed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
