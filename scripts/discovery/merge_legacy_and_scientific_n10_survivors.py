from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    merge_legacy_and_scientific_n10_survivors,
    resolve_strict_legacy_n10_portfolio,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge strict legacy N10 survivors with independently N10-authorized "
            "scientific cross-lane synthesis survivors. Legacy artifacts are not mutated."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--scientific-source-portfolio", required=True, type=Path)
    parser.add_argument("--scientific-survivors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()

    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    source = HypothesisPortfolio.model_validate_json(
        args.scientific_source_portfolio.read_text(encoding="utf-8")
    )
    survivors = HypothesisPortfolio.model_validate_json(
        args.scientific_survivors.read_text(encoding="utf-8")
    )
    legacy_resolution = resolve_strict_legacy_n10_portfolio(
        run_dir=args.run_dir
    )
    merged, report = merge_legacy_and_scientific_n10_survivors(
        context=context,
        legacy_resolution=legacy_resolution,
        scientific_source_portfolio=source,
        scientific_survivors=survivors,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        merged.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Legacy + scientific strict-N10 merge complete")
    print("Legacy authority:", report.legacy_resolution.authority_kind)
    print("Legacy survivors:", report.legacy_survivor_count)
    print(
        "Scientific synthesis survivors:",
        report.scientific_synthesis_survivor_count,
    )
    print("Merged survivors:", report.merged_survivor_count)
    print("Legacy production selection mutated: false")
    print("Merged portfolio:", args.output)
    print("Authority report:", args.report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
