from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.evidence_family_selection import (
    FamilyPremiseSelectionReport,
)
from pipeline_core.discovery.hypothesis_evidence_diversity import (
    HypothesisEvidenceDiversityReport,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyGapPlan,
)
from pipeline_core.discovery.portfolio_diversity_shadow import (
    build_portfolio_diversity_shadow_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine frozen premise-diversity diagnostics, optional "
            "evidence-family pseudo-diversity guards, and explicit novelty "
            "sharpening-operator opportunities into one diagnostic-only "
            "portfolio diversity report."
        )
    )
    parser.add_argument(
        "--evidence-diversity",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--family-selection",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--novelty-gap-plan",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--high-overlap-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    evidence = (
        HypothesisEvidenceDiversityReport
        .model_validate_json(
            args.evidence_diversity
            .read_text(encoding="utf-8")
        )
    )

    family = (
        None
        if args.family_selection is None
        else FamilyPremiseSelectionReport
        .model_validate_json(
            args.family_selection
            .read_text(encoding="utf-8")
        )
    )

    gap_plan = (
        None
        if args.novelty_gap_plan is None
        else NoveltyGapPlan.model_validate_json(
            args.novelty_gap_plan
            .read_text(encoding="utf-8")
        )
    )

    report = (
        build_portfolio_diversity_shadow_report(
            evidence=evidence,
            family_selection=family,
            novelty_gap_plan=gap_plan,
            high_overlap_threshold=(
                args.high_overlap_threshold
            ),
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=== PORTFOLIO DIVERSITY SHADOW ==="
    )
    print(
        "evidence recommendation:",
        report.evidence_recommendation,
    )
    print(
        "operator recommendation:",
        report.operator_recommendation,
    )
    print(
        "eligible/used/unused:",
        report.eligible_statement_count,
        "/",
        report.used_statement_count,
        "/",
        report.unused_eligible_statement_count,
    )
    print(
        "exact duplicate groups:",
        report.exact_premise_set_duplicate_group_count,
    )
    print(
        "max pairwise Jaccard:",
        f"{report.max_pairwise_statement_jaccard:.3f}",
    )
    print(
        "family pseudo-diversity guard:",
        report.family_pseudo_diversity_guard_available,
    )
    print(
        "operator opportunities:",
        report.operator_opportunities,
    )
    print(
        "diagnostic only; selection unchanged."
    )
    print(
        "artifact:",
        args.output,
    )
    print(
        "PORTFOLIO_DIVERSITY_SHADOW_COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
