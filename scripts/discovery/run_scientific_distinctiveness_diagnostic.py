from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.scientific_distinctiveness import (
    ScientificDistinctivenessAnalyzer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostic-only scientific-distinctiveness "
            "review from frozen external-novelty artifacts. "
            "This runner performs no retrieval and no model review."
        )
    )

    parser.add_argument(
        "--external-report",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--external-query-plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--external-prior-art",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(
            encoding="utf-8"
        )
    )

    plan = LiteratureQueryPlan.model_validate_json(
        args.external_query_plan.read_text(
            encoding="utf-8"
        )
    )

    packet = PriorArtPacket.model_validate_json(
        args.external_prior_art.read_text(
            encoding="utf-8"
        )
    )

    result = (
        ScientificDistinctivenessAnalyzer()
        .build(
            report,
            plan,
            packet,
        )
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result.model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "Scientific distinctiveness diagnostic complete"
    )

    print(
        "Source portfolio:",
        result.source_portfolio_id,
    )

    print(
        "Source external novelty report:",
        result.source_external_novelty_report_id,
    )

    print(
        "Reviews:",
        len(
            result.reviews
        ),
    )

    print(
        "Evidence patterns:",
        result.evidence_pattern_counts,
    )

    for index, review in enumerate(
        result.reviews,
        start=1,
    ):
        print()

        print(
            f"[{index}]",
            review.evidence_pattern,
            "| external=",
            review.external_novelty_status,
        )

        print(
            "    ",
            review.title,
        )

        print(
            "     direct_core=",
            (
                f"{review.direct_prior_art_core_claim_count}/"
                f"{review.core_claim_count}"
            ),
            "relation_backed_core=",
            (
                f"{review.relation_backed_core_claim_count}/"
                f"{review.core_claim_count}"
            ),
            "lower_order_core=",
            (
                f"{review.lower_order_supported_core_claim_count}/"
                f"{review.core_claim_count}"
            ),
        )

        print(
            "     horg_claims=",
            review.higher_order_relational_gap_claim_count,
            "directional_counterevidence_works=",
            (
                review
                .directional_counterevidence_unique_work_count
            ),
            "coverage_sufficient=",
            review.search_coverage_sufficient,
        )

    print()
    print(
        "SEMANTIC_DIMENSIONS_ASSESSED=False"
    )
    print(
        "RETRIEVAL_PERFORMED=False"
    )
    print(
        "MODEL_REVIEW_PERFORMED=False"
    )
    print(
        "ACTION_POLICY_APPLIED=False"
    )
    print(
        "SCIENTIFIC_SELECTION_CHANGED=False"
    )
    print(
        "Saved:",
        args.output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
