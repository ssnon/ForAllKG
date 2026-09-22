from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphReport,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    ScientificHypothesisEvidenceAggregationReport,
)
from pipeline_core.discovery.positive_nonobviousness_basis import (
    build_positive_nonobviousness_basis_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed positive non-obviousness basis registry from "
            "adjudicated directional/counter/conflict evidence. This creates "
            "basis candidates only; it does not infer non-obviousness."
        )
    )
    parser.add_argument(
        "--evidence-graph",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--aggregation",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    graph = ScientificClaimEvidenceGraphReport.model_validate_json(
        args.evidence_graph.read_text(encoding="utf-8")
    )
    aggregation = (
        ScientificHypothesisEvidenceAggregationReport.model_validate_json(
            args.aggregation.read_text(encoding="utf-8")
        )
    )

    report = build_positive_nonobviousness_basis_report(
        graph=graph,
        aggregation=aggregation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
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

    print("Positive non-obviousness basis shadow complete")
    print("Report:", report.report_id)
    print("Hypotheses:", report.hypothesis_count)
    print("Claims:", report.claim_count)
    print(
        "Basis evidence items:",
        report.basis_evidence_count,
    )
    print(
        "Qualifying basis claims:",
        report.qualifying_basis_claim_count,
    )
    print(
        "Basis states:",
        report.positive_basis_state_counts,
    )
    print(
        "Readiness:",
        report.readiness_counts,
    )

    for summary in report.hypothesis_summaries:
        print()
        print("Hypothesis:", summary.hypothesis_id)
        print(
            "Positive basis state:",
            summary.positive_basis_state,
        )
        print(
            "Future adjudication readiness:",
            summary.future_adjudication_readiness,
        )
        print(
            "Qualifying basis claims:",
            summary.qualifying_basis_claim_ids,
        )
        print(
            "Nonqualifying tension claims:",
            summary.nonqualifying_tension_claim_ids,
        )
        print(
            "Direct prior-art blockers:",
            summary.direct_prior_art_blocker_claim_ids,
        )
        print(
            "Lower-order pressure claims:",
            summary.lower_order_pressure_claim_ids,
        )
        print(
            "Basis kinds:",
            summary.basis_kind_counts,
        )
        print(
            "Coverage:",
            summary.epistemic_coverage_profile,
        )

        rows = [
            row
            for row in report.claim_records
            if row.hypothesis_id == summary.hypothesis_id
        ]
        for row in rows:
            print(
                " ",
                row.claim_id,
                "| role=",
                row.novelty_selection_role,
                "| relevance=",
                row.role_relevance,
                "| basis_state=",
                row.basis_state,
                "| basis_kinds=",
                row.basis_kinds,
                "| direct_blocker=",
                row.direct_prior_art_blocker_present,
                "| classified_fraction=",
                round(row.classification_fraction, 4),
            )

    print()
    print("Direct prior art is positive basis: false")
    print("Lower-order prior art is positive basis: false")
    print("Partial prior art is positive basis: false")
    print("Component-only is positive basis: false")
    print("Retrieval miss is positive basis: false")
    print("Positive basis is truth evidence: false")
    print("Positive basis is novelty verdict: false")
    print("Positive basis is non-obviousness verdict: false")
    print("Positive non-obviousness authority created: false")
    print("Certification performed: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
