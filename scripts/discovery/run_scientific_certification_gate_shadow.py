from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.positive_nonobviousness_adjudication import (
    PositiveNonObviousnessAdjudicationReport,
)
from pipeline_core.discovery.scientific_certification_gate import (
    build_scientific_certification_gate_report,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    ScientificHypothesisEvidenceAggregationReport,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed search-bounded scientific certification "
            "gate from evidence aggregation, external novelty coverage, and "
            "positive non-obviousness authority. This remains shadow-only."
        )
    )
    parser.add_argument(
        "--aggregation",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--nonobviousness",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--external-novelty",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    aggregation = (
        ScientificHypothesisEvidenceAggregationReport.model_validate_json(
            args.aggregation.read_text(encoding="utf-8")
        )
    )
    nonobviousness = (
        PositiveNonObviousnessAdjudicationReport.model_validate_json(
            args.nonobviousness.read_text(encoding="utf-8")
        )
    )
    external_novelty = ExternalNoveltyReport.model_validate_json(
        args.external_novelty.read_text(encoding="utf-8")
    )

    report = build_scientific_certification_gate_report(
        aggregation=aggregation,
        nonobviousness=nonobviousness,
        external_novelty=external_novelty,
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

    print("Scientific certification gate shadow complete")
    print("Report:", report.report_id)
    print("Hypotheses:", report.hypothesis_count)
    print("Certified:", report.certified_count)
    print("Unresolved:", report.unresolved_count)
    print("Rejected:", report.rejected_count)
    print("Decision counts:", report.decision_counts)

    for row in report.decisions:
        print()
        print("Hypothesis:", row.hypothesis_id)
        print("Decision:", row.decision)
        print(
            "Bounded review closure:",
            row.bounded_closure_state,
        )
        print(
            "Bounded external distinctness:",
            row.bounded_external_distinctness_state,
        )
        print(
            "Positive non-obviousness:",
            row.positive_nonobviousness_authority_state,
        )
        print(
            "Fatal blocker:",
            row.fatal_blocker_state,
        )
        print(
            "External novelty status:",
            row.external_novelty_status,
        )
        print(
            "External search coverage sufficient:",
            row.external_search_coverage_sufficient,
        )
        print(
            "Positive non-obviousness gate:",
            row.positive_nonobviousness_gate_state,
        )
        print(
            "Reasons:",
            row.reason_codes,
        )

    print()
    print("Certification scope: search-bounded, not literature-wide proof")
    print("Closure is bounded review closure: true")
    print("External distinctness is search-bounded: true")
    print("Positive non-obviousness requires explicit authority: true")
    print("Absence alone can certify: false")
    print("Direct prior art is rejection authority: true")
    print("Fatal contradiction is rejection authority: true")
    print("Production authority created: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
