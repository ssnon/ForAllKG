from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.scientific_claim_centrality import (
    ScientificClaimCentralityReport,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphReport,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    build_scientific_hypothesis_evidence_aggregation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate claim-level structural centrality and bounded "
            "adjudicated evidence pressure at hypothesis level without "
            "creating novelty or non-obviousness authority."
        )
    )
    parser.add_argument(
        "--evidence-graph",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--centrality",
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
    centrality = ScientificClaimCentralityReport.model_validate_json(
        args.centrality.read_text(encoding="utf-8")
    )

    report = build_scientific_hypothesis_evidence_aggregation(
        graph=graph,
        centrality=centrality,
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

    print("Scientific hypothesis evidence aggregation shadow complete")
    print("Report:", report.report_id)
    print("Hypotheses:", report.hypothesis_count)
    print("Claims:", report.claim_count)
    print(
        "Impact profiles:",
        report.hypothesis_profile_counts,
    )
    print(
        "Coverage profiles:",
        report.epistemic_coverage_profile_counts,
    )

    for hypothesis in report.hypothesis_aggregations:
        print()
        print("Hypothesis:", hypothesis.hypothesis_id)
        print("Topology:", hypothesis.topology_state)
        print(
            "Centrality degenerate:",
            hypothesis.centrality_is_degenerate,
        )
        print(
            "Evidence impact profile:",
            hypothesis.evidence_impact_profile,
        )
        print(
            "Weak signal profile:",
            hypothesis.weak_signal_profile,
        )
        print(
            "Epistemic coverage:",
            hypothesis.epistemic_coverage_profile,
        )
        print(
            "Novelty-bearing claims:",
            hypothesis.novelty_bearing_claim_ids,
        )
        print(
            "Highest-centrality claims:",
            hypothesis.highest_centrality_claim_ids,
        )
        print(
            "Strong-pressure claims:",
            hypothesis.strong_pressure_claim_ids,
        )
        print(
            "Novelty-bearing strong-pressure claims:",
            hypothesis.novelty_bearing_strong_pressure_claim_ids,
        )
        print(
            "Highest-centrality strong-pressure claims:",
            hypothesis.highest_centrality_strong_pressure_claim_ids,
        )
        print(
            "Weak/component signal claims:",
            hypothesis.weak_or_component_signal_claim_ids,
        )
        print(
            "Strong pressure kinds:",
            hypothesis.strong_pressure_kind_counts,
        )
        print(
            "Weak pressure kinds:",
            hypothesis.weak_pressure_kind_counts,
        )
        print(
            "Review priority:",
            hypothesis.review_priority_claim_ids,
        )

        rows = [
            row
            for row in report.claim_impacts
            if row.hypothesis_id == hypothesis.hypothesis_id
        ]
        for row in rows:
            print(
                " ",
                row.claim_id,
                "| role=",
                row.novelty_selection_role,
                "| centrality_rank=",
                row.centrality_rank_within_hypothesis,
                "| highest=",
                row.highest_centrality_within_hypothesis,
                "| classified_fraction=",
                round(row.classification_fraction, 4),
                "| pressure=",
                row.evidence_pressure_state,
                "| strong=",
                row.strong_pressure_kinds,
                "| weak=",
                row.weak_pressure_kinds,
            )

    print()
    print("Role precedence is lexicographic, not numeric weight: true")
    print("Centrality is relative topology, not novelty score: true")
    print("Evidence pressure bounded to classified subset: true")
    print("Unclassified is negative evidence: false")
    print("Absence-based novelty authorized: false")
    print("Positive non-obviousness authority created: false")
    print("Novelty verdict created: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
