from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.scientific_claim_centrality import (
    build_scientific_claim_centrality_report,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    ScientificClaimEvidenceGraphReport,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute deterministic structural claim centrality and bounded "
            "evidence-pressure diagnostics from a Scientific Claim Evidence "
            "Graph. This does not aggregate novelty or infer non-obviousness."
        )
    )
    parser.add_argument(
        "--evidence-graph",
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
    report = build_scientific_claim_centrality_report(
        graph
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

    print("Scientific claim centrality shadow complete")
    print("Report:", report.report_id)
    print("Hypotheses:", report.hypothesis_count)
    print("Claims:", report.claim_count)
    print(
        "Single-claim hypotheses:",
        report.single_claim_hypothesis_count,
    )
    print(
        "Connected multi-claim hypotheses:",
        report.multi_claim_connected_hypothesis_count,
    )
    print(
        "Multi-claim hypotheses without exact shared concepts:",
        report.multi_claim_no_shared_concept_hypothesis_count,
    )

    for hypothesis in report.hypothesis_summaries:
        print()
        print("Hypothesis:", hypothesis.hypothesis_id)
        print("Topology:", hypothesis.topology_state)
        print(
            "Claims:",
            hypothesis.claim_count,
            "| shared claim pairs:",
            hypothesis.exact_shared_claim_pair_count,
            "/",
            hypothesis.total_claim_pair_count,
        )
        print(
            "Novelty-bearing claims:",
            hypothesis.novelty_bearing_claim_ids,
        )
        print(
            "Role order:",
            hypothesis.role_ordered_claim_ids,
        )
        print(
            "Centrality order:",
            hypothesis.centrality_ordered_claim_ids,
        )
        print(
            "Coverage states:",
            hypothesis.classification_coverage_states,
        )
        print(
            "Evidence pressure:",
            hypothesis.evidence_pressure_states,
        )
        print(
            "Strong-pressure claims:",
            hypothesis.strong_pressure_claim_ids,
        )
        print(
            "Aggregation readiness:",
            hypothesis.aggregation_readiness,
        )

        records = [
            row
            for row in report.claim_records
            if row.hypothesis_id == hypothesis.hypothesis_id
        ]
        for row in records:
            print(
                " ",
                row.claim_id,
                "| role=",
                row.novelty_selection_role,
                "| centrality=",
                round(row.structural_centrality_score, 4),
                "| rank=",
                row.centrality_rank_within_hypothesis,
                "| degree=",
                round(row.neighbor_degree_centrality, 4),
                "| shared_fraction=",
                round(row.shared_concept_fraction, 4),
                "| classified_fraction=",
                round(row.classification_fraction, 4),
                "| pressure=",
                row.evidence_pressure_state,
            )

    print()
    print("Role semantics kept separate from centrality: true")
    print("Evidence pressure kept separate from centrality: true")
    print("Synonym inference performed: false")
    print("Unclassified is negative evidence: false")
    print("Absence-based novelty authorized: false")
    print("Positive non-obviousness authority created: false")
    print("Hypothesis novelty aggregation performed: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
