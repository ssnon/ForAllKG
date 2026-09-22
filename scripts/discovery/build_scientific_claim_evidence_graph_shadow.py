from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
)
from pipeline_core.discovery.projection_relation_adjudication import (
    ProjectionRelationAdjudicationReport,
    ProjectionRelationCandidateReport,
)
from pipeline_core.discovery.scientific_claim_evidence_graph import (
    build_scientific_claim_evidence_graph,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIRReport,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic shadow Claim Evidence Graph from typed "
            "relations, grounded projections, bounded review candidates, and "
            "compiled relation adjudication. This does not infer novelty."
        )
    )
    parser.add_argument("--relation-ir", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--adjudication", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    relation_ir = ScientificRelationIRReport.model_validate_json(
        args.relation_ir.read_text(encoding="utf-8")
    )
    projection = GroundedFactorProjectionReport.model_validate_json(
        args.projection.read_text(encoding="utf-8")
    )
    candidates = ProjectionRelationCandidateReport.model_validate_json(
        args.candidates.read_text(encoding="utf-8")
    )
    adjudication = ProjectionRelationAdjudicationReport.model_validate_json(
        args.adjudication.read_text(encoding="utf-8")
    )

    report = build_scientific_claim_evidence_graph(
        relation_ir_report=relation_ir,
        projection_report=projection,
        candidate_report=candidates,
        adjudication_report=adjudication,
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

    print("Scientific Claim Evidence Graph shadow complete")
    print("Graph:", report.graph_id)
    print("Hypotheses:", report.hypothesis_count)
    print("Claims:", report.claim_count)
    print("Nodes:", report.node_count)
    print("Edges:", report.edge_count)
    print("Concept nodes:", report.concept_node_count)
    print("Projection nodes:", report.projection_node_count)
    print("Work nodes:", report.work_node_count)
    print(
        "Presented work edges:",
        report.presented_work_edge_count,
    )
    print(
        "Adjudicated relation edges:",
        report.adjudicated_relation_edge_count,
    )
    print(
        "Unclassified presentation edges:",
        report.unclassified_presentation_edge_count,
    )

    for row in report.claim_summaries:
        print()
        print("Claim:", row.claim_id)
        print("Hypothesis:", row.hypothesis_id)
        print("Role:", row.novelty_selection_role)
        print("Typing:", row.typing_status)
        print("Adjudication:", row.adjudication_status)
        print("Relation state:", row.relation_state)
        print(
            "Coverage:",
            f"presented={row.presented_work_count}",
            f"classified={row.classified_work_count}",
            f"unclassified={row.unclassified_work_count}",
        )
        print(
            "Relationship counts:",
            row.relationship_counts,
        )
        print(
            "Concept/projection nodes:",
            len(row.concept_node_ids),
            "/",
            len(row.projection_node_ids),
        )

    print()
    print("Exact-normalized concept coalescing only: true")
    print("Unclassified is negative evidence: false")
    print("Absence-based novelty authorized: false")
    print("Positive non-obviousness authority created: false")
    print("Centrality scoring performed: false")
    print("Aggregation performed: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
