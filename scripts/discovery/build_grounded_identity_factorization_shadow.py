from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.grounded_identity_factorization import (
    build_grounded_identity_factorization_report,
)
from pipeline_core.discovery.reframing.grounded_identity_constituent_shadow import (
    GroundedIdentityAnnotationReport,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIRReport,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote only exact-source grounded, lexically distinct identity "
            "constituent groups into authority-neutral projection factors. "
            "No relation IR, retrieval, N9, N10, or production selection is "
            "changed."
        )
    )
    parser.add_argument("--relation-ir", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    relation_ir = ScientificRelationIRReport.model_validate_json(
        args.relation_ir.read_text(encoding="utf-8")
    )
    annotation = GroundedIdentityAnnotationReport.model_validate_json(
        args.annotation.read_text(encoding="utf-8")
    )

    report = build_grounded_identity_factorization_report(
        relation_ir_report=relation_ir,
        annotation_report=annotation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Grounded identity factorization shadow complete")
    print("Relations:", report.relation_count)
    print("READY:", report.ready_count)
    print("Not ready:", report.not_ready_count)
    print(
        "Grounded factors:",
        report.total_grounded_factor_count,
    )

    for row in report.relations:
        print()
        print("Claim:", row.claim_id)
        print("Status:", row.factorization_status)
        print(
            "Source identity:",
            row.source_identity_term,
        )
        print(
            "Source identity tokens:",
            row.source_identity_tokens,
        )
        print(
            "Effective identity factors:",
            row.effective_identity_factor_count,
        )

        for factor in row.grounded_factors:
            print(
                "  factor:",
                factor.group_label,
                "| basis=",
                factor.identity_basis_tokens,
                "| exclusive=",
                factor.exclusive_identity_basis_tokens,
            )
            print(
                "    exact aliases:",
                factor.query_aliases,
            )
            print(
                "    source refs:",
                [
                    span.candidate_ref
                    + ":"
                    + ",".join(span.matched_source_paths)
                    for span in factor.grounded_spans
                ],
            )

        if row.preserved_identity_concept_ids:
            print(
                "  preserved atomic identity concepts:",
                row.preserved_identity_concept_ids,
            )
        if row.reason_codes:
            print("  reasons:", row.reason_codes)

    print()
    print("Diagnostic only: true")
    print("Relation IR mutated: false")
    print("Query plan changed: false")
    print("Retrieval changed: false")
    print("Prior-art review changed: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
