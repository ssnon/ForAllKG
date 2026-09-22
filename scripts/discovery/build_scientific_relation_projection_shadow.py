from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIRReport,
)
from pipeline_core.discovery.scientific_relation_projection import (
    compile_relation_projection_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile authority-neutral BASE / bounded lower-order identity "
            "subset / FULL projections from ScientificRelationIR. This shadow "
            "does not modify query planning, retrieval, N9, or N10."
        )
    )
    parser.add_argument("--relation-ir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-lower-order-identity-subset-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-projections-per-relation",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    relation_ir = ScientificRelationIRReport.model_validate_json(
        args.relation_ir.read_text(encoding="utf-8")
    )
    report = compile_relation_projection_report(
        relation_ir,
        max_lower_order_identity_subset_size=(
            args.max_lower_order_identity_subset_size
        ),
        max_projections_per_relation=args.max_projections_per_relation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Scientific relation projection shadow complete")
    print("Relations:", report.relation_count)
    print("Projections:", report.projection_count)
    print(
        "Future typed-retrieval eligible:",
        report.future_retrieval_eligible_projection_count,
    )
    print(
        "Truncated relations:",
        report.truncated_relation_count,
    )

    for row in report.projection_sets:
        print()
        print("Claim:", row.claim_id)
        print("Identity factors:", row.identity_concept_count)
        print("Projection count:", row.projection_count)
        if row.reason_codes:
            print("Set reasons:", row.reason_codes)

        for projection in row.projections:
            print(
                " ",
                projection.projection_kind,
                "| order=",
                projection.identity_order,
                "| retained=",
                projection.retained_identity_terms,
            )
            print(
                "   query:",
                projection.search_query,
            )
            print(
                "   future retrieval eligible:",
                projection.eligible_for_future_typed_retrieval,
            )

    print()
    print("Diagnostic only: true")
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
