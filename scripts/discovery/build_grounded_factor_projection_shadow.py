from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.grounded_factor_projection import (
    compile_grounded_factor_projection_report,
)
from pipeline_core.discovery.grounded_identity_factorization import (
    GroundedIdentityFactorizationReport,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIRReport,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile BASE, bounded lower-order, and FULL prior-art targets "
            "from exact-source grounded identity factors. This remains a "
            "shadow planner and does not execute retrieval."
        )
    )
    parser.add_argument("--relation-ir", required=True, type=Path)
    parser.add_argument("--factorization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-lower-order-factor-subset-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-projections-per-relation",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-alias-query-variants-per-projection",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    relation_ir = ScientificRelationIRReport.model_validate_json(
        args.relation_ir.read_text(encoding="utf-8")
    )
    factorization = GroundedIdentityFactorizationReport.model_validate_json(
        args.factorization.read_text(encoding="utf-8")
    )

    report = compile_grounded_factor_projection_report(
        relation_ir_report=relation_ir,
        factorization_report=factorization,
        max_lower_order_factor_subset_size=(
            args.max_lower_order_factor_subset_size
        ),
        max_projections_per_relation=(
            args.max_projections_per_relation
        ),
        max_alias_query_variants_per_projection=(
            args.max_alias_query_variants_per_projection
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Grounded factor projection shadow complete")
    print("Relations:", report.relation_count)
    print(
        "Planning-ready relations:",
        report.planning_ready_relation_count,
    )
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
        print("Status:", row.factorization_status)
        print("Grounded factors:", row.grounded_factor_count)
        print("Projection count:", row.projection_count)
        print("Planning ready:", row.planning_ready)
        if row.reason_codes:
            print("Set reasons:", row.reason_codes)

        for projection in row.projections:
            labels = [
                binding.group_label
                for binding in projection.factor_bindings
            ]
            print(
                " ",
                projection.projection_kind,
                "| order=",
                projection.factor_order,
                "| factors=",
                labels,
            )
            print(
                "   canonical:",
                projection.canonical_search_query,
            )
            if (
                projection.exact_source_query_variant_count
                > 1
            ):
                print(
                    "   exact-source variants:",
                    projection.exact_source_query_variants,
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
