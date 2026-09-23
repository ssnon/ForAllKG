from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_factor_projection import (
    build_relational_atomic_identity_factorization_report,
    compile_relational_atomic_grounded_factor_projection_report,
)
from pipeline_core.discovery.relational_atomic_projection import (
    RelationalAtomicProjectionReport,
)
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIRReport,
)
from pipeline_core.discovery.scientific_relation_projection import (
    compile_relation_projection_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve one already-atomic relational branch identity as a "
            "single exact-source grounded factor and compile BASE/FULL "
            "GroundedFactorProjection targets. No synthetic constituent "
            "decomposition or scientific rewriting is allowed."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--relational-projection",
        required=True,
        type=Path,
    )
    parser.add_argument("--relation-ir", required=True, type=Path)
    parser.add_argument(
        "--relation-projection-output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--factorization-output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--grounded-projection-output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--max-alias-query-variants-per-projection",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    plan = RelationalAtomicBindingPlan.model_validate_json(
        args.plan.read_text(encoding="utf-8")
    )
    relational_projection = (
        RelationalAtomicProjectionReport.model_validate_json(
            args.relational_projection.read_text(encoding="utf-8")
        )
    )
    relation_ir = ScientificRelationIRReport.model_validate_json(
        args.relation_ir.read_text(encoding="utf-8")
    )

    relation_projection = compile_relation_projection_report(
        relation_ir,
    )
    factorization = (
        build_relational_atomic_identity_factorization_report(
            plan=plan,
            projection_report=relational_projection,
            relation_ir_report=relation_ir,
        )
    )
    grounded = (
        compile_relational_atomic_grounded_factor_projection_report(
            relation_ir_report=relation_ir,
            factorization_report=factorization,
            max_alias_query_variants_per_projection=(
                args.max_alias_query_variants_per_projection
            ),
        )
    )

    for path, value in (
        (args.relation_projection_output, relation_projection),
        (args.factorization_output, factorization),
        (args.grounded_projection_output, grounded),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            value.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    print("Relational atomic identity/factor projection shadow complete")
    print("Relations:", relation_ir.relation_count)
    print("0066 projections:", relation_projection.projection_count)
    print(
        "Atomic identity ready/not ready:",
        factorization.ready_count,
        "/",
        factorization.not_ready_count,
    )
    print(
        "0068 planning-ready relations:",
        grounded.planning_ready_relation_count,
    )
    print("0068 projections:", grounded.projection_count)
    print(
        "0068 future typed-retrieval eligible:",
        grounded.future_retrieval_eligible_projection_count,
    )

    for row in factorization.relations:
        print()
        print("Claim:", row.claim_id)
        print("Identity status:", row.factorization_status)
        print("Identity:", row.source_identity_term)
        print("Basis tokens:", row.identity_basis_tokens)
        print("Exact aliases:", row.exact_source_aliases)
        for span in row.source_spans:
            print(
                "  source:",
                span.exact_source_text,
                "| paths=",
                span.matched_source_paths,
            )
        if row.reason_codes:
            print("Reasons:", row.reason_codes)

        projection_set = next(
            (
                item
                for item in grounded.projection_sets
                if item.claim_id == row.claim_id
            ),
            None,
        )
        if projection_set is not None:
            for projection in projection_set.projections:
                print(
                    " ",
                    projection.projection_kind,
                    "| factor_order=",
                    projection.factor_order,
                    "| query=",
                    projection.canonical_search_query,
                )

    print()
    print("Atomic identity split into synthetic constituents: false")
    print("Source-exact whole identity required: true")
    print("Scientific content added: false")
    print("Novelty assessment performed: false")
    print("Retrieval performed: false")
    print("Production selection changed: false")
    print("0066:", args.relation_projection_output)
    print("Identity factorization:", args.factorization_output)
    print("0068:", args.grounded_projection_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
