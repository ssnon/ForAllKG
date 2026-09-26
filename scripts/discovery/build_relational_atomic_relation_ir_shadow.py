from __future__ import annotations

import argparse
from pathlib import Path

from domains.registry import get_domain_profile
from domains.relation_ir_registry import get_relation_typing_adapter
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    AtomicScientificSpecificationBundle,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    RelationalAtomicEndpointBindingReport,
)
from pipeline_core.discovery.relational_atomic_projection import (
    canonical_specifications_from_bundle_for_plan,
    compile_relational_atomic_projection,
    compile_relational_atomic_projection_relation_ir,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project frozen relational N10 claims plus literal endpoint "
            "bindings into the authority-neutral ScientificRelationIR shadow. "
            "Prediction/falsifier observable provenance must resolve exactly "
            "against the original candidate HypothesisCard."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--endpoint-report", required=True, type=Path)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument(
        "--canonical-spec-bundle",
        type=Path,
        default=None,
    )
    parser.add_argument("--projection-output", required=True, type=Path)
    parser.add_argument("--relation-ir-output", required=True, type=Path)
    args = parser.parse_args()

    plan = RelationalAtomicBindingPlan.model_validate_json(
        args.plan.read_text(encoding="utf-8")
    )
    endpoint_report = (
        RelationalAtomicEndpointBindingReport.model_validate_json(
            args.endpoint_report.read_text(encoding="utf-8")
        )
    )

    canonical_specifications = None
    if args.canonical_spec_bundle is not None:
        bundle_path = args.canonical_spec_bundle.expanduser().resolve()
        if not bundle_path.is_file():
            raise ValueError(
                "missing canonical specification bundle: "
                + str(bundle_path)
            )
        bundle = AtomicScientificSpecificationBundle.model_validate_json(
            bundle_path.read_text(encoding="utf-8")
        )
        canonical_specifications = (
            canonical_specifications_from_bundle_for_plan(
                bundle=bundle,
                plan=plan,
            )
        )

    projection = compile_relational_atomic_projection(
        plan=plan,
        endpoint_report=endpoint_report,
        canonical_specifications=canonical_specifications,
    )

    profile = get_domain_profile(args.domain_profile)
    adapter = get_relation_typing_adapter(profile.profile_id)
    relation_ir = compile_relational_atomic_projection_relation_ir(
        report=projection,
        domain_profile=profile,
        typing_adapter=adapter,
    )

    args.projection_output.parent.mkdir(parents=True, exist_ok=True)
    args.projection_output.write_text(
        projection.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.relation_ir_output.parent.mkdir(parents=True, exist_ok=True)
    args.relation_ir_output.write_text(
        relation_ir.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Relational atomic compatibility projection complete")
    print(
        "Source binding mode:",
        (
            "CANONICAL_STABLE_ID"
            if args.canonical_spec_bundle is not None
            else "LEGACY_EXACT_TEXT_FALLBACK"
        ),
    )
    print("Selected claims:", projection.selected_claim_count)
    print("Projected claims:", projection.projected_claim_count)
    print(
        "Endpoint abstentions:",
        projection.skipped_endpoint_abstention_count,
    )
    print(
        "Source-binding abstentions:",
        projection.source_binding_abstention_count,
    )
    for row in projection.rows:
        print(
            " ",
            row.final_hypothesis_id,
            row.claim_id,
            row.projection_status,
            row.reason_codes,
        )

    print()
    print("Scientific relation IR shadow complete")
    print("Source contract:", relation_ir.source_contract)
    print("Domain profile:", relation_ir.domain_profile_id)
    print("Typing adapter:", relation_ir.typing_adapter_id)
    print("Relations:", relation_ir.relation_count)
    print("READY:", relation_ir.ready_count)
    print("PARTIAL:", relation_ir.partial_count)
    print("AMBIGUOUS:", relation_ir.ambiguous_count)
    print("STRUCTURALLY_INVALID:", relation_ir.structurally_invalid_count)
    for relation in relation_ir.relations:
        print()
        print("Claim:", relation.claim_id)
        print("Hypothesis:", relation.hypothesis_id)
        print("Status:", relation.typing_status)
        print(
            "Endpoints:",
            [row.surface_text for row in relation.endpoint_concepts],
        )
        print(
            "Identity:",
            [row.surface_text for row in relation.identity_concepts],
        )
        print("Types:", relation.relation_type_labels)
        if relation.reason_codes:
            print("Reasons:", relation.reason_codes)

    print()
    print("Scientific content added: false")
    print("Novelty assessment performed: false")
    print("Verifier result observed before projection: false")
    print("Production selection changed: false")
    print("Projection:", args.projection_output)
    print("Relation IR:", args.relation_ir_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
