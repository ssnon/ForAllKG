from __future__ import annotations

import argparse
from pathlib import Path

from domains.registry import get_domain_profile
from domains.relation_ir_registry import (
    get_relation_typing_adapter,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)
from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    AtomicScientificSpecificationBundle,
)
from pipeline_core.discovery.atomic_scientific_specification_bundle_relation_ir import (
    compile_atomic_specification_bundle_relation_ir,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_relation_ir_adapter import (
    compile_atomic_report_relation_ir,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile atomic scientific specifications into an authority-neutral "
            "typed ScientificRelationIR shadow. This does not change retrieval, "
            "prior-art adjudication, N9/N10, or production selection."
        )
    )
    parser.add_argument("--atomic-report", required=True, type=Path)
    parser.add_argument(
        "--canonical-spec-bundle",
        type=Path,
        default=None,
    )
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    atomic_report = AtomicCrossLaneSynthesisReport.model_validate_json(
        args.atomic_report.read_text(encoding="utf-8")
    )
    profile = get_domain_profile(args.domain_profile)
    adapter = get_relation_typing_adapter(profile.profile_id)

    if args.canonical_spec_bundle is not None:
        bundle_path = args.canonical_spec_bundle.expanduser().resolve()
        if not bundle_path.is_file():
            raise ValueError(
                "missing canonical atomic specification bundle: "
                + str(bundle_path)
            )
        bundle = AtomicScientificSpecificationBundle.model_validate_json(
            bundle_path.read_text(encoding="utf-8")
        )
        if bundle.source_report_id != atomic_report.report_id:
            raise ValueError(
                "canonical bundle/source atomic report ID mismatch"
            )
        if bundle.source_contract != atomic_report.schema_version:
            raise ValueError(
                "canonical bundle/source atomic report contract mismatch"
            )
        report = compile_atomic_specification_bundle_relation_ir(
            bundle=bundle,
            domain_profile=profile,
            typing_adapter=adapter,
        )
    else:
        report = compile_atomic_report_relation_ir(
            report=atomic_report,
            domain_profile=profile,
            typing_adapter=adapter,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Scientific relation IR shadow complete")
    print(
        "Scientific identity source:",
        (
            "CANONICAL_SPECIFICATION_BUNDLE"
            if args.canonical_spec_bundle is not None
            else "ATOMIC_REPORT_COMPATIBILITY"
        ),
    )
    print("Domain profile:", report.domain_profile_id)
    print("Typing adapter:", report.typing_adapter_id)
    print("Relations:", report.relation_count)
    print("READY:", report.ready_count)
    print("PARTIAL:", report.partial_count)
    print("AMBIGUOUS:", report.ambiguous_count)
    print("STRUCTURALLY_INVALID:", report.structurally_invalid_count)

    for relation in report.relations:
        print()
        print("Claim:", relation.claim_id)
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
        print("Domain labels:", relation.relation_domain_labels)
        print("Scope features:", relation.relation_scope_features)
        if relation.reason_codes:
            print("Reasons:", relation.reason_codes)

    print()
    print("Diagnostic only: true")
    print("Query plan changed: false")
    print("Prior-art review changed: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
