from __future__ import annotations

from pipeline_core.domain.domain_profile import ScientificDomainProfile
from pipeline_core.discovery.atomic_scientific_specification_bundle import (
    AtomicScientificSpecificationBundle,
)
from pipeline_core.discovery.scientific_relation_ir import (
    RelationTypingAdapter,
    ScientificRelationIRReport,
    compile_atomic_specifications_relation_ir_report,
)


def compile_atomic_specification_bundle_relation_ir(
    *,
    bundle: AtomicScientificSpecificationBundle,
    domain_profile: ScientificDomainProfile,
    typing_adapter: RelationTypingAdapter | None = None,
) -> ScientificRelationIRReport:
    """Compile the neutral canonical atomic bundle into ScientificRelationIR.

    The bundle is the scientific-identity container.  Provenance emitted into
    the relation-IR report intentionally preserves the originating synthesis
    report ID/contract so this adapter is behaviorally equivalent to the
    legacy cross-lane report adapter for the same atomic specifications.
    """

    specifications = [
        (hypothesis.hypothesis_id, specification)
        for hypothesis in bundle.hypotheses
        for specification in hypothesis.specifications
    ]
    return compile_atomic_specifications_relation_ir_report(
        source_atomic_report_id=bundle.source_report_id,
        source_contract=bundle.source_contract,
        specifications=specifications,
        domain_profile=domain_profile,
        typing_adapter=typing_adapter,
    )


__all__ = [
    "compile_atomic_specification_bundle_relation_ir",
]
