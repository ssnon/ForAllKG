from __future__ import annotations

from pipeline_core.domain.domain_profile import ScientificDomainProfile
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)
from pipeline_core.discovery.scientific_relation_ir import (
    RelationTypingAdapter,
    ScientificRelationIRReport,
    compile_atomic_specifications_relation_ir_report,
)


def compile_atomic_report_relation_ir(
    *,
    report: AtomicCrossLaneSynthesisReport,
    domain_profile: ScientificDomainProfile,
    typing_adapter: RelationTypingAdapter | None = None,
) -> ScientificRelationIRReport:
    """Compatibility-only adapter for historical cross-lane synthesis reports.

    Fresh runtime scientific identity should be supplied through
    AtomicScientificSpecificationBundle. This adapter remains deterministic
    and authority-neutral for historical/debug compatibility.
    """

    specifications = [
        (hypothesis.hypothesis_id, spec)
        for hypothesis in report.hypotheses
        for spec in hypothesis.atomic_specifications
    ]
    return compile_atomic_specifications_relation_ir_report(
        source_atomic_report_id=report.report_id,
        source_contract="atomic-cross-lane-scientific-synthesis-report-v1",
        specifications=specifications,
        domain_profile=domain_profile,
        typing_adapter=typing_adapter,
    )


__all__ = ["compile_atomic_report_relation_ir"]
