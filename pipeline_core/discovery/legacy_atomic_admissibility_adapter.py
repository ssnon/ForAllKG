from __future__ import annotations

from pipeline_core.discovery.atomic_scientific_admissibility import (
    AtomicScientificAdmissibilityAssessment,
    AtomicScientificAdmissibilityAssessmentReport,
    build_atomic_scientific_admissibility_assessment_report,
)
from pipeline_core.discovery.legacy_atomic_specification_shadow import (
    LegacyAtomicSpecificationShadowReport,
    LegacyAtomicSpecificationShadowRow,
)


def adapt_legacy_atomic_specification_shadow_row(
    source: LegacyAtomicSpecificationShadowRow,
) -> AtomicScientificAdmissibilityAssessment:
    return AtomicScientificAdmissibilityAssessment(
        hypothesis_id=source.hypothesis_id,
        claim_id=source.claim_id,
        claim_rank=source.claim_rank,
        claim_local_id=source.claim_local_id,
        kind=source.kind,
        novelty_selection_role=source.novelty_selection_role,
        source_reference_status=source.source_reference_status,
        proposition_fidelity_status=source.proposition_fidelity_status,
        bridge_fidelity_status=source.bridge_fidelity_status,
        semantic_fidelity_status=source.semantic_fidelity_status,
        specification_status=source.specification_status,
        atomic_kind_status=source.atomic_kind_status,
        compilation_status=source.compilation_status,
        source_reference_reason_codes=list(
            source.source_reference_reason_codes
        ),
        proposition_fidelity_reason_codes=list(
            source.proposition_fidelity_reason_codes
        ),
        bridge_fidelity_reason_codes=list(
            source.bridge_fidelity_reason_codes
        ),
        unclassified_semantic_reason_codes=list(
            source.unclassified_semantic_reason_codes
        ),
        specification_reason_codes=list(
            source.specification_reason_codes
        ),
        semantic_fidelity_reason_codes=list(
            source.semantic_fidelity_reason_codes
        ),
        prediction_observation_id=source.prediction_observation_id,
        falsification_criterion_id=source.falsification_criterion_id,
        source_observable=source.source_observable,
        specification=source.specification,
    )


def adapt_legacy_atomic_specification_shadow_report(
    source: LegacyAtomicSpecificationShadowReport,
) -> AtomicScientificAdmissibilityAssessmentReport:
    rows = [
        adapt_legacy_atomic_specification_shadow_row(row)
        for row in source.rows
    ]
    return build_atomic_scientific_admissibility_assessment_report(
        source_representation_id=source.report_id,
        source_representation_sha256=source.report_sha256,
        source_representation_schema=source.schema_version,
        hypothesis_id=source.hypothesis_id,
        rows=rows,
    )


__all__ = [
    "adapt_legacy_atomic_specification_shadow_report",
    "adapt_legacy_atomic_specification_shadow_row",
]
