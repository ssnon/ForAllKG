from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.atomic_scientific_specification import (
    CompiledAtomicSpecification,
)
from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimScientificStructure,
)
from pipeline_core.discovery.legacy_atomic_admissibility_adapter import (
    adapt_legacy_atomic_specification_shadow_report,
    adapt_legacy_atomic_specification_shadow_row,
)
from pipeline_core.discovery.legacy_atomic_specification_shadow import (
    LegacyAtomicSpecificationShadowReport,
    LegacyAtomicSpecificationShadowRow,
)


def _sha(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _spec() -> CompiledAtomicSpecification:
    return CompiledAtomicSpecification(
        local_id="c1",
        claim_id="claim:1",
        kind="moderator_interaction",
        importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="Factor M moderates descriptor D and outcome O.",
        rationale="test",
        source_candidate_ids=[],
        premise_statement_ids=["statement:1"],
        gap_statement_ids=[],
        prior_art_identity_terms=["Factor M"],
        relation_endpoint_anchors=["descriptor D", "outcome O"],
        scope_qualifier_spans=[],
        directional_qualifier_spans=[],
        relation_nucleus_terms=["descriptor D", "outcome O"],
        distinguishing_terms=["Factor M"],
        required_bridge=(
            "Factor M moderates the relation between descriptor D and outcome O."
        ),
        observable="outcome O",
        predicted_observation="Outcome O changes with descriptor D.",
        falsification_condition="Outcome O does not change with descriptor D.",
        prediction_observation_id="prediction:1",
        falsification_criterion_id="falsifier:1",
        search_concepts=[],
        search_queries=["descriptor D outcome O"],
        scientific_structure=NoveltyClaimScientificStructure(),
        scientific_structure_reason_codes=[],
    )


def _legacy_row() -> LegacyAtomicSpecificationShadowRow:
    return LegacyAtomicSpecificationShadowRow(
        hypothesis_id="hypothesis:h1",
        claim_id="claim:1",
        claim_rank=1,
        claim_local_id="c1",
        kind="moderator_interaction",
        novelty_selection_role="NOVELTY_BEARING",
        source_reference_status="READY",
        proposition_fidelity_status="REVIEW",
        bridge_fidelity_status="INVALID",
        semantic_fidelity_status="INVALID",
        specification_status="COMPLETE",
        atomic_kind_status="SUPPORTED",
        compilation_status="COMPILED_SHADOW",
        source_reference_reason_codes=[],
        proposition_fidelity_reason_codes=[
            "atomic_claim_ordered_language_not_in_source_basis"
        ],
        bridge_fidelity_reason_codes=[
            "required_bridge_relation_endpoint_alignment_unresolved"
        ],
        unclassified_semantic_reason_codes=[],
        specification_reason_codes=[],
        semantic_fidelity_reason_codes=[
            "atomic_claim_ordered_language_not_in_source_basis",
            "required_bridge_relation_endpoint_alignment_unresolved",
        ],
        semantic_fidelity_taxonomy_shadow={"diagnostic_only": True},
        prediction_observation_id="prediction:1",
        falsification_criterion_id="falsifier:1",
        source_observable="outcome O",
        specification=_spec(),
    )


def _legacy_report(
    row: LegacyAtomicSpecificationShadowRow,
) -> LegacyAtomicSpecificationShadowReport:
    body = {
        "schema_version": (
            "legacy-atomic-scientific-specification-shadow-report-v1"
        ),
        "hypothesis_id": "hypothesis:h1",
        "row_count": 1,
        "compiled_count": 1,
        "source_ready_count": 1,
        "semantic_pass_count": 0,
        "semantic_review_count": 0,
        "semantic_invalid_count": 1,
        "specification_complete_count": 1,
        "rows": [row.model_dump(mode="json")],
        "diagnostic_only": True,
        "production_authority": False,
        "canonical_claim_mutated": False,
        "query_plan_mutated": False,
        "vpre_contract_changed": False,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "production_selection_changed": False,
    }
    digest = _sha(body)
    return LegacyAtomicSpecificationShadowReport(
        **body,
        report_id="legacy_atomic_specification_shadow:" + digest[:20],
        report_sha256=digest,
    )


def test_legacy_row_adapter_preserves_admissibility_axes_and_specification():
    source = _legacy_row()
    adapted = adapt_legacy_atomic_specification_shadow_row(source)

    assert adapted.hypothesis_id == source.hypothesis_id
    assert adapted.claim_id == source.claim_id
    assert adapted.claim_rank == source.claim_rank
    assert adapted.claim_local_id == source.claim_local_id
    assert adapted.source_reference_status == source.source_reference_status
    assert adapted.semantic_fidelity_status == source.semantic_fidelity_status
    assert adapted.specification_status == source.specification_status
    assert adapted.atomic_kind_status == source.atomic_kind_status
    assert adapted.compilation_status == source.compilation_status
    assert (
        adapted.semantic_fidelity_reason_codes
        == source.semantic_fidelity_reason_codes
    )
    assert adapted.prediction_observation_id == source.prediction_observation_id
    assert (
        adapted.falsification_criterion_id
        == source.falsification_criterion_id
    )
    assert adapted.specification is not None
    assert (
        adapted.specification.model_dump(mode="json")
        == source.specification.model_dump(mode="json")
    )
    assert adapted.production_authority is False


def test_legacy_report_adapter_preserves_source_provenance_and_counts():
    source = _legacy_report(_legacy_row())
    adapted = adapt_legacy_atomic_specification_shadow_report(source)

    assert adapted.source_representation_id == source.report_id
    assert adapted.source_representation_sha256 == source.report_sha256
    assert adapted.source_representation_schema == source.schema_version
    assert adapted.hypothesis_id == source.hypothesis_id
    assert adapted.row_count == source.row_count
    assert adapted.compiled_count == source.compiled_count
    assert adapted.source_ready_count == source.source_ready_count
    assert adapted.semantic_invalid_count == source.semantic_invalid_count
    assert (
        adapted.specification_complete_count
        == source.specification_complete_count
    )
    assert adapted.diagnostic_only is True
    assert adapted.production_authority is False
    assert adapted.router_authority is False
    assert (
        adapted.exact_text_reconstruction_used_for_source_identity
        is False
    )
