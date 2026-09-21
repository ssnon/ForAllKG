from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.corpus.semantic_ir.annotation import SemanticAnnotation
from pipeline_core.corpus.semantic_ir.backfill import (
    BackfillTarget,
    build_backfill_plan,
)
from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRecord,
    CapabilityRequirement,
    evaluate_capability_requirements,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)


def _chunk_ref(
    chunk_id: str = "P:main:c0",
) -> SemanticObjectRef:
    return SemanticObjectRef(
        paper_id="P",
        chunk_id=chunk_id,
        object_kind="chunk",
        object_id=chunk_id,
        source_path=f"source_chunks/{chunk_id}.json",
    )


def _measurement_ref() -> SemanticObjectRef:
    return SemanticObjectRef(
        paper_id="P",
        chunk_id="P:main:c0",
        object_kind="measurement",
        object_id="measurement:1",
    )


def test_semantic_ir_wraps_existing_payload_without_new_authority() -> None:
    source = _chunk_ref()
    record = SemanticIRRecord(
        ref=_measurement_ref(),
        payload={"id": "measurement:1", "metric": "SERS EF"},
        source_chunk_ref=source,
    )
    bundle = SemanticIRBundle(
        bundle_id="semantic_ir:P:v1",
        paper_id="P",
        source_run_id="run:1",
        records=[record],
        source_chunks=[source],
    )

    assert bundle.llm_calls_performed == 0
    assert bundle.canonical_graph_mutated is False
    assert record.positive_premise_authority_created is False
    assert record.novelty_authority_created is False


def test_semantic_ir_rejects_cross_paper_source_chunk() -> None:
    foreign = SemanticObjectRef(
        paper_id="OTHER",
        chunk_id="OTHER:c0",
        object_kind="chunk",
        object_id="OTHER:c0",
    )

    with pytest.raises(ValidationError):
        SemanticIRRecord(
            ref=_measurement_ref(),
            payload={"id": "measurement:1"},
            source_chunk_ref=foreign,
        )


def test_annotation_is_append_only_and_cannot_claim_authority() -> None:
    annotation = SemanticAnnotation(
        annotation_id="ann:proxy:1",
        capability="proxy_semantics",
        subject_ref=_measurement_ref(),
        value={"proxy_role": "candidate_proxy"},
        source_refs=[_chunk_ref()],
        extractor_version="proxy-semantic-backfill-v1",
        epistemic_status="source_interpretation",
    )

    assert annotation.append_only is True
    assert annotation.canonical_graph_mutated is False
    assert annotation.positive_premise_authority is False
    assert annotation.novelty_authority is False
    assert annotation.selection_authority is False
    assert annotation.rejection_authority is False

    with pytest.raises(ValidationError):
        SemanticAnnotation(
            annotation_id="ann:invalid",
            capability="proxy_semantics",
            subject_ref=_measurement_ref(),
            value={},
            source_refs=[_chunk_ref()],
            extractor_version="test",
            epistemic_status="source_interpretation",
            novelty_authority=True,
        )


def test_missing_capability_is_backfill_need_not_negative_evidence() -> None:
    manifest = CapabilityManifest(
        manifest_id="cap:P:v1",
        scope_kind="paper",
        scope_id="P",
        capabilities={
            "measurement_identity": CapabilityRecord(
                name="measurement_identity",
                state="complete",
            ),
            "measurement_provider": CapabilityRecord(
                name="measurement_provider",
                state="partial",
            ),
            "proxy_semantics": CapabilityRecord(
                name="proxy_semantics",
                state="absent",
            ),
        },
    )
    result = evaluate_capability_requirements(
        manifest=manifest,
        requirements=[
            CapabilityRequirement(
                capability="measurement_identity",
                minimum_state="complete",
                reason="proxy challenge requires identifiable measurements",
            ),
            CapabilityRequirement(
                capability="measurement_provider",
                minimum_state="partial",
                reason="provider context must be recoverable",
            ),
            CapabilityRequirement(
                capability="proxy_semantics",
                minimum_state="complete",
                reason="proxy role has not been extracted yet",
            ),
            CapabilityRequirement(
                capability="regime_semantics",
                minimum_state="partial",
                reason="undeclared capability should trigger enrichment",
            ),
        ],
    )

    assert result.requirement_count == 4
    assert result.satisfied_count == 2
    assert result.backfill_required_count == 2
    assert [
        row.observed_state
        for row in result.assessments
    ] == [
        "complete",
        "partial",
        "absent",
        "undeclared",
    ]
    assert all(
        row.negative_evidence_inferred is False
        for row in result.assessments
    )
    assert result.negative_evidence_inferred is False


def test_backfill_plan_deduplicates_chunks_without_execution() -> None:
    source = _chunk_ref()
    requirements = [
        CapabilityRequirement(
            capability="proxy_semantics",
            minimum_state="complete",
            reason="needed for proxy challenge",
        ),
        CapabilityRequirement(
            capability="measurement_provider",
            minimum_state="complete",
            reason="needed for operationalization audit",
        ),
    ]

    plan = build_backfill_plan(
        plan_id="backfill:test",
        requirements=requirements,
        targets=[
            BackfillTarget(
                source_chunk_ref=source,
                missing_capabilities=["proxy_semantics"],
                reasons=["proxy role missing"],
            ),
            BackfillTarget(
                source_chunk_ref=source,
                missing_capabilities=["measurement_provider"],
                reasons=["provider identity incomplete"],
            ),
        ],
    )

    assert plan.target_count == 1
    assert plan.targets[0].missing_capabilities == [
        "measurement_provider",
        "proxy_semantics",
    ]
    assert plan.execution_performed is False
    assert plan.llm_calls_performed == 0
    assert plan.canonical_graph_mutated is False
    assert plan.positive_premise_authority_created is False
    assert plan.novelty_authority_created is False
    assert plan.selection_authority_created is False
