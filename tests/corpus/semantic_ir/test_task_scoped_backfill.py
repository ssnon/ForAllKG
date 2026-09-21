from __future__ import annotations

from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRecord,
    CapabilityRequirement,
)
from pipeline_core.corpus.semantic_ir.capability_audit import (
    CapabilityAuditMetric,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)
from pipeline_core.corpus.semantic_ir.task_backfill import (
    plan_task_scoped_backfill,
    source_chunks_for_semantic_refs,
)


def _chunk(chunk_id: str) -> SemanticObjectRef:
    return SemanticObjectRef(
        paper_id="P",
        chunk_id=chunk_id,
        object_kind="chunk",
        object_id=chunk_id,
        source_path=f"source_chunks/{chunk_id}.json",
    )


def _metric(
    capability: str,
    *,
    state: str,
    mode: str,
    applicable: int,
    supported: int,
) -> CapabilityAuditMetric:
    return CapabilityAuditMetric(
        capability=capability,
        state=state,
        assessment_mode=mode,
        applicable_count=applicable,
        supported_count=supported,
        coverage_fraction=(supported / applicable if applicable else None),
    )


def _manifest() -> CapabilityManifest:
    return CapabilityManifest(
        manifest_id="manifest:P",
        scope_kind="paper",
        scope_id="P",
        capabilities={
            "proxy_semantics": CapabilityRecord(
                name="proxy_semantics",
                state="absent",
            ),
            "measurement_condition_coverage": CapabilityRecord(
                name="measurement_condition_coverage",
                state="partial",
            ),
        },
    )


def test_explicit_gap_targets_only_supplied_task_source_chunks():
    chunks = [_chunk("c1"), _chunk("c2")]
    result = plan_task_scoped_backfill(
        scope_id="task:K01",
        operator_id="proxy_challenge",
        manifest=_manifest(),
        metrics={
            "proxy_semantics": _metric(
                "proxy_semantics",
                state="absent",
                mode="explicit_gap",
                applicable=1,
                supported=0,
            ),
        },
        requirements=[
            CapabilityRequirement(
                capability="proxy_semantics",
                minimum_state="complete",
                reason="Proxy challenge needs explicit proxy semantics.",
            )
        ],
        task_source_chunks=chunks,
    )

    assert result.plan.target_count == 2
    assert {
        target.source_chunk_ref.object_id
        for target in result.plan.targets
    } == {"c1", "c2"}
    assert result.deferred_capabilities == []
    assert result.llm_calls_performed == 0
    assert result.canonical_graph_mutated is False


def test_global_partial_coverage_does_not_auto_backfill_task_chunks():
    result = plan_task_scoped_backfill(
        scope_id="task:K01",
        operator_id="regime_boundary",
        manifest=_manifest(),
        metrics={
            "measurement_condition_coverage": _metric(
                "measurement_condition_coverage",
                state="partial",
                mode="observed_coverage",
                applicable=46,
                supported=21,
            ),
        },
        requirements=[
            CapabilityRequirement(
                capability="measurement_condition_coverage",
                minimum_state="complete",
                reason="Regime reasoning may need condition context.",
            )
        ],
        task_source_chunks=[_chunk("c1"), _chunk("c2")],
    )

    assert result.plan.target_count == 0
    assert len(result.deferred_capabilities) == 1
    assert result.deferred_capabilities[0].capability == (
        "measurement_condition_coverage"
    )
    assert result.deferred_capabilities[0].task_local_gap_witness_required
    assert result.global_partial_coverage_auto_backfilled is False


def test_task_semantic_refs_map_back_to_source_chunks_without_text_loading():
    source = _chunk("c1")
    entity_ref = SemanticObjectRef(
        paper_id="P",
        chunk_id="c1",
        object_kind="entity",
        object_id="entity:1",
    )
    measurement_ref = SemanticObjectRef(
        paper_id="P",
        chunk_id="c1",
        object_kind="measurement",
        object_id="measurement:1",
    )
    bundle = SemanticIRBundle(
        bundle_id="bundle:P",
        paper_id="P",
        source_chunks=[source],
        records=[
            SemanticIRRecord(
                ref=entity_ref,
                payload={"id": "entity:1"},
                source_chunk_ref=source,
            ),
            SemanticIRRecord(
                ref=measurement_ref,
                payload={"id": "measurement:1"},
                source_chunk_ref=source,
            ),
        ],
    )

    selected = source_chunks_for_semantic_refs(
        bundle=bundle,
        refs=[entity_ref, measurement_ref],
    )

    assert selected == [source]


def test_observed_partial_gap_with_task_local_witness_targets_only_witness_chunk():
    from pipeline_core.corpus.semantic_ir.task_gap import (
        TaskLocalCapabilityGapWitness,
    )

    c1 = _chunk("c1")
    c2 = _chunk("c2")
    witness = TaskLocalCapabilityGapWitness(
        capability="measurement_condition_coverage",
        paper_id="P",
        chunk_id="c1",
        source_chunk_ref=c1,
        applicable_record_count=2,
        missing_record_count=1,
        missing_semantic_refs=[
            SemanticObjectRef(
                paper_id="P",
                chunk_id="c1",
                object_kind="measurement",
                object_id="m1",
            )
        ],
        reason="task-local measurement lacks structured conditions",
    )

    result = plan_task_scoped_backfill(
        scope_id="task:K01",
        operator_id="regime_boundary",
        manifest=_manifest(),
        metrics={
            "measurement_condition_coverage": _metric(
                "measurement_condition_coverage",
                state="partial",
                mode="observed_coverage",
                applicable=46,
                supported=21,
            ),
        },
        requirements=[
            CapabilityRequirement(
                capability="measurement_condition_coverage",
                minimum_state="complete",
                reason="Regime reasoning needs task-local condition context.",
            )
        ],
        task_source_chunks=[c1, c2],
        task_local_gap_witnesses=[witness],
    )

    assert result.plan.target_count == 1
    assert result.plan.targets[0].source_chunk_ref.object_id == "c1"
    assert result.plan.targets[0].missing_capabilities == [
        "measurement_condition_coverage"
    ]
    assert result.deferred_capabilities == []
