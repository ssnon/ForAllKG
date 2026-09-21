from __future__ import annotations

from pipeline_core.corpus.semantic_ir.capability import (
    CapabilityManifest,
    CapabilityRecord,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)
from pipeline_core.corpus.semantic_ir.task_capability import (
    TaskCapabilityMetric,
    TaskCapabilitySnapshot,
)
from pipeline_core.corpus.semantic_ir.task_gap import (
    TaskLocalCapabilityGapReport,
    TaskLocalCapabilityGapWitness,
    TaskLocalCapabilityMetric,
)
from pipeline_core.discovery.reframing.operator_contracts import (
    get_reframing_operator_contracts,
)
from pipeline_core.discovery.reframing.operator_readiness import (
    assess_reframing_operator_readiness,
)


def _chunk(chunk_id: str) -> SemanticObjectRef:
    return SemanticObjectRef(
        paper_id="P1",
        chunk_id=chunk_id,
        object_kind="chunk",
        object_id=chunk_id,
        source_path=f"/tmp/{chunk_id}.json",
    )


def _record(
    *,
    chunk: SemanticObjectRef,
    kind: str,
    object_id: str,
    payload: dict,
) -> SemanticIRRecord:
    return SemanticIRRecord(
        ref=SemanticObjectRef(
            paper_id="P1",
            chunk_id=chunk.chunk_id,
            object_kind=kind,
            object_id=object_id,
        ),
        payload=payload,
        source_chunk_ref=chunk,
    )


def _bundle() -> SemanticIRBundle:
    c1 = _chunk("c1")
    c2 = _chunk("c2")
    return SemanticIRBundle(
        bundle_id="bundle:P1",
        paper_id="P1",
        source_chunks=[c1, c2],
        records=[
            _record(
                chunk=c1,
                kind="measurement",
                object_id="m1",
                payload={
                    "id": "m1",
                    "metric_id": "sers_intensity",
                    "metric": "SERS intensity",
                    "subject_id": "s1",
                    "source_expression": "intensity increased",
                    "conditions": [],
                },
            ),
            _record(
                chunk=c2,
                kind="experiment",
                object_id="e1",
                payload={"id": "e1", "conditions": [{"name": "laser", "value_text": "633 nm"}]},
            ),
        ],
    )


def _metric(name: str, state: str, *, mode: str = "task_local_observed_coverage") -> TaskCapabilityMetric:
    if state == "complete":
        applicable, supported, coverage = 1, 1, 1.0
    elif state == "partial":
        applicable, supported, coverage = 2, 1, 0.5
    elif state == "absent":
        applicable, supported, coverage = 1, 0, 0.0
    else:
        applicable, supported, coverage = 0, 0, None
    return TaskCapabilityMetric(
        capability=name,
        state=state,
        assessment_mode=mode,
        applicable_count=applicable,
        supported_count=supported,
        coverage_fraction=coverage,
    )


def _snapshot(*, proxy_state: str = "absent") -> TaskCapabilitySnapshot:
    metrics = {}
    for name in (
        "relation_structure",
        "edge_provenance",
        "source_chunk_recovery",
        "claim_support_linkage",
        "claim_application_target",
        "experiment_representation",
        "measurement_representation",
        "measurement_identity",
        "measurement_provider_linkage",
    ):
        metrics[name] = _metric(name, "complete")
    metrics["measurement_condition_coverage"] = _metric(
        "measurement_condition_coverage", "partial"
    )
    metrics["experiment_condition_coverage"] = _metric(
        "experiment_condition_coverage", "partial"
    )
    metrics["calculation_condition_coverage"] = _metric(
        "calculation_condition_coverage", "complete"
    )
    metrics["proxy_semantics"] = _metric(
        "proxy_semantics",
        proxy_state,
        mode="explicit_gap" if proxy_state == "absent" else "task_local_observed_coverage",
    )
    manifest = CapabilityManifest(
        manifest_id="manifest:test",
        scope_kind="task",
        scope_id="task:test",
        capabilities={
            name: CapabilityRecord(name=name, state=metric.state)
            for name, metric in metrics.items()
        },
    )
    return TaskCapabilitySnapshot(
        scope_id="task:test",
        manifest=manifest,
        metrics=metrics,
        source_chunk_count=2,
        semantic_record_count=2,
    )


def _gaps() -> TaskLocalCapabilityGapReport:
    c1 = _chunk("c1")
    metrics = {
        "measurement_condition_coverage": TaskLocalCapabilityMetric(
            capability="measurement_condition_coverage",
            applicable_record_count=2,
            supported_record_count=1,
            missing_record_count=1,
            source_chunk_count=2,
            witness_chunk_count=1,
            coverage_fraction=0.5,
        ),
        "experiment_condition_coverage": TaskLocalCapabilityMetric(
            capability="experiment_condition_coverage",
            applicable_record_count=2,
            supported_record_count=1,
            missing_record_count=1,
            source_chunk_count=2,
            witness_chunk_count=1,
            coverage_fraction=0.5,
        ),
        "calculation_condition_coverage": TaskLocalCapabilityMetric(
            capability="calculation_condition_coverage",
            applicable_record_count=1,
            supported_record_count=1,
            missing_record_count=0,
            source_chunk_count=2,
            witness_chunk_count=0,
            coverage_fraction=1.0,
        ),
    }
    return TaskLocalCapabilityGapReport(
        source_chunk_count=2,
        metrics=metrics,
        witnesses=[
            TaskLocalCapabilityGapWitness(
                capability="measurement_condition_coverage",
                paper_id="P1",
                chunk_id="c1",
                source_chunk_ref=c1,
                applicable_record_count=1,
                missing_record_count=1,
                missing_semantic_refs=[
                    SemanticObjectRef(
                        paper_id="P1",
                        chunk_id="c1",
                        object_kind="measurement",
                        object_id="m1",
                    )
                ],
                reason="missing measurement condition",
            ),
            TaskLocalCapabilityGapWitness(
                capability="experiment_condition_coverage",
                paper_id="P1",
                chunk_id="c1",
                source_chunk_ref=c1,
                applicable_record_count=1,
                missing_record_count=1,
                missing_semantic_refs=[
                    SemanticObjectRef(
                        paper_id="P1",
                        chunk_id="c1",
                        object_kind="experiment",
                        object_id="e-missing",
                    )
                ],
                reason="missing experiment condition",
            ),
        ],
    )


def _by_id(report):
    return {row.operator_id: row for row in report.assessments}


def test_latent_variable_is_zero_cost_when_grounded_contract_is_satisfied():
    bundle = _bundle()
    report = assess_reframing_operator_readiness(
        scope_id="task:test",
        task_capabilities=_snapshot(),
        gap_report=_gaps(),
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
    )
    row = _by_id(report)["LATENT_VARIABLE"]
    assert row.status == "ready_now"
    assert row.mandatory_backfill_plan.target_count == 0
    assert row.optional_backfill_plan.target_count == 0


def test_regime_boundary_uses_partial_conditions_and_only_optionally_backfills_witness_chunks():
    bundle = _bundle()
    report = assess_reframing_operator_readiness(
        scope_id="task:test",
        task_capabilities=_snapshot(),
        gap_report=_gaps(),
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
    )
    row = _by_id(report)["REGIME_BOUNDARY"]
    assert row.status == "ready_with_optional_enrichment"
    assert row.mandatory_backfill_plan.target_count == 0
    assert row.optional_backfill_plan.target_count == 1
    assert row.optional_backfill_plan.targets[0].source_chunk_ref.chunk_id == "c1"


def test_proxy_challenge_requires_only_measurement_bearing_chunk_enrichment():
    bundle = _bundle()
    report = assess_reframing_operator_readiness(
        scope_id="task:test",
        task_capabilities=_snapshot(),
        gap_report=_gaps(),
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
    )
    row = _by_id(report)["PROXY_CHALLENGE"]
    assert row.status == "targeted_enrichment_required"
    assert row.measurement_bearing_chunk_count == 1
    assert row.mandatory_backfill_plan.target_count == 1
    assert row.mandatory_backfill_plan.targets[0].source_chunk_ref.chunk_id == "c1"
    assert row.mandatory_backfill_plan.targets[0].missing_capabilities == [
        "proxy_semantics"
    ]


def test_proxy_challenge_is_ready_if_proxy_semantics_are_already_enriched():
    bundle = _bundle()
    report = assess_reframing_operator_readiness(
        scope_id="task:test",
        task_capabilities=_snapshot(proxy_state="complete"),
        gap_report=_gaps(),
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
    )
    row = _by_id(report)["PROXY_CHALLENGE"]
    assert row.status == "ready_now"
    assert row.mandatory_backfill_plan.target_count == 0


def test_hard_grounding_failure_blocks_operator_without_guessing_backfill_targets():
    bundle = _bundle()
    snapshot = _snapshot()
    snapshot.manifest.capabilities["edge_provenance"] = CapabilityRecord(
        name="edge_provenance",
        state="partial",
    )
    snapshot.metrics["edge_provenance"] = _metric(
        "edge_provenance", "partial"
    )
    report = assess_reframing_operator_readiness(
        scope_id="task:test",
        task_capabilities=snapshot,
        gap_report=_gaps(),
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
    )
    for row in report.assessments:
        assert row.status == "blocked_insufficient_substrate"
        assert row.mandatory_backfill_plan.target_count == 0


def test_operator_readiness_remains_shadow_only_without_authority():
    bundle = _bundle()
    report = assess_reframing_operator_readiness(
        scope_id="task:test",
        task_capabilities=_snapshot(),
        gap_report=_gaps(),
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
    )
    assert report.shadow_only is True
    assert report.llm_calls_performed == 0
    assert report.production_selection_changed is False
    for row in report.assessments:
        assert row.operator_execution_performed is False
        assert row.scientific_trigger_evaluated is False
        assert row.external_novelty_evaluated is False
        assert row.canonical_graph_mutated is False


def test_task_capability_snapshot_is_task_scoped_and_preserves_explicit_gaps():
    from pipeline_core.corpus.semantic_ir.task_capability import (
        build_task_capability_snapshot,
    )

    bundle = _bundle()
    snapshot = build_task_capability_snapshot(
        scope_id="task:test",
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
        task_gap_report=_gaps(),
    )
    assert snapshot.manifest.scope_kind == "task"
    assert snapshot.metrics["proxy_semantics"].state == "absent"
    assert snapshot.metrics["proxy_semantics"].assessment_mode == "explicit_gap"
    assert snapshot.metrics["measurement_condition_coverage"].state == "partial"
    assert snapshot.metrics["source_chunk_recovery"].state == "complete"
    assert snapshot.global_coverage_extrapolated is False


def test_unresolved_grounded_node_is_visible_but_does_not_silently_change_status():
    bundle = _bundle()
    report = assess_reframing_operator_readiness(
        scope_id="task:test",
        task_capabilities=_snapshot(),
        gap_report=_gaps(),
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
        unresolved_grounded_node_count=1,
    )
    assert report.grounded_scope_fully_resolved is False
    assert report.unresolved_grounded_node_count == 1
    latent = _by_id(report)["LATENT_VARIABLE"]
    assert latent.status == "ready_now"
    assert any("unresolved" in reason for reason in latent.reasons)


def test_true_grounded_ambiguity_blocks_all_operator_execution_fail_closed():
    bundle = _bundle()
    report = assess_reframing_operator_readiness(
        scope_id="task:test",
        task_capabilities=_snapshot(),
        gap_report=_gaps(),
        bundles={"P1": bundle},
        task_source_chunks=bundle.source_chunks,
        ambiguous_grounded_node_count=1,
    )
    assert report.grounded_scope_fully_resolved is False
    assert report.ambiguous_grounded_node_count == 1
    assert all(
        row.status == "blocked_insufficient_substrate"
        for row in report.assessments
    )
