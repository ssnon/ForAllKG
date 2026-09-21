from __future__ import annotations

from pipeline_core.corpus.semantic_ir.capability_audit import (
    audit_semantic_ir_capabilities,
)
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)


def _ref(kind: str, object_id: str, chunk: str = "P:c0"):
    return SemanticObjectRef(
        paper_id="P",
        chunk_id=chunk,
        object_kind=kind,
        object_id=object_id,
        source_path=f"chunks/{chunk}.json",
    )


def _source(chunk: str = "P:c0"):
    return SemanticObjectRef(
        paper_id="P",
        chunk_id=chunk,
        object_kind="chunk",
        object_id=chunk,
        source_path=f"source_chunks/{chunk}.json",
    )


def _record(kind: str, object_id: str, payload: dict, chunk: str = "P:c0"):
    return SemanticIRRecord(
        ref=_ref(kind, object_id, chunk),
        payload=payload,
        source_chunk_ref=_source(chunk),
    )


def _edge(source: str, relation: str, target: str, chunk: str = "P:c0"):
    return _record(
        "edge",
        f"edge:{source}:{relation}:{target}",
        {
            "source": source,
            "relation": relation,
            "target": target,
            "evidence_pointers": [{
                "document_id": "main",
                "document_role": "main",
                "page_id": 1,
                "asset_ids": [],
                "locator_text": "Results",
            }],
        },
        chunk,
    )


def _bundle(records: list[SemanticIRRecord]):
    return SemanticIRBundle(
        bundle_id="bundle:P",
        paper_id="P",
        records=records,
        source_chunks=[_source()],
    )


def test_audit_distinguishes_structural_support_from_observed_coverage():
    records = [
        _record(
            "experiment",
            "exp:1",
            {
                "id": "exp:1",
                "conditions": [{"name": "pH", "value_text": "7"}],
            },
        ),
        _record(
            "measurement",
            "m:1",
            {
                "id": "m:1",
                "metric_id": "sers_ef",
                "metric": "SERS enhancement factor",
                "subject_id": "sample:1",
                "source_expression": "EF = 10^6",
                "conditions": [{"name": "analyte", "value_text": "R6G"}],
            },
        ),
        _record(
            "measurement",
            "m:2",
            {
                "id": "m:2",
                "metric_id": "rsd",
                "metric": "relative standard deviation",
                "subject_id": "sample:1",
                "source_expression": "RSD = 8%",
                "conditions": [],
            },
        ),
        _edge("exp:1", "HAS_MEASUREMENT", "m:1"),
        _edge("exp:1", "HAS_MEASUREMENT", "m:2"),
    ]

    result = audit_semantic_ir_capabilities(_bundle(records))

    assert result.manifest.capabilities[
        "measurement_representation"
    ].state == "complete"
    assert result.manifest.capabilities["measurement_identity"].state == (
        "complete"
    )
    assert result.manifest.capabilities[
        "measurement_provider_linkage"
    ].state == "complete"
    metric = result.metrics["measurement_condition_coverage"]
    assert metric.state == "partial"
    assert metric.applicable_count == 2
    assert metric.supported_count == 1
    assert metric.coverage_fraction == 0.5


def test_no_applicable_measurements_is_unknown_not_absent():
    result = audit_semantic_ir_capabilities(
        _bundle([
            _record("entity", "sample:1", {"id": "sample:1"}),
        ])
    )

    assert result.manifest.capabilities["measurement_identity"].state == (
        "unknown"
    )
    assert result.manifest.capabilities[
        "measurement_provider_linkage"
    ].state == "unknown"
    assert result.manifest.capabilities[
        "measurement_condition_coverage"
    ].state == "unknown"


def test_missing_one_measurement_provider_is_partial():
    records = [
        _record("experiment", "exp:1", {"id": "exp:1", "conditions": []}),
        _record(
            "measurement",
            "m:1",
            {
                "id": "m:1",
                "metric_id": "a",
                "metric": "A",
                "subject_id": "sample:1",
                "source_expression": "A=1",
                "conditions": [],
            },
        ),
        _record(
            "measurement",
            "m:2",
            {
                "id": "m:2",
                "metric_id": "b",
                "metric": "B",
                "subject_id": "sample:1",
                "source_expression": "B=2",
                "conditions": [],
            },
        ),
        _edge("exp:1", "HAS_MEASUREMENT", "m:1"),
    ]

    result = audit_semantic_ir_capabilities(_bundle(records))
    metric = result.metrics["measurement_provider_linkage"]

    assert metric.state == "partial"
    assert metric.supported_count == 1
    assert metric.applicable_count == 2


def test_explicit_future_semantic_gaps_do_not_become_negative_evidence():
    result = audit_semantic_ir_capabilities(_bundle([]))

    for name in ("proxy_semantics", "regime_semantics", "author_rationale"):
        metric = result.metrics[name]
        assert metric.state == "absent"
        assert metric.assessment_mode == "explicit_gap"
        assert metric.negative_evidence_inferred is False
        assert result.manifest.capabilities[name].state == "absent"

    assert result.negative_evidence_inferred is False
    assert result.manifest.negative_evidence_authority is False


def test_claim_support_and_target_linkage_are_audited_separately():
    records = [
        _record("measurement", "m:1", {
            "id": "m:1",
            "metric_id": "sers_ef",
            "metric": "SERS EF",
            "subject_id": "sample:1",
            "source_expression": "EF = 10^6",
            "conditions": [],
        }),
        _record("observation_claim", "obs:1", {"id": "obs:1"}),
        _edge("m:1", "SUPPORTS_CLAIM", "obs:1"),
    ]

    result = audit_semantic_ir_capabilities(_bundle(records))

    assert result.metrics["claim_support_linkage"].state == "complete"
    assert result.metrics["claim_application_target"].state == "absent"


def test_source_chunk_recovery_detects_unlinked_ir_record():
    linked = _record("entity", "sample:1", {"id": "sample:1"})
    unlinked = SemanticIRRecord(
        ref=_ref("entity", "sample:2"),
        payload={"id": "sample:2"},
        source_chunk_ref=None,
    )
    result = audit_semantic_ir_capabilities(_bundle([linked, unlinked]))

    metric = result.metrics["source_chunk_recovery"]
    assert metric.state == "partial"
    assert metric.supported_count == 1
    assert metric.applicable_count == 2


def test_manifest_id_is_deterministic_for_same_bundle_and_content():
    records = [
        _record("entity", "sample:1", {"id": "sample:1"}),
        _edge("sample:1", "COMPARED_WITH", "sample:2"),
    ]

    first = audit_semantic_ir_capabilities(_bundle(records))
    second = audit_semantic_ir_capabilities(_bundle(records))

    assert first.manifest.manifest_id == second.manifest.manifest_id
    assert first.llm_calls_performed == 0
    assert first.source_text_loaded is False
    assert first.canonical_graph_mutated is False
