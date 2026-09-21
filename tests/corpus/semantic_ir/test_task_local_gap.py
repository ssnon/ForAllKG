from __future__ import annotations

from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)
from pipeline_core.corpus.semantic_ir.task_gap import (
    detect_task_local_capability_gaps,
)


def _chunk(chunk_id: str) -> SemanticObjectRef:
    return SemanticObjectRef(
        paper_id="P",
        chunk_id=chunk_id,
        object_kind="chunk",
        object_id=chunk_id,
        source_path=f"source_chunks/{chunk_id}.json",
    )


def _record(kind: str, object_id: str, chunk: SemanticObjectRef, conditions):
    return SemanticIRRecord(
        ref=SemanticObjectRef(
            paper_id="P",
            chunk_id=chunk.chunk_id,
            object_kind=kind,
            object_id=object_id,
        ),
        payload={"id": object_id, "conditions": conditions},
        source_chunk_ref=chunk,
    )


def test_gap_detection_is_limited_to_selected_task_chunks():
    c1 = _chunk("c1")
    c2 = _chunk("c2")
    bundle = SemanticIRBundle(
        bundle_id="bundle:P",
        paper_id="P",
        source_chunks=[c1, c2],
        records=[
            _record("measurement", "m1", c1, []),
            _record("measurement", "m2", c2, []),
        ],
    )

    report = detect_task_local_capability_gaps(
        bundles={"P": bundle},
        task_source_chunks=[c1],
        capabilities=["measurement_condition_coverage"],
    )

    metric = report.metrics["measurement_condition_coverage"]
    assert metric.applicable_record_count == 1
    assert metric.missing_record_count == 1
    assert metric.witness_chunk_count == 1
    assert [row.chunk_id for row in report.witnesses] == ["c1"]


def test_gap_detection_reports_supported_and_missing_records_separately():
    c1 = _chunk("c1")
    bundle = SemanticIRBundle(
        bundle_id="bundle:P",
        paper_id="P",
        source_chunks=[c1],
        records=[
            _record("experiment", "e1", c1, [{"name": "laser"}]),
            _record("experiment", "e2", c1, []),
        ],
    )

    report = detect_task_local_capability_gaps(
        bundles={"P": bundle},
        task_source_chunks=[c1],
        capabilities=["experiment_condition_coverage"],
    )

    metric = report.metrics["experiment_condition_coverage"]
    assert metric.applicable_record_count == 2
    assert metric.supported_record_count == 1
    assert metric.missing_record_count == 1
    assert metric.coverage_fraction == 0.5
    assert len(report.witnesses) == 1


def test_no_applicable_records_does_not_invent_a_gap():
    c1 = _chunk("c1")
    bundle = SemanticIRBundle(
        bundle_id="bundle:P",
        paper_id="P",
        source_chunks=[c1],
        records=[],
    )

    report = detect_task_local_capability_gaps(
        bundles={"P": bundle},
        task_source_chunks=[c1],
        capabilities=["calculation_condition_coverage"],
    )

    metric = report.metrics["calculation_condition_coverage"]
    assert metric.applicable_record_count == 0
    assert metric.coverage_fraction is None
    assert report.witnesses == []
