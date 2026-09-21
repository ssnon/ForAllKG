from __future__ import annotations

import json

import pytest

from pipeline_core.corpus.semantic_ir.existing_extraction import (
    load_existing_extraction_semantic_ir,
)


def _edge(source: str, target: str) -> dict:
    return {
        "source": source,
        "relation": "COMPARED_WITH",
        "target": target,
        "evidence_type": "synthesis_procedure",
        "evidence_strength": "direct",
        "evidence_text": "The two materials were compared.",
        "confidence": "high",
        "evidence_pointers": [{
            "document_id": "main",
            "document_role": "main",
            "page_id": None,
            "asset_ids": [],
            "locator_text": "Results",
        }],
        "subsection": "Results",
    }


def _graph_payload() -> dict:
    return {
        "paper_id": "P",
        "chunk_id": "P:main:c0",
        "section": "Results",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [3],
        "asset_ids": [],
        "entities": [
            {
                "id": "material:a",
                "type": "Material",
                "label": "Material A",
                "description": None,
            },
            {
                "id": "material:b",
                "type": "Material",
                "label": "Material B",
                "description": None,
            },
        ],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [_edge("material:a", "material:b")],
    }


def _source_payload() -> dict:
    return {
        "paper_id": "P",
        "chunk_id": "P:main:c0",
        "document_id": "main",
        "document_role": "main",
        "section": "Results",
        "chunk_index": 0,
        "split_depth": 0,
        "page_ids": [3],
        "asset_ids": [],
        "asset_paths": [],
        "asset_pages": [],
        "asset_locators": [],
        "asset_context": "",
        "left_context": "left",
        "core_text": "Material A was compared with Material B.",
        "right_context": "right",
    }


def _write_attempt(tmp_path, *, stale_paths: bool = False):
    attempt = tmp_path / "attempt"
    source_dir = attempt / "source_chunks"
    chunk_dir = attempt / "chunks"
    source_dir.mkdir(parents=True)
    chunk_dir.mkdir(parents=True)

    source_path = source_dir / "P__main__c0.json"
    output_path = chunk_dir / "P__main__c0.json"
    source_path.write_text(json.dumps(_source_payload()), encoding="utf-8")
    output_path.write_text(json.dumps(_graph_payload()), encoding="utf-8")

    recorded_source = (
        "/stale/location/source_chunks/P__main__c0.json"
        if stale_paths
        else str(source_path)
    )
    recorded_output = (
        "/stale/location/chunks/P__main__c0.json"
        if stale_paths
        else str(output_path)
    )
    active = {
        "paper_id": "P",
        "run_id": "run:1",
        "active_chunk_count": 1,
        "chunks": [{
            "paper_id": "P",
            "chunk_id": "P:main:c0",
            "document_id": "main",
            "document_role": "main",
            "section": "Results",
            "status": "success",
            "source_path": recorded_source,
            "output_path": recorded_output,
        }],
    }
    (attempt / "active_chunks.json").write_text(
        json.dumps(active),
        encoding="utf-8",
    )
    return attempt


def test_existing_attempt_builds_ir_without_copying_source_text(tmp_path):
    attempt = _write_attempt(tmp_path)

    result = load_existing_extraction_semantic_ir(attempt)

    assert result.llm_calls_performed == 0
    assert result.source_text_copied_into_ir is False
    assert result.canonical_graph_mutated is False
    assert result.active_chunk_count == 1
    assert result.source_chunk_count == 1
    assert result.chunk_record_count == 1
    assert result.node_record_count == 2
    assert result.edge_record_count == 1
    assert result.semantic_record_count == 4

    bundle = result.bundle
    assert bundle.source_run_id == "run:1"
    assert bundle.source_chunks[0].source_path == (
        "source_chunks/P__main__c0.json"
    )
    assert all(
        "core_text" not in record.payload
        for record in bundle.records
    )


def test_existing_attempt_preserves_node_and_edge_provenance(tmp_path):
    result = load_existing_extraction_semantic_ir(
        _write_attempt(tmp_path)
    )

    entity = next(
        record
        for record in result.bundle.records
        if record.ref.object_kind == "entity"
        and record.ref.object_id == "material:a"
    )
    edge = next(
        record
        for record in result.bundle.records
        if record.ref.object_kind == "edge"
    )

    assert entity.ref.source_path == "chunks/P__main__c0.json"
    assert entity.source_chunk_ref is not None
    assert entity.source_chunk_ref.object_id == "P:main:c0"
    assert edge.payload["source"] == "material:a"
    assert edge.payload["target"] == "material:b"
    assert edge.source_chunk_ref == entity.source_chunk_ref


def test_existing_attempt_rebases_stale_recorded_paths(tmp_path):
    attempt = _write_attempt(tmp_path, stale_paths=True)

    result = load_existing_extraction_semantic_ir(attempt)

    assert result.artifact_path_rebase_count == 2
    assert result.bundle.source_chunks[0].source_path == (
        "source_chunks/P__main__c0.json"
    )
    assert any(
        record.ref.source_path == "chunks/P__main__c0.json"
        for record in result.bundle.records
        if record.ref.object_kind == "entity"
    )


def test_bundle_identity_depends_on_artifacts_not_recorded_paths(tmp_path):
    attempt = _write_attempt(tmp_path)
    first = load_existing_extraction_semantic_ir(attempt)

    active_path = attempt / "active_chunks.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    active["chunks"][0]["source_path"] = (
        "/moved/source_chunks/P__main__c0.json"
    )
    active["chunks"][0]["output_path"] = (
        "/moved/chunks/P__main__c0.json"
    )
    active_path.write_text(json.dumps(active), encoding="utf-8")

    second = load_existing_extraction_semantic_ir(attempt)

    assert second.bundle.bundle_id == first.bundle.bundle_id


def test_existing_attempt_rejects_source_graph_identity_mismatch(tmp_path):
    attempt = _write_attempt(tmp_path)
    source_path = attempt / "source_chunks" / "P__main__c0.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["chunk_id"] = "P:main:wrong"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="source chunk chunk_id"):
        load_existing_extraction_semantic_ir(attempt)
