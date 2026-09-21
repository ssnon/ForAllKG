from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.corpus.semantic_ir.task_scope import (
    resolve_graph_explorer_task_scope,
)


def _edge(source: str, relation: str, target: str) -> dict:
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "evidence_type": "experimental_observation",
        "evidence_strength": "direct",
        "evidence_text": "reported",
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


def _write_attempt(root: Path, paper_id: str, *, condition: bool = True) -> Path:
    attempt = root / paper_id / "runs" / "r" / "attempts" / "a"
    (attempt / "source_chunks").mkdir(parents=True)
    (attempt / "chunks").mkdir()
    chunk_id = f"{paper_id}:main:c0"
    source_path = attempt / "source_chunks" / f"{paper_id}.json"
    output_path = attempt / "chunks" / f"{paper_id}.json"
    source_path.write_text(json.dumps({
        "paper_id": paper_id,
        "chunk_id": chunk_id,
        "document_id": "main",
        "document_role": "main",
        "section": "Results",
        "page_ids": [],
        "asset_ids": [],
        "core_text": "reported",
    }), encoding="utf-8")
    conditions = ([{
        "name": "temperature",
        "value_numeric": 25.0,
        "value_text": None,
        "unit": "C",
        "reference": None,
    }] if condition else [])
    output_path.write_text(json.dumps({
        "paper_id": paper_id,
        "chunk_id": chunk_id,
        "section": "Results",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [],
        "asset_ids": [],
        "entities": [{
            "id": "sample",
            "type": "Material",
            "label": "sample",
            "description": None,
        }],
        "experiments": [{
            "id": "exp",
            "name": "Raman",
            "experiment_type": "raman",
            "experiment_family": "spectroscopy",
            "method_label": "Raman",
            "raw_method_name": None,
            "conditions": conditions,
            "description": None,
        }],
        "calculations": [],
        "measurements": [{
            "id": "m",
            "metric_id": "sers_intensity",
            "metric": "SERS intensity",
            "subject_id": "sample",
            "source_expression": "reported",
            "group_id": None,
            "value_numeric": 1.0,
            "value_text": None,
            "unit": None,
            "uncertainty": None,
            "qualifier": None,
            "basis": None,
            "conditions": conditions,
            "description": None,
        }],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [
            _edge("exp", "HAS_MEASUREMENT", "m"),
            _edge("m", "MEASURED_FOR", "sample"),
        ],
    }), encoding="utf-8")
    (attempt / "active_chunks.json").write_text(json.dumps({
        "paper_id": paper_id,
        "run_id": "r",
        "attempt_id": "a",
        "active_chunk_count": 1,
        "chunks": [{
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "source_path": str(source_path),
            "output_path": str(output_path),
        }],
    }), encoding="utf-8")
    return attempt


def _write_packet(path: Path, node_ids: list[str]) -> None:
    nodes = {
        node_id: {
            "node_id": node_id,
            "node_type": "Material",
            "label": node_id,
            "node_text": node_id,
        }
        for node_id in node_ids
    }
    payload = {
        "schema_version": "graph-explorer-input-v1",
        "task": {"task_id": "task:K01"},
        "direct_concept_hits": [],
        "paths": [{
            "path_id": "p1",
            "node_ids": node_ids,
            "endpoint": {},
            "steps": [],
        }],
        "evidence_catalog": {"nodes": nodes, "edges": {}},
        "alignment_contexts": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_packet_resolves_exact_paper_local_nodes_to_source_chunks(tmp_path):
    root = tmp_path / "corpus"
    _write_attempt(root, "P1")
    packet = tmp_path / "explorer.packet.json"
    _write_packet(packet, ["paper::P1::sample", "paper::P1::m"])

    result, bundles = resolve_graph_explorer_task_scope(
        packet_path=packet,
        corpus_root=root,
    )

    assert result.referenced_paper_ids == ["P1"]
    assert result.available_paper_ids == ["P1"]
    assert result.missing_paper_ids == []
    assert len(result.resolved_objects) == 2
    assert len(result.source_chunks) == 1
    assert set(bundles) == {"P1"}
    assert result.text_similarity_resolution_used is False


def test_packet_missing_paper_is_reported_not_treated_as_negative_evidence(tmp_path):
    root = tmp_path / "corpus"
    _write_attempt(root, "P1")
    packet = tmp_path / "explorer.packet.json"
    _write_packet(packet, ["paper::P1::sample", "paper::LEGACY::node"])

    result, _ = resolve_graph_explorer_task_scope(
        packet_path=packet,
        corpus_root=root,
    )

    assert result.referenced_paper_ids == ["LEGACY", "P1"]
    assert result.available_paper_ids == ["P1"]
    assert result.missing_paper_ids == ["LEGACY"]
    assert len(result.resolved_objects) == 1


def test_alignment_context_member_nodes_are_used_for_resolution(tmp_path):
    root = tmp_path / "corpus"
    _write_attempt(root, "P1")
    packet = tmp_path / "explorer.packet.json"
    payload = {
        "schema_version": "graph-explorer-input-v1",
        "task": {"task_id": "task:align"},
        "direct_concept_hits": [],
        "paths": [],
        "evidence_catalog": {"nodes": {}, "edges": {}},
        "alignment_contexts": [{
            "hub_node_id": "corpus::alignment::x",
            "member_node_ids": ["paper::P1::sample"],
            "traversed_entry_node_ids": [],
            "traversed_exit_node_ids": [],
        }],
    }
    packet.write_text(json.dumps(payload), encoding="utf-8")

    result, _ = resolve_graph_explorer_task_scope(
        packet_path=packet,
        corpus_root=root,
    )

    assert result.referenced_corpus_node_count == 2
    assert result.paper_local_node_count == 1
    assert result.alignment_or_nonpaper_node_count == 1
    assert len(result.resolved_objects) == 1
