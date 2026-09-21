from __future__ import annotations

import hashlib

from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)
from pipeline_core.corpus.semantic_ir.task_scope import (
    resolve_graph_explorer_task_scope_from_bundles,
)


def _chunk_ref(paper: str, chunk: str) -> SemanticObjectRef:
    return SemanticObjectRef(
        paper_id=paper,
        chunk_id=chunk,
        object_kind="chunk",
        object_id=chunk,
        source_path=f"source_chunks/{chunk}.json",
    )


def _chunk_record(
    paper: str,
    chunk: str,
    *,
    page: int,
) -> SemanticIRRecord:
    ref = _chunk_ref(paper, chunk)
    return SemanticIRRecord(
        ref=ref,
        payload={
            "paper_id": paper,
            "chunk_id": chunk,
            "document_id": "main",
            "document_role": "main",
            "section": "Results",
            "page_ids": [page],
            "asset_ids": [],
        },
        source_chunk_ref=ref,
    )


def _entity_record(
    paper: str,
    chunk: str,
    *,
    object_id: str,
    entity_type: str,
    page: int,
) -> SemanticIRRecord:
    return SemanticIRRecord(
        ref=SemanticObjectRef(
            paper_id=paper,
            chunk_id=chunk,
            object_kind="entity",
            object_id=object_id,
            source_path=f"chunks/{chunk}.json",
        ),
        payload={
            "id": object_id,
            "type": entity_type,
            "label": object_id,
            "description": None,
        },
        source_chunk_ref=_chunk_ref(paper, chunk),
    )


def _bundle(*records: SemanticIRRecord) -> SemanticIRBundle:
    chunks = {
        record.source_chunk_ref.identity_key(): record.source_chunk_ref
        for record in records
        if record.source_chunk_ref is not None
    }
    return SemanticIRBundle(
        bundle_id="bundle:P1",
        paper_id="P1",
        records=list(records),
        source_chunks=[chunks[key] for key in sorted(chunks)],
    )


def _packet(
    node_id: str,
    *,
    node_type: str = "Material",
    page_id: int | None = None,
) -> dict:
    edges = {}
    if page_id is not None:
        edges = {
            "e1": {
                "scientific_source": node_id,
                "scientific_target": "corpus::other",
                "relation": "RELATED_TO",
                "supporting_node_ids": [node_id],
                "evidence_pointers": [{
                    "document_id": "main",
                    "document_role": "main",
                    "page_id": page_id,
                    "asset_ids": [],
                    "locator_text": "Results",
                }],
            }
        }
    return {
        "schema_version": "graph-explorer-input-v1",
        "task": {"task_id": "task:test"},
        "direct_concept_hits": [],
        "paths": [{
            "path_id": "p1",
            "node_ids": [node_id],
            "endpoint": {},
            "steps": [],
        }],
        "evidence_catalog": {
            "nodes": {
                node_id: {
                    "node_id": node_id,
                    "node_type": node_type,
                    "label": node_id,
                    "node_text": node_id,
                }
            },
            "edges": edges,
        },
        "alignment_contexts": [],
    }


def test_same_paper_local_id_across_chunks_is_multi_mention_not_ambiguous():
    records = [
        _chunk_record("P1", "c1", page=1),
        _chunk_record("P1", "c2", page=2),
        _entity_record(
            "P1", "c1", object_id="sample", entity_type="Material", page=1
        ),
        _entity_record(
            "P1", "c2", object_id="sample", entity_type="Material", page=2
        ),
    ]
    result = resolve_graph_explorer_task_scope_from_bundles(
        packet_payload=_packet("paper::P1::sample"),
        bundles={"P1": _bundle(*records)},
    )

    assert result.ambiguous_paper_local_node_ids == []
    assert result.multi_mention_paper_local_node_ids == ["paper::P1::sample"]
    assert len(result.resolved_objects) == 1
    row = result.resolved_objects[0]
    assert row.resolution_mode == "paper_merge_multi_mention"
    assert row.candidate_match_count == 2
    assert row.selected_match_count == 2
    assert len(row.source_chunk_refs) == 2


def test_packet_page_provenance_narrows_multi_mention_to_supporting_chunk():
    records = [
        _chunk_record("P1", "c1", page=1),
        _chunk_record("P1", "c2", page=2),
        _entity_record(
            "P1", "c1", object_id="sample", entity_type="Material", page=1
        ),
        _entity_record(
            "P1", "c2", object_id="sample", entity_type="Material", page=2
        ),
    ]
    result = resolve_graph_explorer_task_scope_from_bundles(
        packet_payload=_packet("paper::P1::sample", page_id=2),
        bundles={"P1": _bundle(*records)},
    )

    row = result.resolved_objects[0]
    assert row.resolution_mode == "provenance_narrowed_multi_mention"
    assert row.candidate_match_count == 2
    assert row.selected_match_count == 1
    assert row.source_chunk_refs[0].chunk_id == "c2"
    assert result.provenance_narrowed_paper_local_node_ids == [
        "paper::P1::sample"
    ]


def test_collision_scoped_paper_node_reconstructs_exact_chunk_mention_id():
    records = [
        _chunk_record("P1", "c1", page=1),
        _chunk_record("P1", "c2", page=2),
        _entity_record(
            "P1", "c1", object_id="shared", entity_type="Material", page=1
        ),
        _entity_record(
            "P1",
            "c2",
            object_id="shared",
            entity_type="PlasmonicSubstrate",
            page=2,
        ),
    ]
    digest = hashlib.sha256(
        "c2|shared|PlasmonicSubstrate".encode("utf-8")
    ).hexdigest()[:12]
    scoped = f"shared__mention_plasmonicsubstrate_{digest}"
    corpus_id = f"paper::P1::{scoped}"

    result = resolve_graph_explorer_task_scope_from_bundles(
        packet_payload=_packet(corpus_id, node_type="PlasmonicSubstrate"),
        bundles={"P1": _bundle(*records)},
    )

    assert result.unresolved_paper_local_node_ids == []
    assert result.collision_scoped_paper_local_node_ids == [corpus_id]
    row = result.resolved_objects[0]
    assert row.resolution_mode == "collision_scoped_exact"
    assert row.semantic_refs[0].chunk_id == "c2"
