from __future__ import annotations

import pipeline_core.corpus.graph.knowledge_graph_schema as schema
from pipeline_core.corpus.graph.knowledge_graph_compat_validation import (
    validate_graph_integrity_compat,
)


def _empty_payload():
    return {
        "paper_id": "paper",
        "chunk_id": "chunk",
        "section": "Results",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [],
        "asset_ids": [],
        "entities": [],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [],
    }


def test_compat_helper_returns_same_valid_graph_object():
    graph = schema.KnowledgeGraph.model_validate(
        _empty_payload()
    )

    assert validate_graph_integrity_compat(graph) is graph


def test_knowledge_graph_model_validator_delegates_to_helper(
    monkeypatch,
):
    calls = []

    def fake_validator(graph):
        calls.append(graph)
        return graph

    monkeypatch.setattr(
        schema,
        "validate_graph_integrity_compat",
        fake_validator,
    )

    graph = schema.KnowledgeGraph.model_validate(
        _empty_payload()
    )

    assert len(calls) == 1
    assert calls[0] is graph


def test_direct_model_keeps_public_validator_method():
    assert hasattr(
        schema.KnowledgeGraph,
        "validate_graph_integrity",
    )
