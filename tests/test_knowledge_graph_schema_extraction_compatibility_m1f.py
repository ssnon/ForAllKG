from __future__ import annotations

import dac_her.schemas as legacy
import pipeline_core.knowledge_graph_schema as core


def test_legacy_knowledge_graph_is_core_object():
    assert legacy.KnowledgeGraph is core.KnowledgeGraph


def test_knowledge_graph_is_owned_by_pipeline_core():
    assert core.KnowledgeGraph.__module__ == (
        "pipeline_core.knowledge_graph_schema"
    )


def test_legacy_facade_preserves_knowledge_graph_constructor():
    graph = legacy.KnowledgeGraph(
        paper_id="paper",
        chunk_id="chunk",
        section="Results",
        document_id="main",
        document_role="main",
        page_ids=[],
        asset_ids=[],
        entities=[],
        experiments=[],
        calculations=[],
        measurements=[],
        measurement_groups=[],
        observation_claims=[],
        mechanism_claims=[],
        edges=[],
    )

    assert isinstance(graph, core.KnowledgeGraph)
    assert graph.paper_id == "paper"
    assert graph.chunk_id == "chunk"
