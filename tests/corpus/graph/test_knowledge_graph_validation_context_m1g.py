from __future__ import annotations

import pytest

from pipeline_core.corpus.graph.knowledge_graph_compat_validation import (
    validate_graph_integrity_compat,
)
from pipeline_core.corpus.graph.knowledge_graph_schema import KnowledgeGraph
from pipeline_core.corpus.graph.knowledge_graph_validation_context import (
    RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY,
)


def _pointer():
    return {
        "document_id": "main",
        "document_role": "main",
        "page_id": None,
        "asset_ids": [],
        "locator_text": None,
    }


def _entity(node_id, node_type):
    return {
        "id": node_id,
        "type": node_type,
        "label": node_id,
        "description": None,
    }


def _edge(source, relation, target):
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "evidence_type": "structural_characterization",
        "evidence_strength": "direct",
        "evidence_text": "Evidence.",
        "confidence": "high",
        "evidence_pointers": [_pointer()],
        "subsection": None,
    }


def _payload():
    return {
        "paper_id": "paper",
        "chunk_id": "chunk",
        "section": "Results",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [],
        "asset_ids": [],
        "entities": [
            _entity("metal", "Metal"),
            _entity("catalyst", "Catalyst"),
        ],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [
            _edge(
                "metal",
                "HAS_METAL",
                "catalyst",
            ),
        ],
    }


def _validated_context():
    return {
        RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY: True,
    }


def test_default_model_validation_preserves_legacy_relation_gate():
    with pytest.raises(
        ValueError,
        match="Graph relation validation failed",
    ):
        KnowledgeGraph.model_validate(
            _payload()
        )


def test_explicit_context_skips_only_legacy_relation_gate():
    graph = KnowledgeGraph.model_validate(
        _payload(),
        context=_validated_context(),
    )

    assert graph.edges[0].relation == "HAS_METAL"


def test_context_does_not_skip_shared_structural_validation():
    payload = _payload()
    payload["edges"][0]["source"] = "missing"

    with pytest.raises(
        ValueError,
        match="undefined source",
    ):
        KnowledgeGraph.model_validate(
            payload,
            context=_validated_context(),
        )


def test_compat_helper_default_preserves_legacy_relations():
    graph = KnowledgeGraph.model_construct(
        **KnowledgeGraph.model_validate(
            _payload(),
            context=_validated_context(),
        ).__dict__
    )

    with pytest.raises(
        ValueError,
        match="Graph relation validation failed",
    ):
        validate_graph_integrity_compat(graph)


def test_compat_helper_can_skip_legacy_relations_explicitly():
    graph = KnowledgeGraph.model_validate(
        _payload(),
        context=_validated_context(),
    )

    assert (
        validate_graph_integrity_compat(
            graph,
            validate_legacy_relations=False,
        )
        is graph
    )
