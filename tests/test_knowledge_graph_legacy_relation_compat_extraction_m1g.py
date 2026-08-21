from __future__ import annotations

import pytest

from pipeline_core.corpus.graph.knowledge_graph_legacy_relation_compat import (
    validate_legacy_relation_semantics_compat,
)
from pipeline_core.corpus.graph.knowledge_graph_schema import KnowledgeGraph


def _pointer():
    return {
        "document_id": "main",
        "document_role": "main",
        "page_id": None,
        "asset_ids": [],
        "locator_text": None,
    }


def _edge(source, relation, target):
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "evidence_type": "structural_characterization",
        "evidence_strength": "direct",
        "evidence_text": "Source-grounded evidence.",
        "confidence": "high",
        "evidence_pointers": [_pointer()],
        "subsection": None,
    }


def _entity(node_id, node_type):
    return {
        "id": node_id,
        "type": node_type,
        "label": node_id,
        "description": None,
    }


def _payload(*, entities=None, edges=None):
    return {
        "paper_id": "paper",
        "chunk_id": "chunk",
        "section": "Results",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [],
        "asset_ids": [],
        "entities": list(entities or []),
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": list(edges or []),
    }


def test_legacy_relation_helper_accepts_valid_dac_relation():
    graph = KnowledgeGraph.model_validate(
        _payload(
            entities=[
                _entity("catalyst", "Catalyst"),
                _entity("metal", "Metal"),
            ],
            edges=[
                _edge(
                    "catalyst",
                    "HAS_METAL",
                    "metal",
                ),
            ],
        )
    )

    assert (
        validate_legacy_relation_semantics_compat(graph)
        is graph
    )


def test_legacy_relation_helper_ignores_unknown_relation():
    graph = KnowledgeGraph.model_validate(
        _payload(
            entities=[
                _entity("substrate", "PlasmonicSubstrate"),
                _entity("metal", "Metal"),
            ],
            edges=[
                _edge(
                    "substrate",
                    "TESTED_IN",
                    "metal",
                ),
            ],
        )
    )

    # Unknown relations are intentionally ignored by the historical
    # direct-call relation fallback.
    assert (
        validate_legacy_relation_semantics_compat(graph)
        is graph
    )


def test_invalid_has_metal_is_still_rejected_by_public_model():
    payload = _payload(
        entities=[
            _entity("metal", "Metal"),
            _entity("catalyst", "Catalyst"),
        ],
        edges=[
            _edge(
                "metal",
                "HAS_METAL",
                "catalyst",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Graph relation validation failed",
    ):
        KnowledgeGraph.model_validate(payload)
