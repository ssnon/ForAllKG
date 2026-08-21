from __future__ import annotations

import pytest

from pipeline_core.corpus.extraction.draft_schema import KnowledgeGraphDraft
from pipeline_core.corpus.graph.graph_validation import collect_graph_issues
from pipeline_core.corpus.graph.knowledge_graph_schema import KnowledgeGraph


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


def test_explicit_empty_relation_contract_does_not_disable_legacy_gate():
    draft = KnowledgeGraphDraft.model_validate(
        _payload()
    )

    # Explicit relation-contract mode is active because the argument
    # is not None. With an empty tuple there are no endpoint relation
    # constraints to apply.
    report = collect_graph_issues(
        draft,
        relation_constraints=(),
    )

    relation_issues = [
        item
        for item in report.issues
        if item.stage.value == "relation"
    ]

    assert relation_issues == []

    # Strict wire construction nevertheless applies the historical
    # direct-call DAC relation compatibility policy.
    with pytest.raises(
        ValueError,
        match="Graph relation validation failed",
    ):
        KnowledgeGraph.model_validate(
            draft.model_dump()
        )


def test_historical_gate_preserves_has_metal_error_details():
    draft = KnowledgeGraphDraft.model_validate(
        _payload()
    )

    with pytest.raises(ValueError) as captured:
        KnowledgeGraph.model_validate(
            draft.model_dump()
        )

    message = str(captured.value)

    assert "HAS_METAL source must be" in message
    assert "HAS_METAL target must be" in message
