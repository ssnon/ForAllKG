from __future__ import annotations

import pytest

from pipeline_core.corpus.extraction.draft_schema import KnowledgeGraphDraft
from pipeline_core.corpus.graph.graph_domain import RelationConstraint
from pipeline_core.corpus.graph_normalization import (
    normalize_graph_vocabularies,
)
from pipeline_core.corpus.strict_validation import (
    ValidationContext,
    finalize_draft,
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
            _entity("support", "Support"),
            _entity("metal", "Metal"),
        ],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [
            _edge(
                "support",
                "HAS_METAL",
                "metal",
            ),
        ],
    }


def _validation_context():
    return ValidationContext(
        paper_id="paper",
        chunk_id="chunk",
        section="Results",
        document_id="main",
        document_role="main",
        page_ids=[],
        asset_ids=[],
    )


def _explicit_support_has_metal_contract():
    return (
        RelationConstraint(
            relation="HAS_METAL",
            source_types=frozenset({"Support"}),
            target_types=frozenset({"Metal"}),
            severity="error",
        ),
    )


def _already_validated_context():
    return {
        RELATION_SEMANTICS_ALREADY_VALIDATED_CONTEXT_KEY: True,
    }


def test_domain_aware_finalization_uses_explicit_relation_contract_end_to_end():
    draft = KnowledgeGraphDraft.model_validate(
        _payload()
    )

    result = finalize_draft(
        draft=draft,
        context=_validation_context(),
        experiment_registry=object(),
        metric_registry=object(),
        relation_constraints=(
            _explicit_support_has_metal_contract()
        ),
    )

    assert result.report.valid
    assert result.graph is not None
    assert result.graph.edges[0].relation == "HAS_METAL"


def test_no_explicit_relation_contract_preserves_legacy_rejection():
    draft = KnowledgeGraphDraft.model_validate(
        _payload()
    )

    result = finalize_draft(
        draft=draft,
        context=_validation_context(),
        experiment_registry=object(),
        metric_registry=object(),
        relation_constraints=None,
    )

    assert not result.report.valid
    assert result.graph is None


def test_normalization_default_preserves_legacy_relation_gate():
    graph = KnowledgeGraph.model_validate(
        _payload(),
        context=_already_validated_context(),
    )

    with pytest.raises(
        ValueError,
        match="Graph relation validation failed",
    ):
        normalize_graph_vocabularies(
            graph,
            experiment_registry=object(),
            metric_registry=object(),
        )


def test_normalization_can_propagate_upstream_relation_validation():
    graph = KnowledgeGraph.model_validate(
        _payload(),
        context=_already_validated_context(),
    )

    normalized, issues = normalize_graph_vocabularies(
        graph,
        experiment_registry=object(),
        metric_registry=object(),
        relation_semantics_already_validated=True,
    )

    assert normalized.edges[0].relation == "HAS_METAL"
    assert issues == []


def test_standalone_model_validation_still_rejects_legacy_invalid_relation():
    with pytest.raises(
        ValueError,
        match="Graph relation validation failed",
    ):
        KnowledgeGraph.model_validate(
            _payload()
        )
