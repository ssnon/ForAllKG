from __future__ import annotations

import pytest

from dac_her.schemas import (
    EntityNode,
    ExperimentNode,
    EvidencePointer,
    KGEdge,
    KnowledgeGraph,
)


def _pointer(*, document_id: str = "main") -> EvidencePointer:
    return EvidencePointer(
        document_id=document_id,
        document_role="main",
        page_id=None,
        asset_ids=[],
        locator_text=None,
    )


def _edge(
    source: str,
    relation: str,
    target: str,
    *,
    document_id: str = "main",
) -> KGEdge:
    return KGEdge(
        source=source,
        relation=relation,
        target=target,
        evidence_type="structural_characterization",
        evidence_strength="direct",
        evidence_text="Source-grounded evidence.",
        confidence="high",
        evidence_pointers=[
            _pointer(document_id=document_id)
        ],
        subsection=None,
    )


def _graph(
    *,
    entities=None,
    experiments=None,
    edges=None,
) -> KnowledgeGraph:
    return KnowledgeGraph(
        paper_id="paper",
        chunk_id="chunk",
        section="Results",
        document_id="main",
        document_role="main",
        page_ids=[],
        asset_ids=[],
        entities=list(entities or []),
        experiments=list(experiments or []),
        calculations=[],
        measurements=[],
        measurement_groups=[],
        observation_claims=[],
        mechanism_claims=[],
        edges=list(edges or []),
    )


def _entity(
    node_id: str,
    node_type: str,
) -> EntityNode:
    return EntityNode(
        id=node_id,
        type=node_type,
        label=node_id,
        description=None,
    )


def _experiment(node_id: str) -> ExperimentNode:
    return ExperimentNode(
        id=node_id,
        name="Experiment",
        experiment_type="custom_method",
        experiment_family="spectroscopy",
        method_label="Custom method",
        raw_method_name=None,
        conditions=[],
        description=None,
    )


def test_known_dac_relation_semantics_remain_enforced():
    graph = _graph(
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

    assert graph.edges[0].relation == "HAS_METAL"


def test_invalid_known_dac_relation_direction_is_rejected():
    with pytest.raises(ValueError):
        _graph(
            entities=[
                _entity("catalyst", "Catalyst"),
                _entity("metal", "Metal"),
            ],
            edges=[
                _edge(
                    "metal",
                    "HAS_METAL",
                    "catalyst",
                ),
            ],
        )


def test_domain_extensible_relation_is_structurally_accepted():
    graph = _graph(
        entities=[
            _entity(
                "substrate",
                "PlasmonicSubstrate",
            ),
        ],
        experiments=[
            _experiment("experiment"),
        ],
        edges=[
            _edge(
                "substrate",
                "TESTED_IN",
                "experiment",
            ),
        ],
    )

    assert graph.edges[0].relation == "TESTED_IN"


def test_undefined_edge_endpoint_is_structurally_rejected():
    with pytest.raises(
        ValueError,
        match="undefined source",
    ):
        _graph(
            entities=[
                _entity("metal", "Metal"),
            ],
            edges=[
                _edge(
                    "missing",
                    "HAS_METAL",
                    "metal",
                ),
            ],
        )


def test_duplicate_node_ids_are_structurally_rejected():
    with pytest.raises(
        ValueError,
        match="Duplicate node IDs",
    ):
        _graph(
            entities=[
                _entity("same", "Catalyst"),
                _entity("same", "Metal"),
            ],
        )


def test_edge_provenance_must_match_graph_document():
    with pytest.raises(
        ValueError,
        match="Graph provenance validation failed",
    ):
        _graph(
            entities=[
                _entity("catalyst", "Catalyst"),
                _entity("metal", "Metal"),
            ],
            edges=[
                _edge(
                    "catalyst",
                    "HAS_METAL",
                    "metal",
                    document_id="wrong-document",
                ),
            ],
        )
