import networkx as nx
import pytest

import dac_her.candidate_unit_selection as candidate_unit_selection_module
from dac_her.candidate_unit_selection import CandidateUnitSelector

from pipeline_core.discovery_semantics import (
    is_alignment_edge,
    is_alignment_node,
    is_mechanism_edge,
    is_mechanism_node,
    is_scaffold_edge,
    is_shared_entity_node,
)
from pipeline_core.domain_profile import DiscoverySemantics
from dac_her.explorer_packet import GraphExplorerPacketBuilder
from dac_her.path_quality import PathQualityScorer


def _sers_like_semantics() -> DiscoverySemantics:
    return DiscoverySemantics(
        generic_entity_types=frozenset({
            "PLASMONICSUBSTRATE",
            "NANOSTRUCTURE",
            "MATERIAL",
        }),
        mechanism_node_markers=(
            "PLASMONICMECHANISM",
            "MECHANISM",
        ),
        mechanism_relation_markers=("FOCUS", "COUPL"),
        scaffold_relations=frozenset({
            "HAS_COMPONENT",
            "HAS_STRUCTURAL_MOTIF",
        }),
        context_node_types=frozenset({"ANALYTE"}),
        shared_entity_types=frozenset({
            "PLASMONICSUBSTRATE",
            "NANOSTRUCTURE",
        }),
        legacy_mechanism_id_prefixes=(),
    )


def test_shared_discovery_detectors_use_selected_semantics():
    semantics = _sers_like_semantics()
    assert is_mechanism_node(
        "paper::p::x",
        {"type": "PlasmonicMechanismClaim"},
        semantics,
    )
    assert not is_mechanism_node(
        "paper::p::mech_legacy",
        {"type": "Material"},
        semantics,
    )
    assert is_mechanism_edge(
        {"relation": "FOCUSES_LOCAL_FIELD"},
        semantics,
    )
    assert not is_mechanism_edge(
        {"relation": "CATALYZES"},
        semantics,
    )
    assert is_scaffold_edge(
        {"relation": "HAS_COMPONENT"},
        semantics,
    )
    assert is_shared_entity_node(
        "paper::p::substrate",
        {"type": "PlasmonicSubstrate"},
        semantics,
    )


def test_alignment_semantics_are_technical_not_domain_specific():
    assert is_alignment_node({"type": "CorpusPattern"})
    assert is_alignment_node({"corpus_node_kind": "alignment_hub"})
    assert is_alignment_edge({"edge_class": "registry_alignment"})
    assert is_alignment_edge({
        "evidence_status": "derived_corpus_alignment"
    })


def test_path_quality_uses_injected_domain_semantics():
    semantics = _sers_like_semantics()
    graph = nx.DiGraph()
    graph.add_node("s", type="PlasmonicSubstrate")
    graph.add_node("m", type="PlasmonicMechanismClaim")
    graph.add_node("t", type="Nanostructure")
    graph.add_edge("s", "m", relation="FOCUSES_LOCAL_FIELD")
    graph.add_edge("m", "t", relation="HAS_COMPONENT")

    row = {
        "path_id": "p1",
        "nodes": ["s", "m", "t"],
        "steps": [
            {
                "source": "s",
                "target": "m",
                "relation": "FOCUSES_LOCAL_FIELD",
                "edge_class": "",
            },
            {
                "source": "m",
                "target": "t",
                "relation": "HAS_COMPONENT",
                "edge_class": "",
            },
        ],
        "hop_count": 2,
        "scientific_edge_count": 2,
        "alignment_edge_count": 0,
        "candidate_edge_count": 0,
        "reverse_edge_count": 0,
        "visited_paper_ids": ["p"],
        "visited_paper_count": 1,
    }
    quality = PathQualityScorer(
        graph,
        discovery_semantics=semantics,
    ).score(row)

    assert quality.mechanism_edge_count == 1
    assert quality.mechanism_node_count == 1
    assert quality.mechanism_bearing is True
    assert quality.scaffold_edge_count == 1


def test_explorer_packet_carries_traversal_domain_lineage():
    packet = GraphExplorerPacketBuilder().build(
        traversal_payload={
            "corpus_id": "sers_test",
            "mode": "mechanism",
            "algorithm": "top_n",
            "domain_profile_id": "sers_au_ag",
            "paths": [],
        },
        node_rows=[],
        edge_rows=[],
        corpus_manifest={
            "corpus_id": "sers_test",
            "mode": "mechanism",
            "papers": [],
            "domain_profile_id": "sers_au_ag",
        },
    )
    assert packet.domain_profile_id == "sers_au_ag"


def test_explorer_packet_rejects_explicit_domain_lineage_mismatch():
    with pytest.raises(ValueError, match="domain profile mismatch"):
        GraphExplorerPacketBuilder().build(
            traversal_payload={
                "corpus_id": "x",
                "mode": "mechanism",
                "algorithm": "top_n",
                "domain_profile_id": "sers_au_ag",
                "paths": [],
            },
            node_rows=[],
            edge_rows=[],
            corpus_manifest={
                "corpus_id": "x",
                "mode": "mechanism",
                "papers": [],
                "domain_profile_id": "dac_her",
            },
        )


def test_candidate_unit_right_branch_keeps_selected_semantics(monkeypatch):
    semantics = _sers_like_semantics()
    graph = nx.DiGraph()
    graph.add_node("left", type="PlasmonicSubstrate")
    graph.add_node("candidate", type="CandidateHypothesis")
    graph.add_node("right", type="Nanostructure")
    graph.add_edge("left", "candidate", relation="CANDIDATE")
    graph.add_edge("candidate", "right", relation="CANDIDATE")

    selector = CandidateUnitSelector(
        graph,
        graph.copy(),
    )
    selector.discovery_semantics = semantics

    seen = []

    def fake_is_mechanism_node(node_id, attrs, selected_semantics):
        del node_id, attrs
        seen.append(selected_semantics)
        return True

    monkeypatch.setattr(
        candidate_unit_selection_module,
        "is_mechanism_node",
        fake_is_mechanism_node,
    )

    continuity, *_ = selector._route_diagnostics(
        ("left", "candidate", "right"),
        candidate_id="candidate",
        entry_id="left",
        exit_id="right",
    )

    assert continuity == 1.0
    assert seen
    assert all(item is semantics for item in seen)
