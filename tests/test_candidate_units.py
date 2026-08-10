from __future__ import annotations

import networkx as nx

from dac_her.candidate_units import CandidateUnitBuilder, confirmed_navigation_graph


def _candidate_pair(g: nx.DiGraph, anchor: str, candidate: str) -> None:
    g.add_edge(
        anchor,
        candidate,
        edge_id=f"nav:{anchor}:{candidate}",
        selected_original_edge_id=f"orig:{anchor}:{candidate}",
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_class="semantic_candidate",
        exploration_cost=2.5,
        requires_verification=True,
        reverse_navigation=False,
    )
    g.add_edge(
        candidate,
        anchor,
        edge_id=f"nav:{candidate}:{anchor}",
        selected_original_edge_id=f"orig:{anchor}:{candidate}",
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_class="semantic_candidate",
        exploration_cost=3.1,
        requires_verification=True,
        reverse_navigation=True,
    )


def test_builder_groups_original_anchors_into_one_unit() -> None:
    g = nx.DiGraph()
    g.add_node("A", type="CatalystModel", label="anchor A")
    g.add_node("B", type="MechanismClaim", label="anchor B")
    g.add_node("C", type="BridgeConcept", label="candidate C", policy_lane="semantic_candidate", requires_verification=True)
    g.add_node("D", type="BridgeConcept", label="single-anchor D", policy_lane="semantic_candidate", requires_verification=True)
    _candidate_pair(g, "A", "C")
    _candidate_pair(g, "B", "C")
    _candidate_pair(g, "A", "D")

    units = CandidateUnitBuilder(g).build(bridge_capable_only=True)
    assert len(units) == 1
    assert units[0].candidate_node_id == "C"
    assert {anchor.node_id for anchor in units[0].anchors} == {"A", "B"}
    assert units[0].anchor_count == 2


def test_confirmed_graph_removes_candidate_nodes_and_edges() -> None:
    g = nx.DiGraph()
    g.add_node("A", type="MechanismClaim")
    g.add_node("B", type="ObservationClaim")
    g.add_node("C", type="BridgeConcept", policy_lane="semantic_candidate", requires_verification=True)
    g.add_edge("A", "B", relation="INTERPRETED_AS", edge_class="scientific_confirmed", exploration_cost=1.0)
    _candidate_pair(g, "A", "C")
    confirmed = confirmed_navigation_graph(g)
    assert set(confirmed.nodes) == {"A", "B"}
    assert confirmed.has_edge("A", "B")
    assert "C" not in confirmed
