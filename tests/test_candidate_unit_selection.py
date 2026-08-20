from __future__ import annotations

import networkx as nx

from dac_her.candidate_unit_selection import CandidateUnitSelectionPolicy, CandidateUnitSelector
from pipeline_core.discovery.candidate_units import CandidateUnitBuilder, confirmed_navigation_graph


def _confirmed(g: nx.DiGraph, u: str, v: str, relation: str = "INTERPRETED_AS", cost: float = 1.0) -> None:
    g.add_edge(
        u,
        v,
        relation=relation,
        edge_class="scientific_confirmed",
        exploration_cost=cost,
        requires_verification=False,
        reverse_navigation=False,
    )


def _candidate_pair(g: nx.DiGraph, anchor: str, candidate: str) -> None:
    g.add_edge(
        anchor,
        candidate,
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_class="semantic_candidate",
        exploration_cost=2.5,
        requires_verification=True,
        reverse_navigation=False,
        edge_id=f"f:{anchor}:{candidate}",
        selected_original_edge_id=f"o:{anchor}:{candidate}",
    )
    g.add_edge(
        candidate,
        anchor,
        relation="GROUNDS_SEMANTIC_CANDIDATE",
        edge_class="semantic_candidate",
        exploration_cost=3.1,
        requires_verification=True,
        reverse_navigation=True,
        edge_id=f"r:{candidate}:{anchor}",
        selected_original_edge_id=f"o:{anchor}:{candidate}",
    )


def _base_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    # Good route: S -> MECH_L -> A -> C -> B -> MECH_R -> HER
    for node, typ, label in [
        ("S", "CoordinationMotif", "nitrogen coordination"),
        ("ML", "MechanismClaim", "coordination changes symmetry"),
        ("A", "CatalystModel", "geometry anchor"),
        ("B", "MechanismClaim", "orbital hybridization anchor"),
        ("MR", "MechanismClaim", "adsorption mechanism"),
        ("HER", "Reaction", "Hydrogen evolution reaction"),
        ("C", "BridgeConcept", "coordination-geometry–formation-energy dependence"),
    ]:
        attrs = {"type": typ, "label": label}
        if node == "C":
            attrs.update(policy_lane="semantic_candidate", requires_verification=True)
        g.add_node(node, **attrs)
    _confirmed(g, "S", "ML", "INTERPRETED_AS")
    _confirmed(g, "ML", "A", "APPLIES_TO")
    _candidate_pair(g, "A", "C")
    _candidate_pair(g, "B", "C")
    _confirmed(g, "B", "MR", "INTERPRETED_AS")
    _confirmed(g, "MR", "HER", "APPLIES_TO")

    # Bad reaction-switch route: S -> CO2 -> X -> D -> Y -> HER
    for node, typ, label in [
        ("CO2", "Reaction", "CO2 reduction reaction"),
        ("X", "CatalystModel", "CO2 catalyst"),
        ("Y", "CatalystModel", "water-splitting catalyst"),
        ("D", "BridgeConcept", "generic atom-utilization relation"),
    ]:
        attrs = {"type": typ, "label": label}
        if node == "D":
            attrs.update(policy_lane="semantic_candidate", requires_verification=True)
        g.add_node(node, **attrs)
    _confirmed(g, "S", "CO2", "CATALYZES")
    _confirmed(g, "CO2", "X", "APPLIES_TO")
    _candidate_pair(g, "X", "D")
    _candidate_pair(g, "Y", "D")
    _confirmed(g, "Y", "HER", "CATALYZES")
    return g


def test_selector_requires_distinct_anchors_and_penalizes_reaction_switch() -> None:
    g = _base_graph()
    units = CandidateUnitBuilder(g).build(bridge_capable_only=True)
    unit_by_node = {unit.candidate_node_id: unit for unit in units}
    selector = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        policy=CandidateUnitSelectionPolicy(max_depth=8, top_k=10, max_routes_per_unit=1),
        unit_relevance={"C": 0.8, "D": 0.8},
    )
    routes = selector.enumerate_routes(
        units,
        [{"node_id": "S", "semantic_similarity": 0.8}],
        [{"node_id": "HER", "semantic_similarity": 0.8}],
    )
    assert routes
    assert all(route.entry_anchor.node_id != route.exit_anchor.node_id for route in routes)
    good = next(route for route in routes if route.unit.candidate_node_id == "C")
    bad = next(route for route in routes if route.unit.candidate_node_id == "D")
    assert good.score.reaction_switch_penalty == 0.0
    assert bad.score.reaction_switch_penalty > 0.0
    assert good.score.total > bad.score.total
    assert unit_by_node["C"].anchor_count == 2


def test_selected_route_uses_one_unit_but_two_candidate_edges_semantically() -> None:
    g = _base_graph()
    units = CandidateUnitBuilder(g).build(bridge_capable_only=True)
    selector = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        policy=CandidateUnitSelectionPolicy(max_depth=8, top_k=2),
        unit_relevance={"C": 0.9, "D": 0.2},
    )
    routes = selector.enumerate_routes(
        units,
        [{"node_id": "S", "semantic_similarity": 0.8}],
        [{"node_id": "HER", "semantic_similarity": 0.8}],
    )
    selected = selector.select(routes)
    assert selected
    row = selected[0].to_dict()
    assert row["candidate_unit_count"] == 1
    assert row["candidate_edge_count"] == 2
