from __future__ import annotations

import networkx as nx

from dac_her.candidate_unit_selection import (
    CandidateUnitSelectionPolicy,
    CandidateUnitSelector,
)
from dac_her.candidate_units import CandidateUnitBuilder, confirmed_navigation_graph
from dac_her.domains import get_domain_profile


def _confirmed(g: nx.DiGraph, u: str, v: str, relation: str = "APPLIES_TO") -> None:
    g.add_edge(
        u,
        v,
        relation=relation,
        edge_class="scientific_confirmed",
        exploration_cost=1.0,
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


def _selector_graph(*, second_analyte: bool) -> nx.DiGraph:
    g = nx.DiGraph()
    nodes = [
        ("SRC", "PlasmonicSubstrate", "Au-Ag substrate"),
        ("AN1", "Analyte", "R6G"),
        ("A1", "Nanostructure", "entry anchor"),
        ("A2", "Nanostructure", "exit anchor"),
        ("OPT", "OpticalCondition", "785 nm excitation"),
        ("TGT", "RamanReporter", "SERS response"),
        ("C", "BridgeConcept", "candidate bridge"),
    ]
    if second_analyte:
        nodes.append(("AN2", "Analyte", "crystal violet"))

    for node, typ, label in nodes:
        attrs = {"type": typ, "label": label}
        if node == "C":
            attrs.update(policy_lane="semantic_candidate", requires_verification=True)
        g.add_node(node, **attrs)

    _confirmed(g, "SRC", "AN1", "TESTED_IN")
    if second_analyte:
        _confirmed(g, "AN1", "AN2")
        _confirmed(g, "AN2", "A1")
    else:
        _confirmed(g, "AN1", "A1")

    _candidate_pair(g, "A1", "C")
    _candidate_pair(g, "A2", "C")
    _confirmed(g, "A2", "OPT", "TESTED_IN")
    _confirmed(g, "OPT", "TGT", "MEASURED_FOR")
    return g


def _best_route(g: nx.DiGraph):
    units = CandidateUnitBuilder(g).build(bridge_capable_only=True)
    selector = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        policy=CandidateUnitSelectionPolicy(max_depth=9, top_k=2),
        unit_relevance={"C": 0.8},
        domain_profile=get_domain_profile("sers_au_ag"),
    )
    routes = selector.enumerate_routes(
        units,
        [{"node_id": "SRC", "semantic_similarity": 0.8}],
        [{"node_id": "TGT", "semantic_similarity": 0.8}],
    )
    assert routes
    return routes[0]


def test_complementary_sers_context_dimensions_do_not_count_as_switch():
    route = _best_route(_selector_graph(second_analyte=False))
    assert set(route.context_node_labels) == {
        "R6G",
        "785 nm excitation",
        "SERS response",
    }
    assert route.score.context_switch_penalty == 0.0


def test_two_values_within_same_sers_context_type_are_penalized():
    route = _best_route(_selector_graph(second_analyte=True))
    assert {"R6G", "crystal violet"} <= set(route.context_node_labels)
    assert route.score.context_switch_penalty == 0.5
    assert route.score.reaction_switch_penalty == 0.5
