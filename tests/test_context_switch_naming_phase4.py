from __future__ import annotations

import networkx as nx

from dac_her.candidate_unit_selection import (
    CandidateUnitScore,
    CandidateUnitSelectionPolicy,
    CandidateUnitSelector,
)
from pipeline_core.discovery.candidate_units import CandidateUnitBuilder, confirmed_navigation_graph
from dac_her.discovery_bundle import DiscoveryBundleBuilder, DiscoveryPolicy
from pipeline_core.discovery.discovery_contracts import DiscoveryScoreBreakdown
from dac_her.domains import get_domain_profile


def _confirmed(g: nx.DiGraph, u: str, v: str, relation: str = "INTERPRETED_AS") -> None:
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


def _sers_context_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for node, typ, label in [
        ("SRC", "PlasmonicSubstrate", "Au-Ag substrate"),
        ("A1", "Nanostructure", "entry anchor"),
        ("AN", "Analyte", "R6G"),
        ("A2", "Nanostructure", "exit anchor"),
        ("OPT", "OpticalCondition", "785 nm excitation"),
        ("TGT", "RamanReporter", "SERS response"),
        ("C", "BridgeConcept", "candidate bridge"),
    ]:
        attrs = {"type": typ, "label": label}
        if node == "C":
            attrs.update(policy_lane="semantic_candidate", requires_verification=True)
        g.add_node(node, **attrs)

    _confirmed(g, "SRC", "AN", "TESTED_IN")
    _confirmed(g, "AN", "A1", "APPLIES_TO")
    _candidate_pair(g, "A1", "C")
    _candidate_pair(g, "A2", "C")
    _confirmed(g, "A2", "OPT", "TESTED_IN")
    _confirmed(g, "OPT", "TGT", "MEASURED_FOR")
    return g


def test_candidate_score_writes_canonical_and_legacy_alias():
    score = CandidateUnitScore(
        endpoint_relevance=0.5,
        unit_relevance=0.5,
        mechanistic_continuity=0.5,
        scientific_content_density=0.5,
        cross_paper_span=0.0,
        generic_entity_penalty=0.0,
        alignment_penalty=0.0,
        reverse_penalty=0.0,
        context_switch_penalty=0.5,
        path_length_penalty=0.2,
        total=0.3,
    )
    payload = score.to_dict()
    assert score.context_switch_penalty == 0.5
    assert score.reaction_switch_penalty == 0.5
    assert payload["context_switch_penalty"] == 0.5
    assert payload["reaction_switch_penalty"] == 0.5


def test_candidate_policy_exposes_legacy_weight_alias():
    policy = CandidateUnitSelectionPolicy(context_switch_penalty_weight=0.23)
    assert policy.context_switch_penalty_weight == 0.23
    assert policy.reaction_switch_penalty_weight == 0.23
    assert policy.to_dict()["reaction_switch_penalty_weight"] == 0.23


def test_sers_selector_uses_profile_context_nodes_not_reaction_semantics():
    g = _sers_context_graph()
    units = CandidateUnitBuilder(g).build(bridge_capable_only=True)
    selector = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        policy=CandidateUnitSelectionPolicy(max_depth=8, top_k=2),
        unit_relevance={"C": 0.8},
        domain_profile=get_domain_profile("sers_au_ag"),
    )
    routes = selector.enumerate_routes(
        units,
        [{"node_id": "SRC", "semantic_similarity": 0.8}],
        [{"node_id": "TGT", "semantic_similarity": 0.8}],
    )
    assert routes
    route = routes[0]
    assert set(route.context_node_labels) == {
        "785 nm excitation",
        "R6G",
        "SERS response",
    }
    # These are complementary context dimensions, not a domain/context switch.
    assert route.score.context_switch_penalty == 0.0
    payload = route.to_dict()
    assert payload["context_node_labels"] == payload["reaction_node_labels"]
    assert payload["candidate_unit_selection"]["context_switch_penalty"] == (
        payload["candidate_unit_selection"]["reaction_switch_penalty"]
    )


def _candidate_discovery_payload(*, canonical: bool) -> tuple[dict, nx.DiGraph]:
    g = nx.DiGraph()
    g.add_node("a", type="Mechanism", label="a", source_paper_id="P1")
    g.add_node("b", type="Mechanism", label="b", source_paper_id="P1")
    g.add_edge("a", "b", relation="MODULATES", edge_class="scientific", exploration_cost=1.0)
    selection = {"total": 0.7, "unit_relevance": 0.8}
    selection[
        "context_switch_penalty" if canonical else "reaction_switch_penalty"
    ] = 0.5
    path = {
        "path_id": "candidate",
        "nodes": ["a", "b"],
        "steps": [
            {
                "source": "a",
                "target": "b",
                "relation": "MODULATES",
                "edge_class": "scientific",
                "navigation_edge_id": "e1",
                "selected_original_edge_id": "e1",
                "requires_verification": True,
            }
        ],
        "visited_paper_ids": ["P1"],
        "candidate_unit_count": 1,
        "candidate_unit": {
            "unit_id": "unit:1",
            "label": "candidate",
            "entry_anchor_id": "a",
            "entry_anchor_label": "a",
            "exit_anchor_id": "b",
            "exit_anchor_label": "b",
            "proposed_subject": "a",
            "proposed_relation": "MODULATES",
            "proposed_object": "b",
        },
        "candidate_unit_selection": selection,
        "path_quality": {
            "path_type": "CANDIDATE_EXPLORATION",
            "mechanistic_content": "high",
            "mechanistic_edge_density": 1.0,
            "mechanistic_node_density": 1.0,
            "navigation_edge_fraction": 0.0,
            "reverse_fraction": 0.0,
            "candidate_fraction": 1.0,
            "endpoint_pair_score": 0.8,
        },
    }
    payload = {
        "corpus_id": "c",
        "domain_profile_id": "sers_au_ag",
        "mode": "exploratory",
        "source_query": "Au-Ag",
        "target_query": "SERS",
        "semantic_stop_query": None,
        "paths": [],
        "candidate_paths": [path],
    }
    return payload, g


def test_discovery_bundle_prefers_canonical_but_loads_legacy_candidate_penalty():
    profile = get_domain_profile("sers_au_ag")
    values = []
    for canonical in (True, False):
        payload, graph = _candidate_discovery_payload(canonical=canonical)
        bundle = DiscoveryBundleBuilder(
            DiscoveryPolicy(
                top_k=1,
                semantic_diversity_enabled=False,
                min_exploration_score=0.0,
                min_reserved_candidate_unit_score=0.0,
            ),
            domain_profile=profile,
        ).build([("candidate.json", payload, graph)])
        assert bundle.inspirations
        inspiration = bundle.inspirations[0]
        assert inspiration.context_switch_penalty == 0.5
        assert inspiration.reaction_domain_switch_penalty == 0.5
        assert "scientific_context_switch" in inspiration.reason_codes
        assert "reaction_domain_switch" not in inspiration.reason_codes
        values.append(inspiration.score_breakdown.context_switch_penalty)
    assert values == [0.5, 0.5]


def test_discovery_contract_accepts_legacy_only_penalty():
    row = DiscoveryScoreBreakdown.model_validate(
        {
            "endpoint_relevance": 0.1,
            "mechanistic_content": 0.1,
            "cross_paper_span": 0.0,
            "community_span": 0.0,
            "relation_rarity": 0.0,
            "exploratory_mode_bonus": 0.0,
            "grounding_redundancy_penalty": 0.0,
            "navigation_burden_penalty": 0.0,
            "reverse_burden_penalty": 0.0,
            "reaction_domain_switch_penalty": 0.4,
            "total": 0.1,
        }
    )
    assert row.context_switch_penalty == 0.4
    assert row.reaction_domain_switch_penalty == 0.4
