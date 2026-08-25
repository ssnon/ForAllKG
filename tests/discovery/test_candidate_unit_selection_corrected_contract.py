from __future__ import annotations

import networkx as nx

from domains.dac_her.profile import DAC_HER_PROFILE

from pipeline_core.discovery.candidate_unit_selection import (
    CandidateUnitSelectionPolicy,
    CandidateUnitSelector,
    _bounded_shortest_paths,
)

from pipeline_core.discovery.candidate_units import (
    CandidateUnitBuilder,
    confirmed_navigation_graph,
)


def _confirmed(
    g: nx.DiGraph,
    u: str,
    v: str,
    relation: str = "APPLIES_TO",
    cost: float = 1.0,
) -> None:
    g.add_edge(
        u,
        v,
        relation=relation,
        edge_class="scientific_confirmed",
        exploration_cost=cost,
        requires_verification=False,
        reverse_navigation=False,
    )


def _candidate_pair(
    g: nx.DiGraph,
    anchor: str,
    candidate: str,
) -> None:
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


def test_corrected_state_retains_costlier_mechanism_same_hop() -> None:
    g = nx.DiGraph()

    g.add_node("S", type="Material")
    g.add_node("X", type="Material")
    g.add_node("M", type="MechanismClaim")
    g.add_node("A", type="CatalystModel")

    _confirmed(g, "S", "X", cost=0.5)
    _confirmed(g, "X", "A", cost=0.5)

    _confirmed(g, "S", "M", cost=1.5)
    _confirmed(g, "M", "A", cost=1.5)

    legacy = _bounded_shortest_paths(
        g,
        "S",
        max_hops=2,
    )

    corrected = _bounded_shortest_paths(
        g,
        "S",
        max_hops=2,
        discovery_semantics=
            DAC_HER_PROFILE.discovery,
        retain_semantic_state=True,
    )

    legacy_paths = {
        path
        for _, path
        in legacy["A"]
    }

    corrected_paths = {
        path
        for _, path
        in corrected["A"]
    }

    assert legacy_paths == {
        ("S", "X", "A")
    }

    assert corrected_paths == {
        ("S", "X", "A"),
        ("S", "M", "A"),
    }


def _carrier_reuse_graph() -> nx.DiGraph:
    g = nx.DiGraph()

    g.add_node(
        "S",
        type="Material",
        label="source",
    )

    g.add_node(
        "H",
        type="CorpusAlignment",
        label="routing hub",
        graph_layer="corpus_alignment",
        corpus_node_kind="alignment_hub",
        source_paper_ids_json='["HUB_A","HUB_B","HUB_C"]',
    )

    g.add_node(
        "A",
        type="CatalystModel",
        label="entry",
        source_paper_id="P1",
    )

    g.add_node(
        "B",
        type="CatalystModel",
        label="exit",
        source_paper_id="P2",
    )

    g.add_node(
        "T",
        type="Reaction",
        label="target",
    )

    g.add_node(
        "C",
        type="BridgeConcept",
        label="candidate",
        policy_lane="semantic_candidate",
        requires_verification=True,
    )

    _confirmed(g, "S", "H")
    _confirmed(g, "H", "A")
    _confirmed(g, "B", "H")
    _confirmed(g, "H", "T")

    _candidate_pair(
        g,
        "A",
        "C",
    )

    _candidate_pair(
        g,
        "B",
        "C",
    )

    return g


def test_corrected_contract_allows_only_narrow_carrier_reuse() -> None:
    g = _carrier_reuse_graph()

    units = CandidateUnitBuilder(
        g
    ).build(
        bridge_capable_only=True
    )

    unit = units[0]

    legacy = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        domain_profile=DAC_HER_PROFILE,
        policy=CandidateUnitSelectionPolicy(
            max_depth=6,
            top_k=10,
            corrected_route_contract=False,
        ),
        unit_relevance={
            "C": 0.5
        },
    )

    corrected = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        domain_profile=DAC_HER_PROFILE,
        policy=CandidateUnitSelectionPolicy(
            max_depth=6,
            top_k=10,
            corrected_route_contract=True,
        ),
        unit_relevance={
            "C": 0.5
        },
    )

    source = [
        {
            "node_id": "S",
            "semantic_similarity": 0.8,
        }
    ]

    target = [
        {
            "node_id": "T",
            "semantic_similarity": 0.8,
        }
    ]

    legacy_routes = legacy.enumerate_routes(
        units,
        source,
        target,
    )

    corrected_routes = (
        corrected.enumerate_routes(
            units,
            source,
            target,
        )
    )

    assert legacy_routes == []

    assert corrected_routes

    route = corrected_routes[0]

    assert route.nodes == (
        "S",
        "H",
        "A",
        "C",
        "B",
        "H",
        "T",
    )

    # The corpus hub is reusable.
    assert corrected._route_reuse_allowed(
        route.nodes,
        unit,
    )

    # Candidate anchors remain forbidden even if a future domain
    # were to classify one as a reusable carrier.
    assert not corrected._route_reuse_allowed(
        (
            "S",
            "A",
            "C",
            "B",
            "A",
            "T",
        ),
        unit,
    )


def test_corrected_contract_excludes_alignment_scope_from_provenance() -> None:
    g = _carrier_reuse_graph()

    units = CandidateUnitBuilder(
        g
    ).build(
        bridge_capable_only=True
    )

    corrected = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        domain_profile=DAC_HER_PROFILE,
        policy=CandidateUnitSelectionPolicy(
            max_depth=6,
            corrected_route_contract=True,
        ),
        unit_relevance={
            "C": 0.5
        },
    )

    routes = corrected.enumerate_routes(
        units,
        [
            {
                "node_id": "S",
                "semantic_similarity": 0.8,
            }
        ],
        [
            {
                "node_id": "T",
                "semantic_similarity": 0.8,
            }
        ],
    )

    assert routes

    papers = set(
        routes[0].visited_paper_ids
    )

    assert "P1" in papers
    assert "P2" in papers

    assert "HUB_A" not in papers
    assert "HUB_B" not in papers
    assert "HUB_C" not in papers


def _score_first_graph() -> nx.DiGraph:
    g = nx.DiGraph()

    for node, typ in [
        ("S", "Material"),
        ("X", "Material"),
        ("ML", "MechanismClaim"),
        ("MY", "MechanismClaim"),
        ("A", "CatalystModel"),
        ("B", "CatalystModel"),
        ("MR", "MechanismClaim"),
        ("T", "Reaction"),
        ("C", "BridgeConcept"),
    ]:
        attrs = {
            "type": typ,
            "label": node,
        }

        if node == "C":
            attrs.update(
                policy_lane="semantic_candidate",
                requires_verification=True,
            )

        g.add_node(
            node,
            **attrs,
        )

    # Cheap / scientifically weaker prefix, 2 hops.
    _confirmed(
        g,
        "S",
        "X",
        cost=0.1,
    )

    _confirmed(
        g,
        "X",
        "A",
        cost=0.1,
    )

    # More expensive / mechanism-bearing prefix, 3 hops.
    _confirmed(
        g,
        "S",
        "ML",
        cost=1.0,
    )

    _confirmed(
        g,
        "ML",
        "MY",
        cost=1.0,
    )

    _confirmed(
        g,
        "MY",
        "A",
        cost=1.0,
    )

    _candidate_pair(
        g,
        "A",
        "C",
    )

    _candidate_pair(
        g,
        "B",
        "C",
    )

    # Fixed mechanism-bearing suffix.
    _confirmed(
        g,
        "B",
        "MR",
        cost=1.0,
    )

    _confirmed(
        g,
        "MR",
        "T",
        cost=1.0,
    )

    return g


def test_corrected_contract_scores_before_local_cost_tiebreak() -> None:
    g = _score_first_graph()

    units = CandidateUnitBuilder(
        g
    ).build(
        bridge_capable_only=True
    )

    source = [
        {
            "node_id": "S",
            "semantic_similarity": 0.8,
        }
    ]

    target = [
        {
            "node_id": "T",
            "semantic_similarity": 0.8,
        }
    ]

    legacy = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        domain_profile=DAC_HER_PROFILE,
        policy=CandidateUnitSelectionPolicy(
            max_depth=7,
            top_k=10,
            corrected_route_contract=False,
        ),
        unit_relevance={
            "C": 0.5
        },
    )

    corrected = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        domain_profile=DAC_HER_PROFILE,
        policy=CandidateUnitSelectionPolicy(
            max_depth=7,
            top_k=10,
            corrected_route_contract=True,
        ),
        unit_relevance={
            "C": 0.5
        },
    )

    legacy_routes = legacy.enumerate_routes(
        units,
        source,
        target,
    )

    corrected_routes = (
        corrected.enumerate_routes(
            units,
            source,
            target,
        )
    )

    assert legacy_routes
    assert corrected_routes

    legacy_route = legacy_routes[0]
    corrected_route = corrected_routes[0]

    assert "X" in legacy_route.nodes
    assert "ML" not in legacy_route.nodes

    assert "ML" in corrected_route.nodes
    assert "X" not in corrected_route.nodes

    assert (
        corrected_route.score.mechanistic_continuity
        == 1.0
    )

    assert (
        corrected_route.score.total
        > legacy_route.score.total
    )

    assert (
        corrected_route.total_cost
        > legacy_route.total_cost
    )
