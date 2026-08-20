from __future__ import annotations

import networkx as nx

from pipeline_core.discovery.endpoint_selection import EndpointPairSelector


def test_same_paper_pair_counts_as_one_new_unique_paper():
    graph = nx.DiGraph()
    graph.add_edge(
        "paper::P1::source",
        "paper::P1::target",
        exploration_cost=1.0,
    )

    selected, _ = EndpointPairSelector(
        graph,
        paper_novelty_bonus=0.01,
    ).select(
        [
            {
                "node_id": "paper::P1::source",
                "semantic_similarity": 0.9,
            }
        ],
        [
            {
                "node_id": "paper::P1::target",
                "semantic_similarity": 0.9,
            }
        ],
        top_k=1,
        max_depth=2,
    )

    assert len(selected) == 1
    assert selected[0].source_paper_id == "P1"
    assert selected[0].target_paper_id == "P1"
    assert selected[0].diversity_bonus == 0.01


def test_two_new_cross_papers_beat_near_equal_same_paper_pair():
    graph = nx.DiGraph()

    graph.add_edge(
        "paper::P1::source",
        "paper::P1::target",
        exploration_cost=1.0,
    )
    graph.add_edge(
        "paper::P2::source",
        "paper::P3::target",
        exploration_cost=1.0,
    )

    sources = [
        {
            "node_id": "paper::P1::source",
            "semantic_similarity": 0.900,
        },
        {
            "node_id": "paper::P2::source",
            "semantic_similarity": 0.899,
        },
    ]
    targets = [
        {
            "node_id": "paper::P1::target",
            "semantic_similarity": 0.900,
        },
        {
            "node_id": "paper::P3::target",
            "semantic_similarity": 0.899,
        },
    ]

    selected, _ = EndpointPairSelector(
        graph,
        paper_novelty_bonus=0.01,
    ).select(
        sources,
        targets,
        top_k=1,
        max_depth=2,
    )

    assert len(selected) == 1
    assert selected[0].source_paper_id == "P2"
    assert selected[0].target_paper_id == "P3"
    assert selected[0].diversity_bonus == 0.02
