from __future__ import annotations

import json

import networkx as nx

from dac_her.broad_corpus_audit import audit_broad_corpus


def test_broad_audit_counts_recurrent_mechanism_signatures_across_papers():
    graph = nx.MultiDiGraph()
    for paper_id, suffix in (("P1", "1"), ("P2", "2")):
        left = f"paper::{paper_id}::env{suffix}"
        right = f"paper::{paper_id}::barrier{suffix}"
        graph.add_node(left, type="InterfacialEnvironment", source_paper_id=paper_id)
        graph.add_node(right, type="MechanisticFactor", source_paper_id=paper_id)
        graph.add_edge(
            left,
            right,
            relation="MODULATES",
            graph_layer="broad_mechanism_abstract",
            requires_verification=True,
            source_paper_id=paper_id,
            source_paper_ids_json=json.dumps([paper_id]),
        )

    report, signatures = audit_broad_corpus(
        graph,
        expected_paper_ids=["P1", "P2"],
    )
    assert report["mechanism_bearing_paper_fraction"] == 1.0
    assert report["direct_mechanism_edges"] == 2
    assert report["unique_mechanism_signatures"] == 1
    assert report["recurring_mechanism_signatures"] == 1
    assert signatures[0].paper_support == 2
    assert signatures[0].relation == "MODULATES"
