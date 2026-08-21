from __future__ import annotations
import networkx as nx
from pipeline_core.discovery.endpoint_selection import EndpointPairSelector, _paper_id_for_node


def graph():
    g = nx.DiGraph()
    ss = [f"paper::S{i}::s" for i in (1,2,3)]
    ts = [f"paper::T{i}::t" for i in (1,2,3)]
    for n in ss + ts: g.add_node(n)
    for s in ss:
        for t in ts: g.add_edge(s, t, exploration_cost=1.0)
    return g


def matches():
    ss = [{"node_id": f"paper::S{i}::s", "semantic_similarity": v} for i,v in [(1,.900),(2,.899),(3,.898)]]
    ts = [{"node_id": f"paper::T{i}::t", "semantic_similarity": v} for i,v in [(1,.900),(2,.899),(3,.898)]]
    return ss, ts


def pairs(rows): return [(x.source_paper_id, x.target_paper_id) for x in rows]


def test_attribute_and_node_id_paper_provenance():
    g = nx.DiGraph(); g.add_node("custom", source_paper_id="P0"); g.add_node("paper::P1::x")
    assert _paper_id_for_node(g, "custom") == "P0"
    assert _paper_id_for_node(g, "paper::P1::x") == "P1"


def test_bounded_diversity_spreads_endpoint_papers():
    g = graph(); ss, ts = matches()
    selected, _ = EndpointPairSelector(g, paper_novelty_bonus=.01).select(ss, ts, top_k=3, max_depth=2)
    assert pairs(selected) == [("S1","T1"),("S2","T2"),("S3","T3")]
    assert [x.selection_rank for x in selected] == [1,2,3]


def test_zero_bonus_recovers_relevance_first_selection():
    g = graph(); ss, ts = matches()
    selected, _ = EndpointPairSelector(g, paper_novelty_bonus=0).select(ss, ts, top_k=3, max_depth=2)
    assert pairs(selected) == [("S1","T1"),("S1","T2"),("S2","T1")]


def test_semantic_tier_is_hard_priority():
    g = nx.DiGraph(); g.add_edge("paper::A::s", "paper::X::t", exploration_cost=1.0); g.add_edge("paper::B::s", "paper::Y::t", exploration_cost=1.0)
    ss = [{"node_id":"paper::A::s","semantic_similarity":.70,"exact_label_match":True}, {"node_id":"paper::B::s","semantic_similarity":.99}]
    ts = [{"node_id":"paper::X::t","semantic_similarity":.70,"exact_label_match":True}, {"node_id":"paper::Y::t","semantic_similarity":.99}]
    selected, _ = EndpointPairSelector(g, paper_novelty_bonus=1.0).select(ss, ts, top_k=1, max_depth=2)
    assert pairs(selected) == [("A","X")]
    assert selected[0].semantic_tier == 0
