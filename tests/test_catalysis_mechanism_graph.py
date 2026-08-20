from __future__ import annotations

import networkx as nx

from domains.catalysis_mechanism.graph import (
    BROAD_DIRECT_MECHANISM_RELATIONS,
    BROAD_MECHANISM_NODE_TYPES,
    CATALYSIS_MECHANISM_GRAPH_ADAPTER,
)


def test_broad_graph_adapter_preserves_strict_roles():
    graph = nx.MultiDiGraph()
    graph.add_node("env", type="InterfacialEnvironment")
    normalized, adjustments = CATALYSIS_MECHANISM_GRAPH_ADAPTER.normalize_semantic_roles(
        graph,
        chunk_id="abstract:0",
    )
    assert normalized is graph
    assert adjustments == []


def test_broad_graph_adapter_accepts_direct_mechanism_contracts():
    graph = nx.MultiDiGraph()
    graph.add_node("water", type="InterfacialEnvironment")
    graph.add_node("barrier", type="MechanisticFactor")
    graph.add_node("state_a", type="StructuralState")
    graph.add_node("state_b", type="StructuralState")
    graph.add_node("coverage", type="AdsorbateState")
    graph.add_node("volmer", type="ReactionStep")
    graph.add_edge("water", "barrier", relation="MODULATES")
    graph.add_edge("state_a", "state_b", relation="RECONSTRUCTS_TO")
    graph.add_edge("coverage", "volmer", relation="CHANGES_RDS")

    assert CATALYSIS_MECHANISM_GRAPH_ADAPTER.diagnose_relation_contracts(graph) == []
    assert "MODULATES" in BROAD_DIRECT_MECHANISM_RELATIONS
    assert "InterfacialEnvironment" in BROAD_MECHANISM_NODE_TYPES


def test_broad_graph_adapter_flags_wrong_rds_target_type():
    graph = nx.MultiDiGraph()
    graph.add_node("coverage", type="AdsorbateState")
    graph.add_node("wrong", type="Descriptor")
    graph.add_edge("coverage", "wrong", relation="CHANGES_RDS")

    issues = CATALYSIS_MECHANISM_GRAPH_ADAPTER.diagnose_relation_contracts(graph)
    assert len(issues) == 1
    assert issues[0].code == "relation_target_type_mismatch"
