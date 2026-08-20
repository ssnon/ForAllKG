from __future__ import annotations

import networkx as nx

from domains.graph_registry import get_graph_adapter


def _issues(graph: nx.MultiDiGraph):
    return get_graph_adapter("sers_au_ag").diagnose_relation_contracts(graph)


def test_support_can_be_prepared_by_synthesis_method():
    graph = nx.MultiDiGraph()
    graph.add_node("support", type="Support", label="silica support")
    graph.add_node("method", type="SynthesisMethod", label="Stober silica synthesis")
    graph.add_edge("support", "method", key="e1", relation="PREPARED_BY")
    assert _issues(graph) == []


def test_support_can_have_material_component():
    graph = nx.MultiDiGraph()
    graph.add_node("support", type="Support", label="silica nanoparticle support")
    graph.add_node("material", type="Material", label="silica")
    graph.add_edge("support", "material", key="e1", relation="HAS_COMPONENT")
    assert _issues(graph) == []


def test_support_cannot_use_material_as_synthesis_input():
    graph = nx.MultiDiGraph()
    graph.add_node("support", type="Support", label="silica support")
    graph.add_node("material", type="Material", label="silica")
    graph.add_edge("support", "material", key="e1", relation="USES_MATERIAL")
    issues = _issues(graph)
    assert len(issues) == 1
    assert issues[0].code == "relation_source_type_mismatch"
    assert issues[0].relation == "USES_MATERIAL"


def test_reporter_is_not_structural_component():
    graph = nx.MultiDiGraph()
    graph.add_node("substrate", type="PlasmonicSubstrate", label="SiO2@Au@Ag")
    graph.add_node("reporter", type="RamanReporter", label="4-ATP")
    graph.add_edge("substrate", "reporter", key="e1", relation="HAS_COMPONENT")
    issues = _issues(graph)
    assert len(issues) == 1
    assert issues[0].code == "relation_target_type_mismatch"
    assert issues[0].relation == "HAS_COMPONENT"


def test_reporter_use_from_synthesis_method_remains_valid():
    graph = nx.MultiDiGraph()
    graph.add_node("method", type="SynthesisMethod", label="4-ATP loading")
    graph.add_node("reporter", type="RamanReporter", label="4-ATP")
    graph.add_edge("method", "reporter", key="e1", relation="USES_REPORTER")
    assert _issues(graph) == []
