from __future__ import annotations

import networkx as nx

from dac_her.domains.extraction_registry import get_extraction_adapter
from dac_her.domains.graph_registry import get_graph_adapter
from pipeline_core.corpus.graph_semantics import node_role_diagnostics


def test_sers_uses_material_is_official_relation():
    adapter = get_extraction_adapter("sers_au_ag")
    assert "USES_MATERIAL" in adapter.allowed_relation_types


def test_sers_synthesis_method_can_use_material_and_reporter():
    graph = nx.MultiDiGraph()
    graph.add_node("method", type="SynthesisMethod", label="Ag shell growth")
    graph.add_node("material", type="Material", label="PVP")
    graph.add_node("reporter", type="RamanReporter", label="4-ATP")
    graph.add_edge("method", "material", key="e1", relation="USES_MATERIAL")
    graph.add_edge("method", "reporter", key="e2", relation="USES_REPORTER")

    assert get_graph_adapter("sers_au_ag").diagnose_relation_contracts(graph) == []


def test_sers_role_diagnostics_flag_calculation_encoded_as_experiment():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "calc_dda",
        type="Experiment",
        label="DDA electric-field distribution calculation",
        experiment_type="unregistered_discrete_dipole_approximation_calculation",
        experiment_family="other",
        method_label="Discrete dipole approximation calculation",
    )
    issues = node_role_diagnostics(graph, domain_profile_id="sers_au_ag")
    assert any(
        issue["code"] == "calculation_encoded_as_experiment"
        for issue in issues
    )


def test_sers_role_diagnostics_flag_synthesis_encoded_as_experiment():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "exp_synthesis",
        type="Experiment",
        label="Synthesis of SiO2@Au@Ag nanoparticles",
        experiment_type="synthesis_procedure",
        experiment_family="synthesis",
        method_label="Synthesis procedure",
    )
    issues = node_role_diagnostics(graph, domain_profile_id="sers_au_ag")
    assert any(
        issue["code"] == "synthesis_encoded_as_experiment"
        for issue in issues
    )


def test_sers_role_diagnostics_do_not_flag_normal_measurement_experiment():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "exp_sers",
        type="Experiment",
        label="SERS measurement",
        experiment_type="sers_spectroscopy",
        experiment_family="spectroscopy",
        method_label="SERS spectroscopy",
    )
    assert node_role_diagnostics(
        graph,
        domain_profile_id="sers_au_ag",
    ) == []
