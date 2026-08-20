from __future__ import annotations

import networkx as nx

from dac_her.domains.graph_registry import get_graph_adapter
from pipeline_core.corpus.graph_semantics import (
    SERS_GRAPH_DIAGNOSTICS_VERSION,
    integration_component_diagnostics,
    node_role_diagnostics,
)


def _adapter():
    return get_graph_adapter("sers_au_ag")


def test_alpha4a5_diagnostics_version():
    assert SERS_GRAPH_DIAGNOSTICS_VERSION.startswith("sers-alpha4a.5")


def test_support_can_be_tested_in_experiment():
    graph = nx.MultiDiGraph()
    graph.add_node("glass", type="Support", label="Blank glass substrate")
    graph.add_node("exp", type="Experiment", label="Normal Raman measurement")
    graph.add_edge("glass", "exp", key="e1", relation="TESTED_IN")
    assert _adapter().diagnose_relation_contracts(graph) == []


def test_raman_measurement_is_not_misclassified_as_calculation():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "exp_mb",
        type="Experiment",
        label="Methylene blue SERRS concentration and single-molecule test",
        experiment_family="spectroscopy",
        description=(
            "SERRS spectra and enhancement-factor calculation were reported "
            "from concentration-dependent measurements."
        ),
    )
    graph.add_node(
        "exp_blank",
        type="Experiment",
        label="Normal Raman measurement of R6G on blank glass",
        experiment_family="spectroscopy",
        description="Normal Raman spectrum measured on blank glass.",
    )

    rows = node_role_diagnostics(graph, domain_profile_id="sers_au_ag")
    assert rows == []


def test_explicit_computational_method_still_flags_experiment_role():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "bad_fdtd",
        type="Experiment",
        label="FDTD simulation of local electromagnetic field",
        experiment_family="other",
    )
    rows = node_role_diagnostics(graph, domain_profile_id="sers_au_ag")
    assert len(rows) == 1
    assert rows[0]["code"] == "calculation_encoded_as_experiment"


def test_simulated_by_target_encoded_as_experiment_is_flagged():
    graph = nx.MultiDiGraph()
    graph.add_node("nano", type="Nanostructure", label="Au-Ag dimer")
    graph.add_node(
        "bad_calc",
        type="Experiment",
        label="Local field spectra",
        experiment_family="other",
    )
    graph.add_edge("nano", "bad_calc", key="e1", relation="SIMULATED_BY")

    rows = node_role_diagnostics(graph, domain_profile_id="sers_au_ag")
    assert len(rows) == 1
    assert rows[0]["code"] == "calculation_encoded_as_experiment"


def test_missing_subject_anchor_component_has_no_studies_bridge():
    graph = nx.MultiDiGraph()
    graph.add_node("paper", type="Paper", label="Paper")
    graph.add_node("main", type="PlasmonicSubstrate", label="Main substrate")
    graph.add_edge("paper", "main", key="e0", relation="STUDIES")

    graph.add_node("calc", type="Calculation", label="FEM local-field spectra")
    graph.add_node(
        "condition",
        type="OpticalCondition",
        label="Air medium with E-field parallel to dimer axis",
    )
    graph.add_edge(
        "calc",
        "condition",
        key="e1",
        relation="USES_OPTICAL_CONDITION",
    )

    components, bridges = integration_component_diagnostics(
        graph,
        domain_profile_id="sers_au_ag",
    )
    assert len(components) == 1
    assert components[0]["component_subtype"] == "missing_subject_anchor"
    assert components[0]["severity"] == "review"
    assert bridges == []


def test_reporter_only_component_is_context_not_bridge_candidate():
    graph = nx.MultiDiGraph()
    graph.add_node("paper", type="Paper", label="Paper")
    graph.add_node("main", type="PlasmonicSubstrate", label="Main substrate")
    graph.add_edge("paper", "main", key="e0", relation="STUDIES")
    graph.add_node(
        "reporter",
        type="RamanReporter",
        label="Raman probe molecules",
    )

    components, bridges = integration_component_diagnostics(
        graph,
        domain_profile_id="sers_au_ag",
    )
    assert len(components) == 1
    assert components[0]["component_subtype"] == "isolated_context_entity"
    assert components[0]["severity"] == "info"
    assert bridges == []


def test_nanostructure_island_gets_review_only_studies_candidate():
    graph = nx.MultiDiGraph()
    graph.add_node("paper", type="Paper", label="Paper")
    graph.add_node("main", type="PlasmonicSubstrate", label="Main substrate")
    graph.add_edge("paper", "main", key="e0", relation="STUDIES")

    graph.add_node("au_cube", type="Nanostructure", label="Au nanocube")
    graph.add_node("tem", type="Experiment", label="TEM of Au nanocubes")
    graph.add_node("size", type="Measurement", label="Au cube size 55 nm")
    graph.add_edge("au_cube", "tem", key="e1", relation="CHARACTERIZED_IN")
    graph.add_edge("tem", "size", key="e2", relation="HAS_MEASUREMENT")
    graph.add_edge("size", "au_cube", key="e3", relation="MEASURED_FOR")

    components, bridges = integration_component_diagnostics(
        graph,
        domain_profile_id="sers_au_ag",
    )
    assert len(components) == 1
    assert components[0]["component_subtype"] == "scientific_subject_island"
    assert components[0]["severity"] == "review"
    assert len(bridges) == 1
    assert bridges[0]["target_subject_id"] == "au_cube"
    assert bridges[0]["target_subject_type"] == "Nanostructure"
    assert bridges[0]["suggested_relation"] == "STUDIES"
    assert bridges[0]["auto_apply"] is False


def test_blank_glass_control_component_is_not_bridge_candidate():
    graph = nx.MultiDiGraph()
    graph.add_node("paper", type="Paper", label="Paper")
    graph.add_node("main", type="PlasmonicSubstrate", label="Au@Ag nanocube")
    graph.add_edge("paper", "main", key="e0", relation="STUDIES")

    graph.add_node(
        "glass",
        type="Support",
        label="Blank glass substrate",
    )
    graph.add_node("r6g", type="RamanReporter", label="R6G")
    graph.add_node(
        "raman",
        type="Experiment",
        label="Normal Raman measurement of R6G on blank glass",
        experiment_family="spectroscopy",
    )
    graph.add_edge("glass", "raman", key="e1", relation="TESTED_IN")
    graph.add_edge("raman", "r6g", key="e2", relation="USES_REPORTER")

    components, bridges = integration_component_diagnostics(
        graph,
        domain_profile_id="sers_au_ag",
    )
    assert len(components) == 1
    assert components[0]["component_subtype"] == "reference_control_component"
    assert components[0]["severity"] == "info"
    assert bridges == []


def test_bridge_targets_are_restricted_to_core_scientific_subjects():
    graph = nx.MultiDiGraph()
    graph.add_node("paper", type="Paper", label="Paper")
    graph.add_node("main", type="PlasmonicSubstrate", label="Main")
    graph.add_edge("paper", "main", key="e0", relation="STUDIES")

    graph.add_node("nano", type="Nanostructure", label="Au nanocube")
    graph.add_node("support", type="Support", label="glass support")
    graph.add_node("reporter", type="RamanReporter", label="R6G")
    graph.add_node("condition", type="OpticalCondition", label="633 nm")
    graph.add_edge("nano", "support", key="e1", relation="HAS_SUPPORT")
    graph.add_edge("nano", "reporter", key="e2", relation="HAS_COMPONENT")
    graph.add_edge("nano", "condition", key="e3", relation="HAS_COMPONENT")

    _, bridges = integration_component_diagnostics(
        graph,
        domain_profile_id="sers_au_ag",
    )
    assert bridges
    assert {
        row["target_subject_type"]
        for row in bridges
    } <= {"PlasmonicSubstrate", "Nanostructure"}
    assert all(row["auto_apply"] is False for row in bridges)
