from __future__ import annotations

import networkx as nx

from pipeline_core.corpus.graph_semantics import (
    SERS_GRAPH_DIAGNOSTICS_VERSION,
    node_role_diagnostics,
)


def test_alpha4a5_2_1_diagnostics_version():
    assert SERS_GRAPH_DIAGNOSTICS_VERSION == "sers-alpha4a.5.2"


def test_femtomolar_does_not_trigger_fem_marker():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "exp_sers_mb",
        type="Experiment",
        label="Methylene blue SERRS concentration and single-molecule test",
        experiment_type="sers_spectroscopy",
        experiment_family="spectroscopy",
        method_label="Surface-enhanced Raman spectroscopy",
        raw_method_name="SERRS",
        description=(
            "SERRS spectra of MB solutions dispersed over the SERS substrate "
            "by drop-casting, including femtomolar concentration and normal "
            "Raman comparison."
        ),
    )
    assert node_role_diagnostics(
        graph,
        domain_profile_id="sers_au_ag",
    ) == []


def test_explicit_fem_still_triggers():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "bad",
        type="Experiment",
        label="FEM local-field simulation",
        experiment_type="other",
        experiment_family="other",
        method_label="FEM",
    )
    rows = node_role_diagnostics(graph, domain_profile_id="sers_au_ag")
    assert len(rows) == 1
    assert rows[0]["code"] == "calculation_encoded_as_experiment"
