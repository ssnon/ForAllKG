from __future__ import annotations

import json

import networkx as nx

from dac_her.domains.sers_au_ag_comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER,
)


def _graph(
    *,
    description: str = "",
    raw_method_name: str = "",
    source_expression: str = "SERS measurement.",
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "sub",
        type="PlasmonicSubstrate",
        label="DIP substrate",
    )
    graph.add_node(
        "exp",
        type="Experiment",
        label="SERS experiment",
        experiment_type="sers_spectroscopy",
        method_label="Surface-enhanced Raman spectroscopy",
        conditions_json=json.dumps([]),
        description=description,
        raw_method_name=raw_method_name,
    )
    graph.add_node(
        "m",
        type="Measurement",
        label="SERS enhancement",
        metric_id="sers_enhancement_factor",
        metric="SERS enhancement factor",
        value_numeric="1000",
        value_text="",
        unit="",
        source_expression=source_expression,
    )
    graph.add_edge("sub", "exp", relation="TESTED_IN")
    graph.add_edge("exp", "m", relation="HAS_MEASUREMENT")
    graph.add_edge("m", "sub", relation="MEASURED_FOR")
    return graph


def _environment(graph: nx.MultiDiGraph):
    method = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        graph,
        "P1",
    )[0]
    return method.dimension_map["measurement_environment"]


def test_alpha4b3b321_multitask_umbrella_does_not_leak_cellular_environment():
    environment = _environment(_graph(
        description=(
            "Experimental evaluation of DIP SERS enhancement, "
            "polarization dependence, DNA detection, and cell "
            "imaging performance."
        ),
        source_expression="The SERS enhancement factor was estimated.",
    ))
    assert environment.status == "unknown"


def test_alpha4b3b321_measurement_local_cell_source_remains_cellular():
    environment = _environment(_graph(
        description=(
            "Experimental evaluation of DIP SERS enhancement, "
            "polarization dependence, DNA detection, and cell "
            "imaging performance."
        ),
        source_expression=(
            "SERS maps of U87MG cells showed strong signal."
        ),
    ))
    assert environment.status == "known"
    assert environment.normalized_value == "cellular"
    assert "measurement_source_expression" in environment.provenance_scopes


def test_alpha4b3b321_direct_experiment_cell_measurement_remains_cellular():
    environment = _environment(_graph(
        description=(
            "Time-dependent Raman profiles were recorded from "
            "cRGD-functionalized DIPs in a cell."
        ),
    ))
    assert environment.status == "known"
    assert environment.normalized_value == "cellular"
    assert "experiment_method_text" in environment.provenance_scopes


def test_alpha4b3b321_short_cell_imaging_method_label_remains_cellular():
    environment = _environment(_graph(
        raw_method_name="SERS-based target-specific cell imaging",
    ))
    assert environment.status == "known"
    assert environment.normalized_value == "cellular"


def test_alpha4b3b321_solution_method_is_unchanged():
    environment = _environment(_graph(
        raw_method_name="Solution-based SERS spectra",
        description="Solution-state SERS spectroscopy.",
    ))
    assert environment.status == "known"
    assert environment.normalized_value == "solution"


def test_alpha4b3b321_semantics_ids_are_versioned():
    adapter = SERS_AU_AG_COMPARISON_ADAPTER
    assert (
        adapter.semantics_id
        == "sers_au_ag_comparison_v7_alpha4b3b321"
    )
    assert adapter.method_semantics is not None
    assert (
        adapter.method_semantics.semantics_id
        == "sers_au_ag_method_v4_alpha4b3b321"
    )
