from __future__ import annotations

import networkx as nx

from dac_her.comparison_context import (
    build_pairwise_assessments,
)
from domains.sers.comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER,
)


def _graph(
    *,
    analyte: str = "R6G",
    concentration: str = "1e-6 M",
    wavelength: str = "785 nm excitation",
    numeric: str = "100.0",
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "sub",
        type="PlasmonicSubstrate",
        label="Au-Ag substrate",
    )
    graph.add_node(
        "exp",
        type="Experiment",
        label="Raman experiment",
        measurement_environment="aqueous",
        integration_time="10 s",
        sample_state="dry",
        substrate_condition="as-prepared",
    )
    graph.add_node(
        "a",
        type="Analyte",
        label=analyte,
    )
    graph.add_node(
        "r",
        type="RamanReporter",
        label="R6G",
    )
    graph.add_node(
        "opt",
        type="OpticalCondition",
        label=f"{wavelength}; laser power 1 mW",
    )
    graph.add_node(
        "m",
        type="Measurement",
        label="SERS intensity at Raman peak 1620 cm^-1",
        metric_id="raman_intensity",
        metric="Raman intensity",
        value_numeric=numeric,
        value_text="",
        unit="a.u.",
        source_expression="SERS intensity at Raman peak 1620 cm^-1",
        analyte_concentration=concentration,
    )
    graph.add_edge(
        "sub", "exp", relation="TESTED_IN", edge_id="tested",
    )
    graph.add_edge(
        "exp", "a", relation="USES_ANALYTE", edge_id="analyte",
    )
    graph.add_edge(
        "exp", "r", relation="USES_REPORTER", edge_id="reporter",
    )
    graph.add_edge(
        "exp", "opt", relation="USES_OPTICAL_CONDITION", edge_id="optical",
    )
    graph.add_edge(
        "exp", "m", relation="HAS_MEASUREMENT", edge_id="measurement",
    )
    graph.add_edge(
        "m", "sub", relation="MEASURED_FOR", edge_id="for",
    )
    return graph


def test_alpha4b3b_sers_provider_extracts_explicit_context_only():
    contexts = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        _graph(),
        "P1",
    )
    assert len(contexts) == 1
    context = contexts[0]
    assert context.observable_key == "raman_intensity"
    assert context.value_numeric == 100.0
    assert context.subject_ids == ("sub",)

    dimensions = context.dimension_map
    assert dimensions["analyte"].normalized_value == "r6g"
    assert dimensions["reporter"].normalized_value == "r6g"
    assert dimensions["concentration"].status == "known"
    assert dimensions["concentration"].normalized_value == "1e-6 M"
    assert dimensions["excitation_wavelength"].normalized_value == "785 nm"
    assert dimensions["laser_power"].normalized_value == "1 mW"
    assert dimensions["integration_time"].normalized_value == "10 s"
    assert dimensions["raman_peak"].status == "known"
    assert (
        dimensions["measurement_environment"].normalized_value
        == "aqueous"
    )
    assert dimensions["sample_state"].normalized_value == "dry"
    assert (
        dimensions["substrate_condition"].normalized_value
        == "as_prepared"
    )


def test_alpha4b3b_sers_missing_context_remains_unknown_not_invented():
    graph = _graph()
    del graph.nodes["m"]["analyte_concentration"]
    del graph.nodes["exp"]["measurement_environment"]
    del graph.nodes["exp"]["sample_state"]
    del graph.nodes["exp"]["substrate_condition"]
    graph.nodes["opt"]["label"] = "785 nm excitation"
    del graph.nodes["exp"]["integration_time"]

    context = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        graph,
        "P1",
    )[0]
    dimensions = context.dimension_map
    assert dimensions["concentration"].status == "unknown"
    assert dimensions["laser_power"].status == "unknown"
    assert dimensions["integration_time"].status == "unknown"
    assert dimensions["measurement_environment"].status == "unknown"
    assert dimensions["sample_state"].status == "unknown"
    assert dimensions["substrate_condition"].status == "unknown"


def test_alpha4b3b_sers_cross_paper_mismatch_blocks_ranking():
    left = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        _graph(analyte="R6G"),
        "P1",
    )[0]
    right = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        _graph(analyte="Methylene blue"),
        "P2",
    )[0]

    assessments = build_pairwise_assessments(
        [left, right],
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert len(assessments) == 1
    assert assessments[0].compatibility == "incompatible"
    assert "analyte" in assessments[0].mismatched_dimensions
    assert assessments[0].numeric_ranking_allowed is False


def test_alpha4b3b_sers_complete_identical_context_can_rank_numeric_values():
    left = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        _graph(numeric="100"),
        "P1",
    )[0]
    right = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        _graph(numeric="200"),
        "P2",
    )[0]
    assessments = build_pairwise_assessments(
        [left, right],
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert assessments[0].compatibility == "compatible"
    assert assessments[0].numeric_ranking_allowed is True
