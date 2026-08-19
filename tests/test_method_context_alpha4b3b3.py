from __future__ import annotations

import networkx as nx

from dac_her.comparison_context import (
    apply_protocol_numeric_gate,
    build_pairwise_assessments,
    build_protocol_assessments,
)
from domains.sers.comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER,
)


def _graph(
    *,
    source_expression: str,
    analyte_label: str = "Methylene blue",
    analyte_global_concentration: str = "7 M",
    metric_id: str = "sers_enhancement_factor",
    value_numeric: str = "1000",
    unit: str = "",
    wavelength: str | None = None,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "sub",
        type="PlasmonicSubstrate",
        label="substrate",
    )
    graph.add_node(
        "exp",
        type="Experiment",
        label="SERS experiment",
    )
    graph.add_node(
        "a",
        type="Analyte",
        label=analyte_label,
        concentration=analyte_global_concentration,
    )
    graph.add_node(
        "m",
        type="Measurement",
        metric_id=metric_id,
        metric=metric_id,
        label=metric_id,
        value_numeric=value_numeric,
        value_text="",
        unit=unit,
        source_expression=source_expression,
    )
    graph.add_edge("sub", "exp", relation="TESTED_IN")
    graph.add_edge("exp", "a", relation="USES_ANALYTE")
    graph.add_edge("exp", "m", relation="HAS_MEASUREMENT")
    graph.add_edge("m", "sub", relation="MEASURED_FOR")

    if wavelength is not None:
        graph.add_node(
            "opt",
            type="OpticalCondition",
            label=f"{wavelength} excitation",
        )
        graph.add_edge(
            "exp",
            "opt",
            relation="USES_OPTICAL_CONDITION",
        )
    return graph


def test_alpha4b3b3_measurement_local_concentration_overrides_global_entity_leak():
    graph = _graph(
        source_expression=(
            "EF = 3.9 x 10^3 at 1037 cm^-1 "
            "for MB (10^-5 M)"
        ),
    )
    method = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        graph,
        "P1",
    )[0]
    concentration = method.dimension_map["analyte_concentration"]
    assert concentration.status == "known"
    assert concentration.normalized_value == "1e-5 M"
    assert concentration.source_node_ids == ("m",)
    assert concentration.provenance_scopes == (
        "measurement_source_expression",
    )

    comparison = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        graph,
        "P1",
    )[0]
    assert (
        comparison.dimension_map["concentration"].normalized_value
        == "1e-5 M"
    )
    assert "7 M" not in comparison.dimension_map["concentration"].source_values


def test_alpha4b3b3_detection_limit_result_is_not_context_concentration():
    graph = _graph(
        source_expression="The detection limit for MB was 10^-7 M.",
        metric_id="detection_limit",
        value_numeric="1e-7",
        unit="M",
    )
    method = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        graph,
        "P1",
    )[0]
    assert (
        method.dimension_map["analyte_concentration"].status
        == "unknown"
    )


def test_alpha4b3b3_precursor_concentration_does_not_become_analyte_context():
    graph = _graph(
        source_expression=(
            "Raman intensity increased at the optimal "
            "300 mM AgNO3 concentration."
        ),
        analyte_label="ATP",
        metric_id="raman_intensity",
        unit="cps",
    )
    method = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        graph,
        "P1",
    )[0]
    assert (
        method.dimension_map["analyte_concentration"].status
        == "unknown"
    )


def test_alpha4b3b3_probe_scoped_concentration_is_measurement_local():
    graph = _graph(
        source_expression="SERS intensity was 3185 cps with 10 mM ATP.",
        analyte_label="ATP",
        metric_id="raman_intensity",
        unit="cps",
    )
    method = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        graph,
        "P1",
    )[0]
    concentration = method.dimension_map["analyte_concentration"]
    assert concentration.status == "known"
    assert concentration.normalized_value == "0.01 M"


def test_alpha4b3b3_protocol_comparability_distinguishes_partial_from_different():
    left_graph = _graph(
        source_expression="Raman intensity with MB (10^-5 M).",
        wavelength="633 nm",
        metric_id="raman_intensity",
        unit="cps",
    )
    right_graph = _graph(
        source_expression="Raman intensity with MB (10^-7 M).",
        wavelength="633 nm",
        metric_id="raman_intensity",
        unit="cps",
    )

    left_methods = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        left_graph,
        "P1",
    )
    right_methods = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        right_graph,
        "P2",
    )
    left_contexts = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        left_graph,
        "P1",
    )
    right_contexts = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        right_graph,
        "P2",
    )

    protocols = build_protocol_assessments(
        left_contexts + right_contexts,
        left_methods + right_methods,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert len(protocols) == 1
    assert protocols[0].comparability == "partially_matched"
    assert "analyte" in protocols[0].matched_dimensions
    assert "excitation_wavelength" in protocols[0].matched_dimensions
    assert "analyte_concentration" in protocols[0].mismatched_dimensions

    different_graph = _graph(
        source_expression="Raman intensity with ATP (10^-5 M).",
        analyte_label="ATP",
        wavelength="633 nm",
        metric_id="raman_intensity",
        unit="cps",
    )
    different_methods = (
        SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
            different_graph,
            "P3",
        )
    )
    different_contexts = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        different_graph,
        "P3",
    )
    protocols = build_protocol_assessments(
        left_contexts + different_contexts,
        left_methods + different_methods,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert protocols[0].comparability == "different_protocol"
    assert "analyte" in protocols[0].critical_mismatches


def test_alpha4b3b3_numeric_ranking_is_protocol_gated():
    # Use EF because its observable policy can rank when complete. Populate
    # every comparison-required dimension plus every method dimension.
    def complete_graph(paper_id: str, value: str):
        graph = _graph(
            source_expression="EF for MB (10^-5 M) at 1624 cm^-1.",
            metric_id="sers_enhancement_factor",
            value_numeric=value,
            unit="dimensionless",
            wavelength="785 nm",
        )
        graph.nodes["m"]["raman_peak"] = "1624 cm^-1"
        graph.nodes["exp"]["sample_preparation"] = "drop-cast"
        graph.nodes["exp"]["preparation_medium"] = "ethanol"
        graph.nodes["exp"]["measurement_environment"] = "air"
        graph.nodes["exp"]["sample_state"] = "dry"
        graph.nodes["exp"]["substrate_condition"] = "as-prepared"
        graph.nodes["exp"]["integration_time"] = "10 s"
        graph.nodes["opt"]["label"] = (
            "785 nm excitation; laser power 1 mW"
        )
        # reporter is required by current EF numeric policy
        graph.add_node("r", type="RamanReporter", label="Methylene blue")
        graph.add_edge("exp", "r", relation="USES_REPORTER")
        return graph

    left_graph = complete_graph("P1", "1e6")
    right_graph = complete_graph("P2", "1e7")
    left_methods = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        left_graph,
        "P1",
    )
    right_methods = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        right_graph,
        "P2",
    )
    left_contexts = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        left_graph,
        "P1",
    )
    right_contexts = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        right_graph,
        "P2",
    )
    assessments = build_pairwise_assessments(
        left_contexts + right_contexts,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    protocols = build_protocol_assessments(
        left_contexts + right_contexts,
        left_methods + right_methods,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    gated = apply_protocol_numeric_gate(
        assessments,
        protocols,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assert protocols[0].comparability == "same_protocol"
    assert gated[0].protocol_comparability == "same_protocol"
    assert gated[0].numeric_ranking_allowed is True
