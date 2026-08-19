from __future__ import annotations

import networkx as nx

from domains.sers.metric_definition import (
    SERS_AU_AG_METRIC_DEFINITION_ADAPTER,
)


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node("sub", type="PlasmonicSubstrate", label="substrate")
    return graph


def _measurement(
    graph: nx.MultiDiGraph,
    measurement_id: str,
    *,
    metric_id: str,
    source_expression: str,
    description: str = "",
) -> None:
    graph.add_node(
        measurement_id,
        type="Measurement",
        label=metric_id,
        metric_id=metric_id,
        metric=metric_id,
        subject_id="sub",
        source_expression=source_expression,
        description=description,
        qualifier="",
        value_numeric="1",
        value_text="",
        unit="",
        basis="",
        conditions_json="[]",
    )
    graph.add_edge(measurement_id, "sub", relation="MEASURED_FOR")


def _producer(
    graph: nx.MultiDiGraph,
    measurement_id: str,
    *,
    description: str,
) -> None:
    graph.add_node(
        "exp",
        type="Experiment",
        label="experiment",
        description=description,
        method_details="",
        conditions_json="[]",
    )
    graph.add_edge("exp", measurement_id, relation="HAS_MEASUREMENT")


def _only(graph: nx.MultiDiGraph):
    rows = SERS_AU_AG_METRIC_DEFINITION_ADAPTER.extract_contexts(
        graph,
        "P1",
    )
    assert len(rows) == 1
    return rows[0]


def test_alpha4b3b4b1_estimated_ef_value_does_not_become_partial():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="sers_enhancement_factor",
        source_expression="EF was estimated to be 3.9 × 10^3.",
    )
    _producer(
        graph,
        "m",
        description="Enhancement factors were estimated at Raman peaks.",
    )
    row = _only(graph)
    assert row.definition_status == "unknown"
    assert row.definition_family == "reported_ef_unspecified"
    assert row.normalization_basis == "unspecified"
    assert row.reference_basis == "unspecified"


def test_alpha4b3b4b1_incomplete_molecule_normalization_is_partial():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="sers_enhancement_factor",
        source_expression="EF was reported for the substrate.",
    )
    _producer(
        graph,
        "m",
        description=(
            "The signal was normalized by the estimated molecule number, "
            "but the reference Raman basis was not reported."
        ),
    )
    row = _only(graph)
    assert row.definition_status == "partial"
    assert row.definition_family == "reported_ef_unspecified"
    assert row.normalization_basis == "molecule_count"
    assert row.reference_basis == "unspecified"


def test_alpha4b3b4b1_theoretical_lod_without_criterion_is_unknown():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="detection_limit",
        source_expression="The theoretical LOD was 2.4 nM.",
    )
    row = _only(graph)
    assert row.definition_status == "unknown"
    assert row.definition_family == "reported_lod_unspecified"
    assert row.criterion == ""


def test_alpha4b3b4b1_bare_lowest_concentration_is_not_detection_criterion():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="detection_limit",
        source_expression=(
            "MB at the lowest concentration of 10^-7 M was adsorbed "
            "on the alloy substrate."
        ),
    )
    row = _only(graph)
    assert row.definition_status == "unknown"
    assert row.definition_family == "reported_lod_unspecified"
    assert row.criterion == ""


def test_alpha4b3b4b1_lowest_concentration_that_can_be_detected_is_known():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="detection_limit",
        source_expression=(
            "The lowest concentration of MB that can be detected was "
            "10^-7 M."
        ),
    )
    row = _only(graph)
    assert row.definition_status == "known"
    assert row.definition_family == "lowest_detected_concentration"
    assert row.criterion == "lowest_observed_detection"


def test_alpha4b3b4b1_detection_level_statement_is_known():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="detection_limit",
        source_expression=(
            "The SERS detection level of MB was almost 10^-7 M."
        ),
    )
    row = _only(graph)
    assert row.definition_status == "known"
    assert row.definition_family == "lowest_detected_concentration"


def test_alpha4b3b4b1_semantics_version():
    assert (
        SERS_AU_AG_METRIC_DEFINITION_ADAPTER.semantics_id
        == "sers_au_ag_metric_definition_v3_alpha4c4c1"
    )
