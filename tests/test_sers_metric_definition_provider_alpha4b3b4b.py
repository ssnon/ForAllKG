from __future__ import annotations

import networkx as nx

from dac_her.domains.sers_au_ag_metric_definition import (
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
    qualifier: str = "",
    value_numeric: str = "1",
    value_text: str = "",
    basis: str = "",
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
        qualifier=qualifier,
        value_numeric=value_numeric,
        value_text=value_text,
        unit="",
        basis=basis,
        conditions_json="[]",
    )
    graph.add_edge(measurement_id, "sub", relation="MEASURED_FOR")


def _producer(
    graph: nx.MultiDiGraph,
    producer_id: str,
    measurement_id: str,
    *,
    node_type: str,
    description: str = "",
    method_details: str = "",
    conditions_json: str = "[]",
) -> None:
    graph.add_node(
        producer_id,
        type=node_type,
        label=producer_id,
        description=description,
        method_details=method_details,
        conditions_json=conditions_json,
    )
    graph.add_edge(producer_id, measurement_id, relation="HAS_MEASUREMENT")


def _only(graph: nx.MultiDiGraph):
    rows = SERS_AU_AG_METRIC_DEFINITION_ADAPTER.extract_contexts(graph, "P1")
    assert len(rows) == 1
    return rows[0]


def test_molecule_normalized_ef_is_known():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="sers_enhancement_factor",
        source_expression="Enhancement factor was 4.2e6.",
        basis="Intensity normalized by estimated molecule number",
    )
    _producer(
        graph,
        "calc",
        "m",
        node_type="Calculation",
        method_details=(
            "Calculated from SERS and normal Raman intensities normalized "
            "by estimated molecule numbers."
        ),
    )
    row = _only(graph)
    assert row.definition_status == "known"
    assert row.definition_family == "molecule_normalized_intensity_ratio"
    assert row.normalization_basis == "molecule_count"
    assert row.reference_basis == "normal_raman"
    assert row.source_calculation_ids == ("calc",)
    assert "normalized by estimated molecule numbers" in row.formula_text


def test_concentration_normalized_ef_on_glass_is_known():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="sers_enhancement_factor",
        source_expression="EF was estimated at the 1624 cm^-1 peak.",
    )
    _producer(
        graph,
        "calc",
        "m",
        node_type="Calculation",
        method_details=(
            "EF = (I_SERS/I_nor) × (C_nor/C_SERS), comparing the SERS "
            "substrate with analyte on glass under the same condition."
        ),
        conditions_json=(
            '[{"name":"Raman peak","value_numeric":1624,'
            '"value_text":null,"unit":"cm^-1","reference":null}]'
        ),
    )
    row = _only(graph)
    assert row.definition_status == "known"
    assert row.definition_family == "concentration_normalized_intensity_ratio"
    assert row.normalization_basis == "concentration"
    assert row.reference_basis == "normal_raman_on_glass"
    assert row.raman_peak == "1624 cm^-1"


def test_calculated_ef_without_formula_is_unknown():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="sers_enhancement_factor",
        source_expression="EF at 1624 cm^-1 = 1.2e7.",
    )
    _producer(
        graph,
        "exp",
        "m",
        node_type="Experiment",
        description="SERS enhancement factors were calculated at Raman peaks.",
    )
    row = _only(graph)
    assert row.definition_status == "unknown"
    assert row.definition_family == "reported_ef_unspecified"
    assert row.formula_text == ""


def test_plain_ef_result_remains_unknown():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="sers_enhancement_factor",
        source_expression="EF at 1624 cm^-1 = 1.2e7.",
    )
    row = _only(graph)
    assert row.definition_status == "unknown"
    assert row.definition_family == "reported_ef_unspecified"


def test_population_mean_is_separate_from_metric_definition():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="sers_enhancement_factor",
        source_expression="Mean SERS EF for 75 single nanoparticles was 4.2e6.",
    )
    row = _only(graph)
    assert row.definition_status == "unknown"
    assert row.aggregation_scope == "population_mean"


def test_calibration_curve_lod_is_known():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="detection_limit",
        source_expression="The theoretical LOD was 2.4 nM.",
        description=(
            "Theoretical detection limit calculated from the response "
            "standard deviation and calibration-curve slope."
        ),
    )
    row = _only(graph)
    assert row.definition_status == "known"
    assert row.definition_family == "calibration_curve_statistical"
    assert row.criterion == "response_standard_deviation_and_calibration_slope"
    assert row.aggregation_scope == "not_applicable"


def test_lowest_detected_concentration_lod_is_known():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="detection_limit",
        source_expression="Signal could be detected even at 500 fM.",
        description="Lowest explicitly reported concentration detected.",
    )
    row = _only(graph)
    assert row.definition_status == "known"
    assert row.definition_family == "lowest_detected_concentration"
    assert row.criterion == "lowest_observed_detection"


def test_generic_lod_value_does_not_imply_definition():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="detection_limit",
        source_expression="LOD of 2.4 nM for ATP.",
    )
    row = _only(graph)
    assert row.definition_status == "unknown"
    assert row.definition_family == "reported_lod_unspecified"


def test_unconnected_formula_does_not_leak():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="sers_enhancement_factor",
        source_expression="EF = 1.0e6.",
    )
    graph.add_node(
        "unrelated_calc",
        type="Calculation",
        label="unrelated",
        method_details="EF = (I_SERS/I_nor) × (C_nor/C_SERS).",
        conditions_json="[]",
    )
    row = _only(graph)
    assert row.definition_status == "unknown"
    assert "unrelated_calc" not in row.source_node_ids


def test_unregistered_measurements_do_not_get_contexts():
    graph = _graph()
    _measurement(
        graph,
        "m",
        metric_id="raman_intensity",
        source_expression="Intensity = 20 cps.",
    )
    assert SERS_AU_AG_METRIC_DEFINITION_ADAPTER.extract_contexts(graph, "P1") == []
