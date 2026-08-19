from __future__ import annotations

import json

import networkx as nx
import pytest

from dac_her.comparison_context import (
    apply_protocol_numeric_gate,
    audit_comparison_outputs,
    build_pairwise_assessments,
    build_protocol_assessments,
)
from domains.sers.comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER,
)


def _base_graph(
    *,
    source_expression: str = "SERS measurement",
    producer_type: str = "Experiment",
    conditions: list[dict[str, object]] | None = None,
    description: str = "",
    raw_method_name: str = "",
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "sub",
        type="PlasmonicSubstrate",
        label="test substrate",
    )
    graph.add_node(
        "exp",
        type=producer_type,
        label="test producer",
        experiment_type=(
            "sers_spectroscopy"
            if producer_type == "Experiment"
            else ""
        ),
        experiment_family=(
            "spectroscopy"
            if producer_type == "Experiment"
            else ""
        ),
        method_label=(
            "Surface-enhanced Raman spectroscopy"
            if producer_type == "Experiment"
            else ""
        ),
        conditions_json=json.dumps(conditions or []),
        description=description,
        raw_method_name=raw_method_name,
    )
    graph.add_node(
        "a",
        type="Analyte",
        label="Methylene blue",
    )
    graph.add_node(
        "m",
        type="Measurement",
        label="Raman intensity",
        metric_id="raman_intensity",
        metric="Raman intensity",
        value_numeric="100",
        value_text="",
        unit="a.u.",
        source_expression=source_expression,
    )
    relation = "TESTED_IN" if producer_type == "Experiment" else "SIMULATED_BY"
    graph.add_edge("sub", "exp", relation=relation)
    graph.add_edge("exp", "a", relation="USES_ANALYTE")
    graph.add_edge("exp", "m", relation="HAS_MEASUREMENT")
    graph.add_edge("m", "sub", relation="MEASURED_FOR")
    return graph


def _dimension(graph: nx.MultiDiGraph, name: str):
    method = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        graph,
        "P1",
    )[0]
    return method.dimension_map[name]


def test_alpha4b3b31_structured_conditions_harvest_protocol_fields():
    graph = _base_graph(
        conditions=[
            {
                "name": "excitation wavelength",
                "value_numeric": 532.0,
                "value_text": None,
                "unit": "nm",
            },
            {
                "name": "laser power",
                "value_numeric": 3.2,
                "value_text": None,
                "unit": "mW",
            },
            {
                "name": "signal acquisition time",
                "value_numeric": 8.0,
                "value_text": None,
                "unit": "s",
            },
            {
                "name": "medium",
                "value_numeric": None,
                "value_text": "pure water",
                "unit": None,
            },
            {
                "name": "substrate state",
                "value_numeric": None,
                "value_text": (
                    "MB solution dried on drop-cast nanoplates"
                ),
                "unit": None,
            },
        ],
        description=(
            "The sample was deposited on glass and dried before SERS."
        ),
    )
    method = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        graph,
        "P1",
    )[0]
    dimensions = method.dimension_map

    assert dimensions["excitation_wavelength"].normalized_value == "532 nm"
    assert dimensions["laser_power"].normalized_value == "3.2 mW"
    assert dimensions["integration_time"].normalized_value == "8 s"
    assert dimensions["preparation_medium"].normalized_value == "aqueous"
    assert (
        dimensions["sample_preparation"].normalized_value
        == "drop_cast+deposition+drying"
    )
    assert dimensions["sample_state"].normalized_value == "dry"
    assert "experiment_conditions_json" in (
        dimensions["preparation_medium"].provenance_scopes
    )


def test_alpha4b3b31_incubation_and_adsorption_are_explicit_events():
    incubation = _base_graph(
        source_expression=(
            "SERS maps were obtained after incubation with "
            "cRGD-functionalized probes."
        )
    )
    assert (
        _dimension(incubation, "sample_preparation").normalized_value
        == "incubation"
    )

    adsorption = _base_graph(
        source_expression=(
            "MB adsorbed on Ag-Au nanoplates was measured."
        )
    )
    assert (
        _dimension(adsorption, "sample_preparation").normalized_value
        == "adsorption"
    )


def test_alpha4b3b31_simulation_water_is_not_physical_protocol_medium():
    graph = _base_graph(
        source_expression=(
            "A water-filled interior nanogap was simulated."
        ),
        producer_type="Calculation",
        conditions=[
            {
                "name": "medium",
                "value_numeric": None,
                "value_text": "water",
                "unit": None,
            }
        ],
    )
    assert _dimension(graph, "preparation_medium").status == "unknown"
    assert _dimension(graph, "measurement_environment").status == "unknown"
    assert _dimension(graph, "sample_preparation").status == "unknown"


def test_alpha4b3b31_subject_global_state_is_not_measurement_state():
    graph = _base_graph()
    graph.nodes["sub"]["substrate_state"] = "dry"
    assert _dimension(graph, "sample_state").status == "unknown"
    assert _dimension(graph, "substrate_condition").status == "unknown"


def test_alpha4b3b31_solution_method_is_measurement_environment():
    graph = _base_graph(
        raw_method_name="Solution-based SERS spectra",
        description="Solution-state SERS analysis of nanoparticles.",
    )
    assert _dimension(graph, "preparation_medium").status == "unknown"
    assert (
        _dimension(graph, "measurement_environment").normalized_value
        == "solution"
    )
    assert _dimension(graph, "sample_state").status == "unknown"
    assert _dimension(graph, "substrate_condition").status == "unknown"


def test_alpha4b3b31_as_synthesized_is_substrate_condition():
    graph = _base_graph(
        description="Spectra were recorded for as-synthesized nanoparticles."
    )
    assert (
        _dimension(graph, "substrate_condition").normalized_value
        == "as_prepared"
    )


def test_alpha4b3b31_malformed_conditions_json_fails_closed():
    graph = _base_graph()
    graph.nodes["exp"]["conditions_json"] = "{bad json"
    with pytest.raises(ValueError, match="Malformed conditions_json"):
        SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
            graph,
            "P1",
        )


def test_alpha4b3b31_audit_reports_method_coverage_and_protocol_matches():
    left = _base_graph(
        conditions=[
            {
                "name": "integration time",
                "value_numeric": 5.0,
                "value_text": None,
                "unit": "s",
            }
        ]
    )
    right = _base_graph(
        conditions=[
            {
                "name": "integration time",
                "value_numeric": 5.0,
                "value_text": None,
                "unit": "s",
            }
        ]
    )

    left_methods = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        left,
        "P1",
    )
    right_methods = SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        right,
        "P2",
    )
    left_contexts = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        left,
        "P1",
    )
    right_contexts = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        right,
        "P2",
    )
    contexts = left_contexts + right_contexts
    methods = left_methods + right_methods
    assessments = build_pairwise_assessments(
        contexts,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    protocols = build_protocol_assessments(
        contexts,
        methods,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    assessments = apply_protocol_numeric_gate(
        assessments,
        protocols,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )
    audit = audit_comparison_outputs(
        contexts=contexts,
        assessments=assessments,
        source_graphs={"P1": left, "P2": right},
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
        method_contexts=methods,
        protocol_assessments=protocols,
    )

    assert audit["passes_structural_gate"] is True
    assert (
        audit["method_dimension_status_counts"]["integration_time"]["known"]
        == 2
    )
    assert audit["protocol_pairs_with_any_match"] == 1
    assert (
        audit["protocol_matched_dimension_counts"]["analyte"]
        >= 1
    )
    assert (
        audit["protocol_matched_dimension_counts"]["integration_time"]
        >= 1
    )
