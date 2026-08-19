from __future__ import annotations

import json

import networkx as nx

from dac_her.comparison_context import build_protocol_assessments
from domains.sers.comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER,
)


def _graph(
    *,
    integration_time: float | None = None,
    source_expression: str = "Raman intensity measurement.",
    description: str = "",
    raw_method_name: str = "",
    conditions: list[dict[str, object]] | None = None,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "sub",
        type="PlasmonicSubstrate",
        label="test substrate",
    )
    rows = list(conditions or [])
    if integration_time is not None:
        rows.append({
            "name": "integration time",
            "value_numeric": integration_time,
            "value_text": None,
            "unit": "s",
        })
    graph.add_node(
        "exp",
        type="Experiment",
        label="SERS experiment",
        experiment_type="sers_spectroscopy",
        method_label="Surface-enhanced Raman spectroscopy",
        conditions_json=json.dumps(rows),
        description=description,
        raw_method_name=raw_method_name,
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
    graph.add_edge("sub", "exp", relation="TESTED_IN")
    graph.add_edge("exp", "m", relation="HAS_MEASUREMENT")
    graph.add_edge("m", "sub", relation="MEASURED_FOR")
    return graph


def _method(graph: nx.MultiDiGraph):
    return SERS_AU_AG_COMPARISON_ADAPTER.extract_method_contexts(
        graph,
        "P1",
    )[0]


def _protocol(
    left: nx.MultiDiGraph,
    right: nx.MultiDiGraph,
):
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
    return build_protocol_assessments(
        left_contexts + right_contexts,
        left_methods + right_methods,
        adapter=SERS_AU_AG_COMPARISON_ADAPTER,
    )[0]


def test_alpha4b3b32_mismatch_only_is_not_partially_matched():
    result = _protocol(
        _graph(integration_time=5.0),
        _graph(integration_time=10.0),
    )
    assert result.matched_dimensions == ()
    assert result.mismatched_dimensions == ("integration_time",)
    assert result.critical_mismatches == ()
    assert result.comparability == "different_protocol"


def test_alpha4b3b32_partial_requires_at_least_one_explicit_match():
    result = _protocol(
        _graph(integration_time=5.0),
        _graph(integration_time=5.0),
    )
    assert result.matched_dimensions == ("integration_time",)
    assert result.mismatched_dimensions == ()
    assert result.comparability == "partially_matched"


def test_alpha4b3b32_solution_method_is_environment_not_state():
    method = _method(_graph(
        raw_method_name="Solution-based SERS spectra",
        description=(
            "Solution-state SERS spectra were recorded for "
            "as-synthesized nanoparticles."
        ),
    ))
    dimensions = method.dimension_map
    assert dimensions["measurement_environment"].normalized_value == "solution"
    assert dimensions["sample_state"].status == "unknown"
    assert (
        dimensions["substrate_condition"].normalized_value
        == "as_prepared"
    )


def test_alpha4b3b32_solid_analyte_is_sample_state_not_substrate_condition():
    method = _method(_graph(
        source_expression="Raman spectrum of MB solid on bare glass.",
    ))
    dimensions = method.dimension_map
    assert dimensions["sample_state"].normalized_value == "solid"
    assert dimensions["substrate_condition"].status == "unknown"


def test_alpha4b3b32_dried_aqueous_sample_separates_preparation_medium():
    method = _method(_graph(
        conditions=[
            {
                "name": "medium",
                "value_numeric": None,
                "value_text": "aqueous methylene-blue solution",
                "unit": None,
            },
            {
                "name": "sample preparation",
                "value_numeric": None,
                "value_text": "solution dropped and air-dried on target",
                "unit": None,
            },
        ],
        description="The sample was drop-cast and dried before Raman analysis.",
    ))
    dimensions = method.dimension_map
    assert dimensions["preparation_medium"].normalized_value == "aqueous"
    assert dimensions["measurement_environment"].status == "unknown"
    assert dimensions["sample_state"].normalized_value == "dry"
    assert "drying" in dimensions["sample_preparation"].normalized_value


def test_alpha4b3b32_as_prepared_and_stored_is_ambiguous_condition():
    method = _method(_graph(
        description=(
            "SERS spectra of as-prepared and eight-month-stored "
            "substrates were compared."
        ),
    ))
    condition = method.dimension_map["substrate_condition"]
    assert condition.status == "ambiguous"
    assert condition.normalized_value == ""
    assert set(condition.source_values) == {
        "SERS spectra of as-prepared and eight-month-stored "
        "substrates were compared."
    }


def test_alpha4b3b32_role_dimensions_replace_legacy_conflation():
    adapter = SERS_AU_AG_COMPARISON_ADAPTER
    method_dimensions = set(adapter.method_semantics.dimensions)
    assert "medium" not in method_dimensions
    assert "substrate_state" not in method_dimensions
    assert {
        "sample_preparation",
        "preparation_medium",
        "measurement_environment",
        "sample_state",
        "substrate_condition",
    }.issubset(method_dimensions)

    comparison_dimensions = set(adapter.dimensions)
    assert "medium" not in comparison_dimensions
    assert "substrate_state" not in comparison_dimensions
    assert {
        "measurement_environment",
        "sample_state",
        "substrate_condition",
    }.issubset(comparison_dimensions)
