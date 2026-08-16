from __future__ import annotations

import networkx as nx

from dac_her.domains.sers_au_ag_trend_alpha4c5g2r2 import (
    _candidate_local_methods_compatible,
)


def _measurement(
    graph,
    measurement_id: str,
    wavelength: float | None,
):
    conditions = []
    if wavelength is not None:
        conditions.append(
            {
                "name": "excitation wavelength",
                "value_numeric": wavelength,
                "unit": "nm",
            }
        )
    import json

    graph.add_node(
        measurement_id,
        type="Measurement",
        conditions_json=json.dumps(
            conditions
        ),
    )


def _ambiguous_method(
    method_id: str,
    *,
    extra_dimension=None,
):
    dimensions = [
        {
            "name": "excitation_wavelength",
            "status": "ambiguous",
            "normalized_value": "",
            "source_values": [
                "532 nm",
                "633 nm",
            ],
        }
    ]
    if extra_dimension is not None:
        dimensions.append(
            extra_dimension
        )
    return {
        "method_context_id": method_id,
        "dimensions": dimensions,
    }


def test_candidate_local_same_532_overrides_only_candidate():
    graph = nx.MultiDiGraph()
    _measurement(graph, "m1", 532.0)
    _measurement(graph, "m2", 532.0)

    rows = [
        {
            "measurement_id": "m1",
            "method": _ambiguous_method("ctx1"),
        },
        {
            "measurement_id": "m2",
            "method": _ambiguous_method("ctx2"),
        },
    ]

    ok, audit = (
        _candidate_local_methods_compatible(
            graph=graph,
            rows=rows,
            varied_control_key="nanogap_size",
        )
    )
    assert ok is True
    assert audit["override_used"] is True
    assert audit["local_value_nm"] == 532.0


def test_candidate_local_disagreeing_wavelengths_fail_closed():
    graph = nx.MultiDiGraph()
    _measurement(graph, "m1", 532.0)
    _measurement(graph, "m2", 633.0)

    rows = [
        {
            "measurement_id": "m1",
            "method": _ambiguous_method("ctx1"),
        },
        {
            "measurement_id": "m2",
            "method": _ambiguous_method("ctx2"),
        },
    ]

    ok, audit = (
        _candidate_local_methods_compatible(
            graph=graph,
            rows=rows,
            varied_control_key="nanogap_size",
        )
    )
    assert ok is False
    assert audit["reason"] == (
        "candidate_local_excitation_values_disagree"
    )


def test_candidate_local_missing_wavelength_fails_closed():
    graph = nx.MultiDiGraph()
    _measurement(graph, "m1", 532.0)
    _measurement(graph, "m2", None)

    rows = [
        {
            "measurement_id": "m1",
            "method": _ambiguous_method("ctx1"),
        },
        {
            "measurement_id": "m2",
            "method": _ambiguous_method("ctx2"),
        },
    ]

    ok, audit = (
        _candidate_local_methods_compatible(
            graph=graph,
            rows=rows,
            varied_control_key="nanogap_size",
        )
    )
    assert ok is False
    assert audit["reason"] == (
        "candidate_measurement_missing_single_local_excitation"
    )


def test_non_excitation_ambiguity_cannot_be_overridden():
    graph = nx.MultiDiGraph()
    _measurement(graph, "m1", 532.0)
    _measurement(graph, "m2", 532.0)

    ambiguous_analyte = {
        "name": "analyte",
        "status": "ambiguous",
        "normalized_value": "",
        "source_values": [
            "R6G",
            "4-ATP",
        ],
    }
    rows = [
        {
            "measurement_id": "m1",
            "method": _ambiguous_method(
                "ctx1",
                extra_dimension=(
                    ambiguous_analyte
                ),
            ),
        },
        {
            "measurement_id": "m2",
            "method": _ambiguous_method(
                "ctx2",
                extra_dimension=(
                    ambiguous_analyte
                ),
            ),
        },
    ]

    ok, audit = (
        _candidate_local_methods_compatible(
            graph=graph,
            rows=rows,
            varied_control_key="nanogap_size",
        )
    )
    assert ok is False
    assert audit["reason"] == (
        "non_excitation_method_dimension_ambiguous"
    )
