from __future__ import annotations

import networkx as nx

from dac_her.domains.sers_au_ag_trend_alpha4c5g2 import (
    _comparative_gap_direction,
    _resolved_claim_control,
    resolve_measurement_local_method_contexts,
)
from dac_her.trend_domain import TrendEvidenceSource


def test_explicit_nanogap_dimension_wins_presence_overlap():
    control = _resolved_claim_control(
        "SERS enhancement increases as the interior "
        "nanogap size decreases."
    )
    assert control == ("nanogap_size", "nanogap size")


def test_explicit_pair_comparison_negative_direction():
    result = _comparative_gap_direction(
        "The SERS enhancement factor is greater for the "
        "2-nm interior gap than for the 8-nm gap."
    )
    assert result == ("negative", "monotonic")


def test_explicit_pair_comparison_positive_direction():
    result = _comparative_gap_direction(
        "The SERS signal is greater for the 8-nm gap "
        "than for the 2-nm gap."
    )
    assert result == ("positive", "monotonic")


def test_measurement_local_excitation_resolves_ambiguity():
    graph = nx.MultiDiGraph()
    for measurement_id in ("m1", "m2"):
        graph.add_node(
            measurement_id,
            type="Measurement",
            conditions_json=(
                '[{"name":"excitation wavelength",'
                '"value_numeric":532.0,"unit":"nm"}]'
            ),
        )

    source = TrendEvidenceSource(
        graph=graph,
        paper_id="P1",
        measurement_result_rows=(),
        method_context_rows=(
            {
                "paper_id": "P1",
                "method_context_id": "ctx1",
                "dimensions": [
                    {
                        "name": "excitation_wavelength",
                        "status": "ambiguous",
                        "normalized_value": "",
                        "source_values": [
                            "532 nm",
                            "633 nm",
                        ],
                    }
                ],
            },
        ),
        comparison_context_rows=(
            {
                "paper_id": "P1",
                "context_id": "c1",
                "measurement_id": "m1",
                "method_context_id": "ctx1",
            },
            {
                "paper_id": "P1",
                "context_id": "c2",
                "measurement_id": "m2",
                "method_context_id": "ctx1",
            },
        ),
    )

    resolved, audit = (
        resolve_measurement_local_method_contexts(source)
    )
    dimension = resolved.method_context_rows[0][
        "dimensions"
    ][0]
    assert dimension["status"] == "known"
    assert dimension["normalized_value"] == "532 nm"
    assert audit[0]["resolved"] is True


def test_locality_fails_closed_when_one_measurement_missing():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "m1",
        type="Measurement",
        conditions_json=(
            '[{"name":"excitation wavelength",'
            '"value_numeric":532.0,"unit":"nm"}]'
        ),
    )
    graph.add_node(
        "m2",
        type="Measurement",
        conditions_json="[]",
    )

    source = TrendEvidenceSource(
        graph=graph,
        paper_id="P1",
        measurement_result_rows=(),
        method_context_rows=(
            {
                "paper_id": "P1",
                "method_context_id": "ctx1",
                "dimensions": [
                    {
                        "name": "excitation_wavelength",
                        "status": "ambiguous",
                        "normalized_value": "",
                        "source_values": [
                            "532 nm",
                            "633 nm",
                        ],
                    }
                ],
            },
        ),
        comparison_context_rows=(
            {
                "paper_id": "P1",
                "measurement_id": "m1",
                "method_context_id": "ctx1",
            },
            {
                "paper_id": "P1",
                "measurement_id": "m2",
                "method_context_id": "ctx1",
            },
        ),
    )

    resolved, audit = (
        resolve_measurement_local_method_contexts(source)
    )
    dimension = resolved.method_context_rows[0][
        "dimensions"
    ][0]
    assert dimension["status"] == "ambiguous"
    assert audit[0]["resolved"] is False
