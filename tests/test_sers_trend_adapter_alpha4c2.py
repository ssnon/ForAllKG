from __future__ import annotations

import json

import networkx as nx

from domains.sers.trend import (
    SERS_AU_AG_TREND_ADAPTER,
    SERS_AU_AG_TREND_SEMANTICS_ID,
)
from dac_her.trend_domain import TrendEvidenceSource


def _dimension(name: str, value: str) -> dict[str, object]:
    return {
        "name": name,
        "status": "known",
        "normalized_value": value,
        "source_values": [value],
        "source_node_ids": ["e1"],
        "provenance_scopes": ["experiment_conditions_json"],
    }


def _method(measurement: str, reporter: str = "r6g") -> dict[str, object]:
    dimensions = [
        _dimension("analyte", "r6g"),
        _dimension("reporter", reporter),
        _dimension("excitation_wavelength", "633 nm"),
    ]
    for name in (
        "analyte_concentration",
        "laser_power",
        "integration_time",
        "sample_preparation",
        "preparation_medium",
        "measurement_environment",
        "sample_state",
        "substrate_condition",
    ):
        dimensions.append({
            "name": name,
            "status": "unknown",
            "normalized_value": "",
            "source_values": [],
            "source_node_ids": [],
            "provenance_scopes": [],
        })
    return {
        "method_context_id": f"method:{measurement}",
        "paper_id": "P1",
        "measurement_id": measurement,
        "producer_ids": ["e1"],
        "subject_ids": ["s1"],
        "dimensions": dimensions,
        "source_node_ids": [measurement, "e1"],
    }


def _context(measurement: str, value: float) -> dict[str, object]:
    return {
        "context_id": f"ctx:{measurement}",
        "paper_id": "P1",
        "measurement_id": measurement,
        "observable_key": "raman_intensity",
        "observable_label": "Raman intensity",
        "value_numeric": value,
        "value_text": "",
        "unit": "a.u.",
        "source_expression": f"Raman intensity {value}",
        "subject_ids": ["s1"],
        "source_node_ids": [measurement, "g1", "e1"],
        "method_context_id": f"method:{measurement}",
    }


def _source(*, mismatch: bool = False) -> TrendEvidenceSource:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node("g1", type="MeasurementGroup")
    graph.add_node("e1", type="Experiment")
    graph.add_node("s1", type="PlasmonicSubstrate")
    shell_values = [3.6, 5.0, 8.4]
    response_values = [10.0, 20.0, 30.0]
    identities = []
    methods = []
    contexts = []
    for index, (shell, response) in enumerate(zip(shell_values, response_values), start=1):
        mid = f"m{index}"
        graph.add_node(
            mid,
            type="Measurement",
            conditions_json=json.dumps([
                {"name": "Ag shell thickness", "value_numeric": shell, "unit": "nm"}
            ]),
        )
        graph.add_edge("e1", mid, relation="HAS_MEASUREMENT")
        graph.add_edge(mid, "g1", relation="IN_MEASUREMENT_GROUP")
        identities.append({
            "identity_id": f"identity:{mid}",
            "representative_measurement_id": mid,
            "source_mention_ids": [mid],
        })
        reporter = "4-atp" if mismatch and index == 3 else "r6g"
        methods.append(_method(mid, reporter=reporter))
        contexts.append(_context(mid, response))

    graph.add_node(
        "claim_shell",
        type="ObservationClaim",
        statement=(
            "Raman intensity increases with Ag shell thickness and approaches "
            "the maximum once the critical thickness is reached."
        ),
    )
    graph.add_node(
        "claim_gap",
        type="ObservationClaim",
        statement=(
            "The SERS enhancement factor increases significantly as the "
            "interior gap size decreases."
        ),
    )
    graph.add_node(
        "claim_ratio",
        type="ObservationClaim",
        statement=(
            "Among the tested Au-Ag ratios, the Au:Ag ratio of 10:7 produced "
            "the strongest SERRS signal."
        ),
    )
    return TrendEvidenceSource(
        graph=graph,
        paper_id="P1",
        measurement_result_rows=tuple(identities),
        method_context_rows=tuple(methods),
        comparison_context_rows=tuple(contexts),
    )


def test_sers_adapter_extracts_numeric_series_and_three_claim_patterns():
    evidence = SERS_AU_AG_TREND_ADAPTER.extract_evidence(_source())
    numeric = [item for item in evidence if item.is_quantitative]
    claims = [item for item in evidence if not item.is_quantitative]
    assert len(numeric) == 1
    assert numeric[0].independent_variable_key == "shell_thickness"
    assert numeric[0].direction == "positive"
    assert numeric[0].shape == "monotonic"
    assert numeric[0].evidence_basis == "controlled_numeric_series"
    assert len(numeric[0].series_points) == 3

    by_control = {item.independent_variable_key: item for item in claims}
    assert by_control["shell_thickness"].direction == "positive"
    assert by_control["shell_thickness"].shape == "saturating"
    assert by_control["nanogap_size"].direction == "negative"
    assert by_control["nanogap_size"].shape == "monotonic"
    assert by_control["ag_to_au_ratio"].direction == "non_monotonic"
    assert by_control["ag_to_au_ratio"].shape == "single_optimum"


def test_explicit_method_mismatch_blocks_numeric_series_only():
    evidence = SERS_AU_AG_TREND_ADAPTER.extract_evidence(_source(mismatch=True))
    assert not any(item.is_quantitative for item in evidence)
    assert any(not item.is_quantitative for item in evidence)


def test_sers_trend_semantics_id():
    assert SERS_AU_AG_TREND_SEMANTICS_ID == "sers_au_ag_trend_v1_alpha4c2"
