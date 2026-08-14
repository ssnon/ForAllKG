from __future__ import annotations

import networkx as nx
import pytest

from dac_her.trend_domain import (
    TREND_EVIDENCE_CONTRACT_SEMANTICS_ID,
    TrendDomainAdapter,
    TrendEvidence,
    TrendEvidenceSource,
    TrendSeriesPoint,
)
from dac_her.trend_evidence import audit_trend_evidence, stable_trend_id


def _numeric_point(point_id: str, x: float, y: float, measurement_id: str) -> TrendSeriesPoint:
    return TrendSeriesPoint(
        point_id=point_id,
        independent_value_numeric=x,
        independent_unit="nm",
        dependent_value_numeric=y,
        dependent_unit="a.u.",
        source_measurement_ids=(measurement_id,),
        source_node_ids=(measurement_id,),
    )


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="test_domain")
    for node_id in ("m1", "m2", "m3"):
        graph.add_node(node_id, type="Measurement")
    graph.add_node("g1", type="MeasurementGroup")
    graph.add_node("e1", type="Experiment")
    graph.add_node("claim1", type="ObservationClaim")
    graph.add_node("subject1", type="Material")
    return graph


def test_controlled_numeric_series_contract():
    item = TrendEvidence(
        trend_id="trend:test",
        domain_profile_id="test_domain",
        trend_semantics_id="test_trend_v1",
        paper_id="P1",
        independent_variable_key="shell_thickness",
        independent_variable_label="shell thickness",
        dependent_observable_key="signal",
        dependent_observable_label="signal",
        direction="positive",
        shape="saturating",
        evidence_basis="controlled_numeric_series",
        varied_dimension="shell_thickness",
        subject_ids=("subject1",),
        series_points=(
            _numeric_point("p1", 1.0, 10.0, "m1"),
            _numeric_point("p2", 2.0, 20.0, "m2"),
            _numeric_point("p3", 3.0, 25.0, "m3"),
        ),
        source_measurement_ids=("m1", "m2", "m3"),
        source_measurement_group_ids=("g1",),
        source_experiment_ids=("e1",),
        source_node_ids=("m1", "m2", "m3", "g1", "e1"),
    )
    assert item.is_quantitative is True


def test_numeric_pair_requires_exactly_two_points():
    with pytest.raises(ValueError):
        TrendEvidence(
            trend_id="trend:test",
            domain_profile_id="test_domain",
            trend_semantics_id="test_trend_v1",
            paper_id="P1",
            independent_variable_key="x",
            independent_variable_label="x",
            dependent_observable_key="y",
            dependent_observable_label="y",
            direction="positive",
            shape="monotonic",
            evidence_basis="controlled_numeric_pair",
            varied_dimension="x",
            series_points=(
                _numeric_point("p1", 1.0, 1.0, "m1"),
                _numeric_point("p2", 2.0, 2.0, "m2"),
                _numeric_point("p3", 3.0, 3.0, "m3"),
            ),
            source_measurement_ids=("m1", "m2", "m3"),
            source_measurement_group_ids=("g1",),
            source_node_ids=("m1", "m2", "m3", "g1"),
        )


def test_numeric_trend_requires_shared_lineage():
    with pytest.raises(ValueError):
        TrendEvidence(
            trend_id="trend:test",
            domain_profile_id="test_domain",
            trend_semantics_id="test_trend_v1",
            paper_id="P1",
            independent_variable_key="x",
            independent_variable_label="x",
            dependent_observable_key="y",
            dependent_observable_label="y",
            direction="positive",
            shape="monotonic",
            evidence_basis="controlled_numeric_pair",
            varied_dimension="x",
            series_points=(
                _numeric_point("p1", 1.0, 1.0, "m1"),
                _numeric_point("p2", 2.0, 2.0, "m2"),
            ),
            source_measurement_ids=("m1", "m2"),
            source_node_ids=("m1", "m2"),
        )


def test_reported_correlation_cannot_be_upgraded_to_causal():
    with pytest.raises(ValueError):
        TrendEvidence(
            trend_id="trend:claim",
            domain_profile_id="test_domain",
            trend_semantics_id="test_trend_v1",
            paper_id="P1",
            independent_variable_key="x",
            independent_variable_label="x",
            dependent_observable_key="y",
            dependent_observable_label="y",
            direction="positive",
            shape="unspecified",
            evidence_basis="reported_correlation",
            causal_status="source_asserted",
            source_expression="x correlated with y",
            source_claim_ids=("claim1",),
            source_node_ids=("claim1",),
        )


def test_claim_lane_cannot_masquerade_as_numeric_series():
    with pytest.raises(ValueError):
        TrendEvidence(
            trend_id="trend:claim",
            domain_profile_id="test_domain",
            trend_semantics_id="test_trend_v1",
            paper_id="P1",
            independent_variable_key="x",
            independent_variable_label="x",
            dependent_observable_key="y",
            dependent_observable_label="y",
            direction="positive",
            shape="unspecified",
            evidence_basis="reported_directional_claim",
            source_expression="x increased y",
            source_claim_ids=("claim1",),
            source_node_ids=("claim1", "m1", "m2"),
            series_points=(
                _numeric_point("p1", 1.0, 1.0, "m1"),
                _numeric_point("p2", 2.0, 2.0, "m2"),
            ),
        )


def test_audit_accepts_grounded_claim_evidence():
    graph = _graph()
    item = TrendEvidence(
        trend_id="trend:claim",
        domain_profile_id="test_domain",
        trend_semantics_id="test_trend_v1",
        paper_id="P1",
        independent_variable_key="x",
        independent_variable_label="x",
        dependent_observable_key="y",
        dependent_observable_label="y",
        direction="positive",
        shape="unspecified",
        evidence_basis="reported_directional_claim",
        source_expression="x increased y",
        source_claim_ids=("claim1",),
        source_node_ids=("claim1",),
    )
    adapter = TrendDomainAdapter(
        adapter_id="test",
        domain_profile_id="test_domain",
        semantics_id="test_trend_v1",
        supported_evidence_bases=frozenset({"reported_directional_claim"}),
        required_inputs=frozenset({"canonical_graph"}),
        extract_evidence_fn=lambda _source: [item],
    )
    source = TrendEvidenceSource(graph=graph, paper_id="P1")
    evidence = adapter.extract_evidence(source)
    audit = audit_trend_evidence(
        evidence=evidence,
        sources={"P1": source},
        adapter=adapter,
    )
    assert audit.structural_gate is True
    assert audit.evidence_count == 1
    assert audit.claim_evidence_count == 1


def test_stable_trend_id_is_source_order_invariant():
    left = stable_trend_id(
        paper_id="P1",
        independent_variable_key="x",
        dependent_observable_key="y",
        evidence_basis="reported_directional_claim",
        source_node_ids=("claim1", "m1"),
    )
    right = stable_trend_id(
        paper_id="P1",
        independent_variable_key="x",
        dependent_observable_key="y",
        evidence_basis="reported_directional_claim",
        source_node_ids=("m1", "claim1"),
    )
    assert left == right


def test_contract_semantics_id():
    assert TREND_EVIDENCE_CONTRACT_SEMANTICS_ID == "trend_evidence_contract_v1_alpha4c1"
