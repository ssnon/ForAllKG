from __future__ import annotations

import networkx as nx

from pipeline_core.discovery.higher_order_operationalization_target_facets import (
    decompose_target_facets,
    resolve_target_facet_operationalizations,
)
from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirement,
    OperationalizationWitnessRequirementSet,
)


def _requirements(target):
    row = OperationalizationWitnessRequirement(
        requirement_id="req:1",
        source_experiment_id="exp:src",
        output_experiment_id="exp:out",
        issue_code="MEASUREMENT_INDEPENDENCE_UNRESOLVED",
        status="unresolved_requires_measurement_witness",
        requested_target_observable=target,
        anchor_observable="anchor",
        observable_pair=[target, "anchor"],
        required_graph_topology=["test"],
        minimum_candidate_evidence=["test"],
        independence_acceptance_rule="test",
        independence_rejection_rule="test",
    )
    return OperationalizationWitnessRequirementSet(
        repaired_experiment_count=1,
        requirement_count=1,
        candidate_inspiration_requirement_count=0,
        requirements=[row],
    )


def _add_measurement(
    graph,
    *,
    mid,
    metric_id,
    metric,
    source_expression,
):
    provider = "calc:" + mid
    graph.add_node(
        provider,
        type="Calculation",
        method_details="test",
        calculation_type="other",
        source_paper_id="paper:1",
    )
    graph.add_node(
        mid,
        type="Measurement",
        metric_id=metric_id,
        metric=metric,
        label=metric,
        source_expression=source_expression,
        subject_id="sample:1",
        source_paper_id="paper:1",
    )
    graph.add_edge(
        provider,
        mid,
        relation="HAS_MEASUREMENT",
        paper_id="paper:1",
    )


def test_coordinated_target_decomposes_head_facets_without_synonymy():
    rows = decompose_target_facets(
        "electromagnetic hotspot location and intensity"
    )

    assert [row.facet_text for row in rows] == [
        "electromagnetic hotspot location",
        "electromagnetic hotspot intensity",
    ]
    assert all(
        row.decomposition_mode == "coordinated_head_facet"
        for row in rows
    )
    assert all(
        row.decomposition_is_scientific_equivalence is False
        for row in rows
    )


def test_uncoordinated_target_remains_single_target():
    rows = decompose_target_facets("field enhancement")

    assert len(rows) == 1
    assert rows[0].facet_text == "field enhancement"
    assert rows[0].decomposition_mode == "single_target"


def test_facet_metric_identity_ignores_source_expression_only_match():
    graph = nx.MultiDiGraph()
    _add_measurement(
        graph,
        mid="m:lspr",
        metric_id="lspr_wavelength",
        metric="LSPR wavelength",
        source_expression="hotspot intensity peak at 440 nm",
    )

    result = resolve_target_facet_operationalizations(
        graph=graph,
        requirements=_requirements(
            "electromagnetic hotspot location and intensity"
        ),
    )

    assert result.exact_supported_facet_count == 0
    assert all(
        row.status != "EXACT_METRIC_IDENTITY_SUPPORT"
        for row in result.resolutions[0].facets
    )


def test_exact_facet_metric_identity_is_detected_without_certification():
    graph = nx.MultiDiGraph()
    _add_measurement(
        graph,
        mid="m:intensity",
        metric_id="electromagnetic_hotspot_intensity",
        metric="Electromagnetic hotspot intensity",
        source_expression="hotspot intensity is high",
    )

    result = resolve_target_facet_operationalizations(
        graph=graph,
        requirements=_requirements(
            "electromagnetic hotspot location and intensity"
        ),
    )

    intensity = next(
        row
        for row in result.resolutions[0].facets
        if row.facet.facet_term == "intensity"
    )
    assert intensity.status == "EXACT_METRIC_IDENTITY_SUPPORT"
    assert intensity.target_facet_operationalization_verified is False
    assert result.target_operationalization_verified_count == 0


def test_facet_resolution_never_creates_authority():
    graph = nx.MultiDiGraph()

    result = resolve_target_facet_operationalizations(
        graph=graph,
        requirements=_requirements(
            "electromagnetic hotspot location and intensity"
        ),
    )

    assert result.domain_specific_synonymy_inferred is False
    assert result.decomposition_is_scientific_equivalence is False
    assert result.metric_identity_uses_source_expression is False
    assert result.target_operationalization_verified_count == 0
    assert result.operationalization_bridge_authority is False
    assert result.measurement_independence_verified_count == 0
    assert result.experiment_selection_performed is False
    assert result.rejection_authority is False
    assert result.production_selection_authority is False
    assert result.positive_premise_authority is False
    assert result.gap_authority is False
    assert result.novelty_authority is False
