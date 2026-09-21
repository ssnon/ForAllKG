from __future__ import annotations

import networkx as nx

from pipeline_core.discovery.higher_order_co_measurement_resolver import (
    resolve_co_measurement_witness_candidates,
)
from pipeline_core.discovery.higher_order_operationalization_adjudicator import (
    OperationalizationWitnessAdjudication,
    OperationalizationWitnessAdjudicationSet,
)
from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirement,
    OperationalizationWitnessRequirementSet,
)


def _requirements():
    row = OperationalizationWitnessRequirement(
        requirement_id="req:1",
        source_experiment_id="exp:src",
        output_experiment_id="exp:out",
        issue_code="MEASUREMENT_INDEPENDENCE_UNRESOLVED",
        status="unresolved_requires_measurement_witness",
        requested_target_observable="hotspot intensity",
        anchor_observable="SERS enhancement factor",
        observable_pair=[
            "hotspot intensity",
            "SERS enhancement factor",
        ],
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


def _adjudication(routed=True):
    routes = (
        ["CO_MEASUREMENT_WITNESS_REQUIRED"]
        if routed
        else []
    )
    row = OperationalizationWitnessAdjudication(
        requirement_id="req:1",
        output_experiment_id="exp:out",
        raw_resolver_status="SUPPORTED_DISTINCT_OPERATIONALIZATION",
        candidate_support_state="CROSS_PAPER_DISTINCT_ONLY",
        raw_supported_status_demoted=True,
        target_best_lexical_coverage=1.0,
        anchor_dominant_metric_fraction=1.0,
        same_metric_pair_fraction=0.0,
        cross_paper_pair_fraction=1.0,
        same_paper_distinct_metric_pair_count=0,
        next_routes=routes,
        issue_codes=[],
    )
    return OperationalizationWitnessAdjudicationSet(
        resolution_count=1,
        adjudication_count=1,
        raw_supported_resolution_count=1,
        demoted_raw_supported_resolution_count=1,
        co_measurement_candidate_resolution_count=0,
        cross_paper_only_resolution_count=1,
        route_counts=(
            {"CO_MEASUREMENT_WITNESS_REQUIRED": 1}
            if routed
            else {}
        ),
        adjudications=[row],
    )


def _add_measurement(
    graph,
    *,
    mid,
    metric,
    metric_id,
    paper,
    subject,
    provider,
):
    graph.add_node(
        provider,
        type="Experiment",
        label=provider,
        method_label="method",
        raw_method_name="method",
        experiment_family="test",
        source_paper_id=paper,
    )
    graph.add_node(
        mid,
        type="Measurement",
        metric=metric,
        metric_id=metric_id,
        label=metric,
        source_expression=metric,
        description="",
        subject_id=subject,
        source_paper_id=paper,
    )
    graph.add_edge(
        provider,
        mid,
        relation="HAS_MEASUREMENT",
        paper_id=paper,
    )


def test_same_paper_distinct_metric_candidate_is_found():
    g = nx.MultiDiGraph()
    _add_measurement(
        g,
        mid="m:t",
        metric="hotspot intensity",
        metric_id="local_field_enhancement",
        paper="paper:1",
        subject="sample:1",
        provider="exp:t",
    )
    _add_measurement(
        g,
        mid="m:a",
        metric="SERS enhancement factor",
        metric_id="sers_enhancement_factor",
        paper="paper:1",
        subject="sample:2",
        provider="exp:a",
    )

    result = resolve_co_measurement_witness_candidates(
        graph=g,
        requirements=_requirements(),
        adjudication=_adjudication(),
    )

    row = result.resolutions[0]
    assert row.status == "SAME_PAPER_DISTINCT_METRIC_CANDIDATE"
    assert row.distinct_metric_pair_count == 1
    assert row.same_subject_pair_count == 0
    assert row.shared_provider_pair_count == 0


def test_same_subject_and_provider_are_recorded_but_not_certified():
    g = nx.MultiDiGraph()
    _add_measurement(
        g,
        mid="m:t",
        metric="hotspot intensity",
        metric_id="local_field_enhancement",
        paper="paper:1",
        subject="sample:1",
        provider="exp:shared",
    )
    _add_measurement(
        g,
        mid="m:a",
        metric="SERS enhancement factor",
        metric_id="sers_enhancement_factor",
        paper="paper:1",
        subject="sample:1",
        provider="exp:shared",
    )

    result = resolve_co_measurement_witness_candidates(
        graph=g,
        requirements=_requirements(),
        adjudication=_adjudication(),
    )

    row = result.resolutions[0]
    assert row.status == (
        "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC_CANDIDATE"
    )
    assert row.same_provider_and_subject_pair_count == 1
    candidate = row.candidates[0]
    assert candidate.candidate_is_independence_witness is False
    assert candidate.measurement_independence_verified is False


def test_same_metric_is_excluded_even_in_same_context():
    g = nx.MultiDiGraph()
    _add_measurement(
        g,
        mid="m:t",
        metric="hotspot intensity",
        metric_id="local_field_enhancement",
        paper="paper:1",
        subject="sample:1",
        provider="exp:shared",
    )
    _add_measurement(
        g,
        mid="m:a",
        metric="SERS enhancement factor",
        metric_id="local_field_enhancement",
        paper="paper:1",
        subject="sample:1",
        provider="exp:shared",
    )

    result = resolve_co_measurement_witness_candidates(
        graph=g,
        requirements=_requirements(),
        adjudication=_adjudication(),
    )

    row = result.resolutions[0]
    assert row.status == "NO_CO_MEASUREMENT_CANDIDATE"
    assert row.distinct_metric_pair_count == 0


def test_not_routed_requirement_is_not_searched_or_promoted():
    g = nx.MultiDiGraph()

    result = resolve_co_measurement_witness_candidates(
        graph=g,
        requirements=_requirements(),
        adjudication=_adjudication(routed=False),
    )

    row = result.resolutions[0]
    assert row.status == "NOT_ROUTED_FOR_CO_MEASUREMENT"
    assert row.co_measurement_witness_required is False
    assert result.measurement_independence_verified_count == 0
    assert result.independence_certification_authority is False
    assert result.production_selection_authority is False
    assert result.positive_premise_authority is False
    assert result.gap_authority is False
    assert result.novelty_authority is False
