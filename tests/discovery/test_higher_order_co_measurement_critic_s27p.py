from __future__ import annotations

import networkx as nx

from pipeline_core.discovery.higher_order_co_measurement_critic import (
    critique_co_measurement_candidates,
)
from pipeline_core.discovery.higher_order_co_measurement_resolver import (
    CoMeasurementContextCandidate,
    CoMeasurementRequirementResolution,
    CoMeasurementWitnessResolutionSet,
)
from pipeline_core.discovery.higher_order_operationalization_witness import (
    OperationalizationWitnessRequirement,
    OperationalizationWitnessRequirementSet,
)


def _requirements(target="hotspot intensity", anchor="field enhancement"):
    req = OperationalizationWitnessRequirement(
        requirement_id="req:1",
        source_experiment_id="exp:src",
        output_experiment_id="exp:out",
        issue_code="MEASUREMENT_INDEPENDENCE_UNRESOLVED",
        status="unresolved_requires_measurement_witness",
        requested_target_observable=target,
        anchor_observable=anchor,
        observable_pair=[target, anchor],
        required_graph_topology=["test"],
        minimum_candidate_evidence=["test"],
        independence_acceptance_rule="test",
        independence_rejection_rule="test",
    )
    return OperationalizationWitnessRequirementSet(
        repaired_experiment_count=1,
        requirement_count=1,
        candidate_inspiration_requirement_count=0,
        requirements=[req],
    )


def _graph(
    *,
    target_metric_id="unregistered_hotspot_intensity",
    target_metric="Hotspot intensity",
    anchor_metric_id="local_field_enhancement",
    anchor_metric="Local field enhancement",
    shared_provider_type=None,
    target_source="hotspot intensity",
    anchor_source="field enhancement",
):
    graph = nx.MultiDiGraph()
    graph.add_node(
        "m:t",
        type="Measurement",
        metric_id=target_metric_id,
        metric=target_metric,
        label=target_metric,
        source_expression=target_source,
        subject_id="sample:1",
    )
    graph.add_node(
        "m:a",
        type="Measurement",
        metric_id=anchor_metric_id,
        metric=anchor_metric,
        label=anchor_metric,
        source_expression=anchor_source,
        subject_id="sample:1",
    )

    if shared_provider_type:
        graph.add_node(
            "provider:shared",
            type=shared_provider_type,
        )
        graph.add_edge(
            "provider:shared",
            "m:t",
            relation="HAS_MEASUREMENT",
        )
        graph.add_edge(
            "provider:shared",
            "m:a",
            relation="HAS_MEASUREMENT",
        )
    else:
        graph.add_node("provider:t", type="Experiment")
        graph.add_node("provider:a", type="Experiment")
        graph.add_edge(
            "provider:t",
            "m:t",
            relation="HAS_MEASUREMENT",
        )
        graph.add_edge(
            "provider:a",
            "m:a",
            relation="HAS_MEASUREMENT",
        )

    return graph


def _resolution(graph, shared_provider=False):
    candidate = CoMeasurementContextCandidate(
        requirement_id="req:1",
        output_experiment_id="exp:out",
        source_paper_id="paper:1",
        target_candidate_id="cand:t",
        anchor_candidate_id="cand:a",
        target_measurement_node_id="m:t",
        anchor_measurement_node_id="m:a",
        target_metric_id=str(graph.nodes["m:t"]["metric_id"]),
        anchor_metric_id=str(graph.nodes["m:a"]["metric_id"]),
        target_subject_id="sample:1",
        anchor_subject_id="sample:1",
        same_subject_id=True,
        target_provider_ids=(
            ["provider:shared"]
            if shared_provider
            else ["provider:t"]
        ),
        anchor_provider_ids=(
            ["provider:shared"]
            if shared_provider
            else ["provider:a"]
        ),
        shared_provider_ids=(
            ["provider:shared"]
            if shared_provider
            else []
        ),
        shared_provider_present=shared_provider,
        target_lexical_coverage=1.0,
        anchor_lexical_coverage=1.0,
        context_strength=(
            "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC"
            if shared_provider
            else "SAME_SUBJECT_DISTINCT_METRIC"
        ),
    )
    row = CoMeasurementRequirementResolution(
        requirement_id="req:1",
        output_experiment_id="exp:out",
        status=(
            "SAME_PROVIDER_AND_SUBJECT_DISTINCT_METRIC_CANDIDATE"
            if shared_provider
            else "SAME_SUBJECT_DISTINCT_METRIC_CANDIDATE"
        ),
        target_candidate_count=1,
        anchor_candidate_count=1,
        shared_source_paper_count=1,
        distinct_metric_pair_count=1,
        same_subject_pair_count=1,
        shared_provider_pair_count=int(shared_provider),
        same_provider_and_subject_pair_count=int(shared_provider),
        context_strength_counts={candidate.context_strength: 1},
        candidates=[candidate],
        co_measurement_witness_required=True,
    )
    return CoMeasurementWitnessResolutionSet(
        requirement_count=1,
        routed_requirement_count=1,
        resolution_count=1,
        status_counts={row.status: 1},
        total_candidate_count=1,
        same_subject_candidate_count=1,
        shared_provider_candidate_count=int(shared_provider),
        same_provider_and_subject_candidate_count=int(shared_provider),
        resolutions=[row],
    )


def test_exact_metric_identity_separate_provider_candidate_is_followup_only():
    graph = _graph()
    report = critique_co_measurement_candidates(
        graph=graph,
        requirements=_requirements(),
        resolutions=_resolution(graph, shared_provider=False),
    )
    row = report.requirements[0].candidates[0]

    assert row.target_metric_identity_coverage == 1.0
    assert row.anchor_metric_identity_coverage == 1.0
    assert row.exact_metric_identity_on_both_sides is True
    assert row.eligible_for_independence_followup is True
    assert row.measurement_independence_verified is False


def test_shared_calculation_is_dependency_unresolved_not_followup():
    graph = _graph(shared_provider_type="Calculation")
    report = critique_co_measurement_candidates(
        graph=graph,
        requirements=_requirements(),
        resolutions=_resolution(graph, shared_provider=True),
    )
    row = report.requirements[0].candidates[0]

    assert "SHARED_CALCULATION_DEPENDENCY_UNRESOLVED" in row.issues
    assert row.eligible_for_independence_followup is False


def test_metric_identity_ignores_source_expression_false_positive():
    graph = _graph(
        anchor_metric_id="lspr_wavelength",
        anchor_metric="LSPR wavelength",
        anchor_source="field enhancement peak was found at 440 nm",
    )
    report = critique_co_measurement_candidates(
        graph=graph,
        requirements=_requirements(),
        resolutions=_resolution(graph, shared_provider=False),
    )
    row = report.requirements[0].candidates[0]

    assert row.anchor_metric_identity_coverage == 0.0
    assert "ANCHOR_METRIC_IDENTITY_UNRESOLVED" in row.issues
    assert row.eligible_for_independence_followup is False


def test_explicit_derivation_blocks_followup():
    graph = _graph()
    graph.add_edge(
        "m:a",
        "m:t",
        relation="DERIVED_FROM",
    )
    report = critique_co_measurement_candidates(
        graph=graph,
        requirements=_requirements(),
        resolutions=_resolution(graph, shared_provider=False),
    )
    row = report.requirements[0].candidates[0]

    assert row.direct_derivation_present is True
    assert "EXPLICIT_MEASUREMENT_DERIVATION_PRESENT" in row.issues
    assert row.eligible_for_independence_followup is False


def test_critic_never_creates_authority():
    graph = _graph()
    report = critique_co_measurement_candidates(
        graph=graph,
        requirements=_requirements(),
        resolutions=_resolution(graph, shared_provider=False),
    )

    assert report.diagnostic_only is True
    assert report.candidate_is_independence_witness is False
    assert report.measurement_independence_verified_count == 0
    assert report.independence_certification_authority is False
    assert report.experiment_selection_performed is False
    assert report.rejection_authority is False
    assert report.production_selection_authority is False
    assert report.positive_premise_authority is False
    assert report.gap_authority is False
    assert report.novelty_authority is False
