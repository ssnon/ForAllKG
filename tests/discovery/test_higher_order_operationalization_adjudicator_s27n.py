from __future__ import annotations

from pipeline_core.discovery.higher_order_operationalization_adjudicator import (
    adjudicate_operationalization_witnesses,
)
from pipeline_core.discovery.higher_order_operationalization_critic import (
    OperationalizationCriticIssue,
    OperationalizationResolutionCritique,
    OperationalizationWitnessCriticReport,
)
from pipeline_core.discovery.higher_order_operationalization_resolver import (
    OperationalizationWitnessResolution,
    OperationalizationWitnessResolutionSet,
)


def _resolution(*, pair_count: int = 1):
    return OperationalizationWitnessResolution(
        requirement_id="req:1",
        output_experiment_id="exp:1",
        requested_target_observable="target",
        anchor_observable="anchor",
        status="SUPPORTED_DISTINCT_OPERATIONALIZATION",
        target_candidates=[],
        anchor_candidates=[],
        pair_candidates=[],
        target_candidate_count=0,
        anchor_candidate_count=0,
        pair_candidate_count=pair_count,
    )


def _resolution_set(row):
    return OperationalizationWitnessResolutionSet(
        requirement_count=1,
        resolution_count=1,
        status_counts={"SUPPORTED_DISTINCT_OPERATIONALIZATION": 1},
        measurement_candidate_count=0,
        pair_candidate_count=row.pair_candidate_count,
        distinct_operationalization_resolution_count=1,
        graph_node_count=1,
        graph_edge_count=1,
        resolutions=[row],
    )


def _critique(
    *,
    issues,
    same_paper_distinct=0,
    same_metric=0.0,
    cross_paper=1.0,
):
    row = OperationalizationResolutionCritique(
        requirement_id="req:1",
        output_experiment_id="exp:1",
        resolver_status="SUPPORTED_DISTINCT_OPERATIONALIZATION",
        target_best_lexical_coverage=0.75,
        anchor_dominant_metric_fraction=0.95,
        same_metric_pair_fraction=same_metric,
        cross_paper_pair_fraction=cross_paper,
        same_paper_distinct_metric_pair_count=same_paper_distinct,
        issues=[
            OperationalizationCriticIssue(
                code=code,
                message="test",
            )
            for code in issues
        ],
    )
    return OperationalizationWitnessCriticReport(
        resolution_count=1,
        flagged_resolution_count=int(bool(issues)),
        issue_counts={code: 1 for code in issues},
        resolutions=[row],
    )


def test_raw_supported_cross_paper_only_is_demoted():
    result = adjudicate_operationalization_witnesses(
        resolutions=_resolution_set(_resolution(pair_count=5)),
        critique=_critique(
            issues=["CO_MEASUREMENT_CONTEXT_UNRESOLVED"],
        ),
    )

    row = result.adjudications[0]
    assert row.candidate_support_state == "CROSS_PAPER_DISTINCT_ONLY"
    assert row.raw_supported_status_demoted is True
    assert "CO_MEASUREMENT_WITNESS_REQUIRED" in row.next_routes
    assert result.demoted_raw_supported_resolution_count == 1


def test_same_paper_distinct_metric_candidate_is_preserved_as_candidate_only():
    result = adjudicate_operationalization_witnesses(
        resolutions=_resolution_set(_resolution(pair_count=2)),
        critique=_critique(
            issues=[],
            same_paper_distinct=1,
            cross_paper=0.5,
        ),
    )

    row = result.adjudications[0]
    assert row.candidate_support_state == (
        "CO_MEASUREMENT_DISTINCT_METRIC_CANDIDATE_PRESENT"
    )
    assert row.raw_supported_status_demoted is False
    assert row.candidate_support_is_certification is False
    assert row.measurement_independence_verified is False


def test_metric_collapse_and_anchor_heterogeneity_route_separately():
    result = adjudicate_operationalization_witnesses(
        resolutions=_resolution_set(_resolution(pair_count=8)),
        critique=_critique(
            issues=[
                "ANCHOR_METRIC_HETEROGENEITY",
                "TARGET_ANCHOR_METRIC_COLLAPSE",
            ],
            same_metric=0.75,
        ),
    )

    routes = set(result.adjudications[0].next_routes)
    assert "ANCHOR_METRIC_DISAMBIGUATION_REQUIRED" in routes
    assert "TARGET_ANCHOR_METRIC_DISENTANGLEMENT_REQUIRED" in routes


def test_adjudicator_never_creates_authority():
    result = adjudicate_operationalization_witnesses(
        resolutions=_resolution_set(_resolution(pair_count=1)),
        critique=_critique(
            issues=[
                "TARGET_OBSERVABLE_COVERAGE_UNRESOLVED",
                "CO_MEASUREMENT_CONTEXT_UNRESOLVED",
            ],
        ),
    )

    assert result.diagnostic_only is True
    assert result.resolver_retrieval_mutated is False
    assert result.measurement_independence_verified_count == 0
    assert result.independence_certification_authority is False
    assert result.experiment_selection_performed is False
    assert result.rejection_authority is False
    assert result.production_selection_authority is False
    assert result.positive_premise_authority is False
    assert result.gap_authority is False
    assert result.novelty_authority is False
