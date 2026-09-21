from __future__ import annotations

from pipeline_core.discovery.higher_order_operationalization_resolver import (
    MeasurementOperationalizationCandidate,
    MeasurementProviderCandidate,
    OperationalizationPairCandidate,
    OperationalizationWitnessResolution,
    OperationalizationWitnessResolutionSet,
)
from pipeline_core.discovery.higher_order_operationalization_critic import (
    critique_operationalization_witnesses,
)


def _provider(pid: str, paper: str):
    return MeasurementProviderCandidate(
        provider_id=pid,
        provider_type="Calculation",
        label="test",
        method_identity="FDTD",
        method_identity_explicit=True,
        source_paper_id=paper,
    )


def _candidate(
    *,
    cid: str,
    role: str,
    metric_id: str,
    coverage: float,
    paper: str,
):
    return MeasurementOperationalizationCandidate(
        candidate_id=cid,
        role=role,
        observable="observable",
        measurement_node_id="m:" + cid,
        metric_id=metric_id,
        metric=metric_id,
        source_expression=metric_id,
        description="",
        subject_id="subject",
        lexical_coverage=coverage,
        matched_token_count=1,
        observable_token_count=1,
        providers=[_provider("p:" + cid, paper)],
        provider_count=1,
        provenance_present=True,
        source_paper_ids=[paper],
    )


def _pair(target, anchor):
    return OperationalizationPairCandidate(
        target_candidate_id=target.candidate_id,
        anchor_candidate_id=anchor.candidate_id,
        target_measurement_node_id=target.measurement_node_id,
        anchor_measurement_node_id=anchor.measurement_node_id,
        target_provider_ids=[target.providers[0].provider_id],
        anchor_provider_ids=[anchor.providers[0].provider_id],
        distinct_provider_ids_present=True,
        explicit_method_identity_present_on_both_sides=True,
        distinct_method_identity_present=True,
        provenance_present_on_both_sides=True,
        classification="SUPPORTED_DISTINCT_OPERATIONALIZATION",
    )


def _resolution(targets, anchors, pairs):
    return OperationalizationWitnessResolution(
        requirement_id="req:1",
        output_experiment_id="exp:1",
        requested_target_observable="target",
        anchor_observable="anchor",
        status="SUPPORTED_DISTINCT_OPERATIONALIZATION",
        target_candidates=targets,
        anchor_candidates=anchors,
        pair_candidates=pairs,
        target_candidate_count=len(targets),
        anchor_candidate_count=len(anchors),
        pair_candidate_count=len(pairs),
    )


def _report(row):
    return critique_operationalization_witnesses(
        OperationalizationWitnessResolutionSet(
            requirement_count=1,
            resolution_count=1,
            status_counts={
                "SUPPORTED_DISTINCT_OPERATIONALIZATION": 1
            },
            measurement_candidate_count=(
                row.target_candidate_count
                + row.anchor_candidate_count
            ),
            pair_candidate_count=row.pair_candidate_count,
            distinct_operationalization_resolution_count=1,
            graph_node_count=1,
            graph_edge_count=1,
            resolutions=[row],
        )
    )


def _codes(report):
    return {
        issue.code
        for issue in report.resolutions[0].issues
    }


def test_cross_paper_distinct_methods_are_not_independence_witness():
    target = _candidate(
        cid="t", role="target", metric_id="field",
        coverage=1.0, paper="paper:t",
    )
    anchor = _candidate(
        cid="a", role="anchor", metric_id="sers",
        coverage=1.0, paper="paper:a",
    )
    report = _report(
        _resolution([target], [anchor], [_pair(target, anchor)])
    )

    assert "CO_MEASUREMENT_CONTEXT_UNRESOLVED" in _codes(report)


def test_same_metric_pair_collapse_is_flagged():
    target = _candidate(
        cid="t", role="target", metric_id="field",
        coverage=1.0, paper="paper:1",
    )
    anchor = _candidate(
        cid="a", role="anchor", metric_id="field",
        coverage=1.0, paper="paper:1",
    )
    report = _report(
        _resolution([target], [anchor], [_pair(target, anchor)])
    )

    assert "TARGET_ANCHOR_METRIC_COLLAPSE" in _codes(report)


def test_partial_target_coverage_and_anchor_heterogeneity_are_diagnostic():
    target = _candidate(
        cid="t", role="target", metric_id="field",
        coverage=0.75, paper="paper:1",
    )
    anchor1 = _candidate(
        cid="a1", role="anchor", metric_id="field",
        coverage=1.0, paper="paper:1",
    )
    anchor2 = _candidate(
        cid="a2", role="anchor", metric_id="sers",
        coverage=1.0, paper="paper:1",
    )
    report = _report(
        _resolution(
            [target],
            [anchor1, anchor2],
            [_pair(target, anchor1), _pair(target, anchor2)],
        )
    )
    codes = _codes(report)

    assert "TARGET_OBSERVABLE_COVERAGE_UNRESOLVED" in codes
    assert "ANCHOR_METRIC_HETEROGENEITY" in codes


def test_critic_never_certifies_or_rejects():
    target = _candidate(
        cid="t", role="target", metric_id="field",
        coverage=1.0, paper="paper:1",
    )
    anchor = _candidate(
        cid="a", role="anchor", metric_id="sers",
        coverage=1.0, paper="paper:1",
    )
    report = _report(
        _resolution([target], [anchor], [_pair(target, anchor)])
    )

    assert report.diagnostic_only is True
    assert report.measurement_independence_verified_count == 0
    assert report.independence_certification_authority is False
    assert report.experiment_selection_performed is False
    assert report.rejection_authority is False
    assert report.production_selection_authority is False
    assert report.positive_premise_authority is False
    assert report.gap_authority is False
    assert report.novelty_authority is False
