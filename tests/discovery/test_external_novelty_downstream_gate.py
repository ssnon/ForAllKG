from __future__ import annotations

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyCard,
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    HypothesisSearchCoverage,
)
from pipeline_core.discovery.external_novelty_downstream_gate import (
    ExternalNoveltyDownstreamGate,
)
from pipeline_core.discovery.prior_art_coverage_probe import (
    PreReviewCoverageReport,
    PreReviewHypothesisCoverage,
)


def _coverage(
    *,
    sufficient: bool,
) -> HypothesisSearchCoverage:
    return HypothesisSearchCoverage(
        hypothesis_id="hypothesis:test",
        query_count=5,
        successful_query_count=5,
        provider_success_count=2,
        unique_work_count=50,
        abstract_work_count=25,
        core_claim_count=1,
        core_claims_with_minimum_abstract_coverage=(
            1 if sufficient else 0
        ),
        sufficient_for_absence_based_novelty=sufficient,
    )


def _pre_review(
    *,
    sufficient: bool,
) -> PreReviewCoverageReport:
    coverage = _coverage(sufficient=sufficient)
    row = PreReviewHypothesisCoverage(
        hypothesis_id="hypothesis:test",
        coverage=coverage,
        claim_coverages=[],
        core_claim_ids=[],
        gate_decision=(
            "KEEP_FOR_CLAIM_REVIEW"
            if sufficient
            else "EVIDENCE_REQUIRED"
        ),
        reason_codes=(
            []
            if sufficient
            else [
                "one_or_more_core_claims_below_abstract_minimum"
            ]
        ),
    )
    return PreReviewCoverageReport(
        source_portfolio_id="portfolio:test",
        source_query_plan_id="plan:test",
        source_query_plan_sha256="plan-sha",
        source_prior_art_packet_id="packet:test",
        source_prior_art_packet_sha256="packet-sha",
        policy=ExternalNoveltyPolicy(),
        hypotheses=[row],
        keep_for_claim_review_count=(
            1 if sufficient else 0
        ),
        evidence_required_count=(
            0 if sufficient else 1
        ),
    )


def _external_report(
    *,
    sufficient: bool,
    status: str,
    reason_codes: list[str],
) -> ExternalNoveltyReport:
    coverage = _coverage(sufficient=sufficient)

    card = ExternalNoveltyCard.model_construct(
        hypothesis_id="hypothesis:test",
        title="Test",
        status=status,
        claim_reviews=[],
        coverage=coverage,
        strongest_prior_art_work_ids=[],
        contextual_conflict_work_ids=[],
        lower_order_prior_art_work_ids=[],
        lower_order_supported_core_claim_ids=[],
        higher_order_relational_gap_claim_ids=[],
        lower_order_core_prior_art_work_ids=[],
        lower_order_core_unique_work_count=0,
        relational_gap_kind="NONE",
        directional_counterevidence_work_ids=[],
        discovery_axis_id=None,
        discovery_inspiration_id=None,
        reason_codes=reason_codes,
        interpretation="fixture",
        search_limitations=[],
    )

    return ExternalNoveltyReport.model_construct(
        report_id="external_novelty_report:test",
        report_sha256="external-report-sha",
        source_portfolio_id="portfolio:test",
        source_prior_art_packet_id="packet:test",
        searched_at_utc="2026-09-15T00:00:00+00:00",
        cards=[card],
        status_counts={status: 1},
        policy={},
        external_novelty_claim_scope=(
            "search-bounded_prior-art-assessment-not-proof"
        ),
        epistemic_usage="prior_art_only_not_positive_premise",
    )


def test_gate_holds_exact_absence_coverage_failure() -> None:
    external = _external_report(
        sufficient=False,
        status="INSUFFICIENT_SEARCH_EVIDENCE",
        reason_codes=[
            "insufficient_coverage_for_absence_based_status"
        ],
    )
    pre = _pre_review(sufficient=False)

    result = ExternalNoveltyDownstreamGate().build(
        external_report=external,
        pre_review_report=pre,
    )

    assert result.continue_count == 0
    assert result.hold_for_evidence_count == 1

    row = result.dispositions[0]
    assert row.decision == "HOLD_FOR_EVIDENCE"
    assert row.scientific_distinctiveness_authorized is False
    assert row.semantic_distinctiveness_authorized is False
    assert row.n10_review_authorized is False
    assert row.evidence_acquisition_required is True
    assert row.novelty_authority is False
    assert row.selection_class_assigned is False
    assert row.production_selection_changed is False


@pytest.mark.parametrize(
    "status",
    [
        "WELL_ESTABLISHED",
        "CONFLICTING_PRIOR_ART",
        "LITERATURE_SUPPORTED_EXTENSION",
    ],
)
def test_gate_preserves_positive_or_conflicting_prior_art_path(
    status: str,
) -> None:
    external = _external_report(
        sufficient=False,
        status=status,
        reason_codes=[],
    )
    pre = _pre_review(sufficient=False)

    result = ExternalNoveltyDownstreamGate().build(
        external_report=external,
        pre_review_report=pre,
    )

    row = result.dispositions[0]
    assert row.decision == "CONTINUE_DOWNSTREAM"
    assert row.semantic_distinctiveness_authorized is True
    assert row.n10_review_authorized is True
    assert (
        "reviewer_found_actionable_positive_or_conflicting_prior_art"
        in row.reason_codes
    )


def test_gate_does_not_hold_noncoverage_insufficient_status() -> None:
    external = _external_report(
        sufficient=False,
        status="INSUFFICIENT_SEARCH_EVIDENCE",
        reason_codes=[
            "core_claim_title_only_unresolved"
        ],
    )
    pre = _pre_review(sufficient=False)

    result = ExternalNoveltyDownstreamGate().build(
        external_report=external,
        pre_review_report=pre,
    )

    row = result.dispositions[0]
    assert row.decision == "CONTINUE_DOWNSTREAM"
    assert (
        "insufficient_status_not_caused_by_absence_coverage_gate"
        in row.reason_codes
    )


def test_gate_continues_when_coverage_is_sufficient() -> None:
    external = _external_report(
        sufficient=True,
        status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        reason_codes=[],
    )
    pre = _pre_review(sufficient=True)

    result = ExternalNoveltyDownstreamGate().build(
        external_report=external,
        pre_review_report=pre,
    )

    row = result.dispositions[0]
    assert row.decision == "CONTINUE_DOWNSTREAM"
    assert row.scientific_distinctiveness_authorized is True
    assert row.semantic_distinctiveness_authorized is True
    assert row.n10_review_authorized is True


def test_gate_rejects_coverage_drift() -> None:
    external = _external_report(
        sufficient=True,
        status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        reason_codes=[],
    )
    pre = _pre_review(sufficient=False)

    with pytest.raises(
        ValueError,
        match="coverage mismatch",
    ):
        ExternalNoveltyDownstreamGate().build(
            external_report=external,
            pre_review_report=pre,
        )


def test_gate_never_assigns_novelty_or_selection_authority() -> None:
    external = _external_report(
        sufficient=False,
        status="INSUFFICIENT_SEARCH_EVIDENCE",
        reason_codes=[
            "insufficient_coverage_for_absence_based_status"
        ],
    )
    pre = _pre_review(sufficient=False)

    result = ExternalNoveltyDownstreamGate().build(
        external_report=external,
        pre_review_report=pre,
    )

    assert result.novelty_authority is False
    assert result.selection_class_assigned is False
    assert result.production_selection_changed is False

    for row in result.dispositions:
        assert row.novelty_authority is False
        assert row.selection_class_assigned is False
        assert row.production_selection_changed is False
