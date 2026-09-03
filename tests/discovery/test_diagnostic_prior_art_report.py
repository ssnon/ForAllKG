from __future__ import annotations

from pipeline_core.discovery.diagnostic_prior_art_report import (
    build_diagnostic_review_report,
)
from pipeline_core.discovery.diagnostic_prior_art_review import (
    DiagnosticClaimPriorArtReview,
    DiagnosticPriorArtMatch,
)


def _review(
    *,
    claim_id: str,
    signal: bool,
) -> DiagnosticClaimPriorArtReview:
    match = DiagnosticPriorArtMatch(
        work_id="work:control",
        relationship=(
            "LOWER_ORDER_RELATION_PRIOR_ART"
        ),
        confidence=0.95,
        rationale="test",
        relevance_score=0.9,
        semantic_similarity=0.9,
        lexical_coverage=0.8,
        reaction_domain_relevance=0.9,
        catalyst_scope_relevance=0.9,
        title="Control relation",
        abstract_available=True,
    )

    return DiagnosticClaimPriorArtReview(
        hypothesis_id="hypothesis:test",
        claim_id=claim_id,
        claim_text="higher-order claim",
        diagnostic_query_kind=(
            "LOWER_ORDER_RELATION"
        ),
        diagnostic_execution_query=(
            "X Y dependence"
        ),
        matches=[
            match
        ],
        signal_work_ids=(
            ["work:control"]
            if signal
            else []
        ),
        reason_codes=[],
        reviewer_unknown_work_ids=[],
        interpretation="test",
    )


def test_report_preserves_source_provenance_and_signals() -> None:
    report = build_diagnostic_review_report(
        source_portfolio_id=(
            "hypothesis_portfolio:test"
        ),
        source_query_plan_id=(
            "literature_query_plan:diag"
        ),
        source_query_plan_sha256=(
            "plan-sha"
        ),
        source_prior_art_packet_id=(
            "prior_art_packet:diag"
        ),
        source_prior_art_packet_sha256=(
            "packet-sha"
        ),
        reviews=[
            _review(
                claim_id="claim:b",
                signal=False,
            ),
            _review(
                claim_id="claim:a",
                signal=True,
            ),
        ],
    )

    assert (
        report.reviewed_claim_count
        == 2
    )

    assert (
        report.signal_claim_count
        == 1
    )

    assert (
        report.signal_work_count
        == 1
    )

    assert [
        row.claim_id
        for row in report.reviews
    ] == [
        "claim:a",
        "claim:b",
    ]

    assert (
        report.shadow_only
        is True
    )

    assert (
        report.scientific_selection_changed
        is False
    )


def test_report_is_deterministic() -> None:
    kwargs = dict(
        source_portfolio_id=(
            "hypothesis_portfolio:test"
        ),
        source_query_plan_id=(
            "literature_query_plan:diag"
        ),
        source_query_plan_sha256=(
            "plan-sha"
        ),
        source_prior_art_packet_id=(
            "prior_art_packet:diag"
        ),
        source_prior_art_packet_sha256=(
            "packet-sha"
        ),
        reviews=[
            _review(
                claim_id="claim:a",
                signal=True,
            ),
        ],
    )

    left = (
        build_diagnostic_review_report(
            **kwargs
        )
    )

    right = (
        build_diagnostic_review_report(
            **kwargs
        )
    )

    assert (
        left.model_dump(mode="json")
        == right.model_dump(mode="json")
    )
