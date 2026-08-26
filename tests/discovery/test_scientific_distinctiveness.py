from __future__ import annotations

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    ExternalNoveltyCard,
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    HypothesisNoveltyClaims,
    HypothesisSearchCoverage,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtMatch,
    PriorArtPacket,
    PriorArtWork,
)
from pipeline_core.discovery.scientific_distinctiveness import (
    ScientificDistinctivenessAnalyzer,
)


def _fixture():
    plan = LiteratureQueryPlan(
        plan_id="plan:test",
        plan_sha256="plan-sha",
        source_portfolio_id="portfolio:test",
        queries=[
            LiteratureQuery(
                query_id="query:c1",
                hypothesis_id="hypothesis:test",
                claim_id="claim:c1",
                query_kind="claim_primary",
                query_text="lower-order relation query",
            ),
            LiteratureQuery(
                query_id="query:c2",
                hypothesis_id="hypothesis:test",
                claim_id="claim:c2",
                query_kind="claim_primary",
                query_text="second relation query",
            ),
        ],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:test",
                title="Test hypothesis",
                claims=[
                    NoveltyClaim(
                        claim_id="claim:c1",
                        hypothesis_id="hypothesis:test",
                        claim_rank=1,
                        kind="moderator_interaction",
                        importance="core",
                        text=(
                            "A and B jointly alter response Y"
                        ),
                        rationale="Core interaction claim.",
                        search_concepts=[
                            "A",
                            "B",
                            "Y",
                        ],
                        search_queries=[
                            "A B Y"
                        ],
                    ),
                    NoveltyClaim(
                        claim_id="claim:c2",
                        hypothesis_id="hypothesis:test",
                        claim_rank=2,
                        kind="distinctive_prediction",
                        importance="core",
                        text=(
                            "The joint regime changes response ordering"
                        ),
                        rationale="Core prediction claim.",
                        search_concepts=[
                            "joint regime",
                            "ordering",
                        ],
                        search_queries=[
                            "joint regime response ordering"
                        ],
                    ),
                ],
            )
        ],
    )

    packet = PriorArtPacket(
        packet_id="packet:test",
        packet_sha256="packet-sha",
        source_portfolio_id="portfolio:test",
        source_query_plan_id="plan:test",
        searched_at_utc="2026-01-01T00:00:00+00:00",
        providers_requested=[
            "fixture"
        ],
        works=[
            PriorArtWork(
                work_id="work:lower",
                title="Known lower-order relation",
                abstract=(
                    "A lower-order relation is reported."
                ),
                retrieval_query_ids=[
                    "query:c1"
                ],
                retrieval_claim_ids=[
                    "claim:c1"
                ],
            )
        ],
        raw_work_count=1,
        canonical_work_count=1,
    )

    lower_match = PriorArtMatch(
        work_id="work:lower",
        relationship=(
            "LOWER_ORDER_RELATION_PRIOR_ART"
        ),
        confidence=0.90,
        rationale="Supports a lower-order subrelation.",
        relevance_score=0.90,
        semantic_similarity=0.85,
        lexical_coverage=0.80,
        abstract_available=True,
        title="Known lower-order relation",
    )

    review_c1 = ClaimPriorArtReview(
        hypothesis_id="hypothesis:test",
        claim_id="claim:c1",
        claim_text="A and B jointly alter response Y",
        importance="core",
        status="COMPONENTS_ONLY",
        matches=[
            lower_match
        ],
        coverage=ClaimSearchCoverage(
            claim_id="claim:c1",
            query_count=1,
            successful_query_count=1,
            unique_work_count=1,
            abstract_work_count=1,
            reviewed_work_count=1,
        ),
        reason_codes=[],
        interpretation=(
            "Lower-order relation exists; full interaction "
            "was not directly reconstructed."
        ),
    )

    review_c2 = ClaimPriorArtReview(
        hypothesis_id="hypothesis:test",
        claim_id="claim:c2",
        claim_text=(
            "The joint regime changes response ordering"
        ),
        importance="core",
        status="NO_DIRECT_MATCH_FOUND",
        matches=[],
        coverage=ClaimSearchCoverage(
            claim_id="claim:c2",
            query_count=1,
            successful_query_count=1,
            unique_work_count=10,
            abstract_work_count=5,
            reviewed_work_count=5,
        ),
        reason_codes=[],
        interpretation=(
            "No direct match in the frozen reviewed set."
        ),
    )

    card = ExternalNoveltyCard(
        hypothesis_id="hypothesis:test",
        title="Test hypothesis",
        status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        claim_reviews=[
            review_c1,
            review_c2,
        ],
        coverage=HypothesisSearchCoverage(
            hypothesis_id="hypothesis:test",
            query_count=2,
            successful_query_count=2,
            provider_success_count=1,
            unique_work_count=10,
            abstract_work_count=5,
            core_claim_count=2,
            core_claims_with_minimum_abstract_coverage=2,
            sufficient_for_absence_based_novelty=True,
        ),
        lower_order_prior_art_work_ids=[
            "work:lower"
        ],
        lower_order_supported_core_claim_ids=[
            "claim:c1"
        ],
        higher_order_relational_gap_claim_ids=[
            "claim:c1"
        ],
        lower_order_core_prior_art_work_ids=[
            "work:lower"
        ],
        lower_order_core_unique_work_count=1,
        relational_gap_kind=(
            "HIGHER_ORDER_RELATIONAL_GAP"
        ),
        reason_codes=[
            "lower_order_relation_prior_art_present",
            "higher_order_relational_gap_present",
        ],
        interpretation=(
            "Frozen external novelty fixture."
        ),
    )

    report = ExternalNoveltyReport(
        report_id="external:test",
        report_sha256="external-sha",
        source_portfolio_id="portfolio:test",
        source_prior_art_packet_id="packet:test",
        searched_at_utc="2026-01-01T00:00:00+00:00",
        cards=[
            card
        ],
        status_counts={
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP":
                1
        },
        policy=ExternalNoveltyPolicy(),
    )

    return (
        report,
        plan,
        packet,
    )


def test_distinctiveness_reuses_frozen_horg_evidence() -> None:
    report, plan, packet = _fixture()

    result = (
        ScientificDistinctivenessAnalyzer()
        .build(
            report,
            plan,
            packet,
        )
    )

    assert len(result.reviews) == 1

    review = result.reviews[0]

    assert review.evidence_pattern == (
        "HIGHER_ORDER_RELATIONAL_GAP_WITH_"
        "LOWER_ORDER_PRIOR_ART"
    )

    assert review.core_claim_count == 2

    assert (
        review
        .lower_order_supported_core_claim_count
        == 1
    )

    assert (
        review
        .lower_order_supported_core_fraction
        == 0.5
    )

    assert (
        review
        .higher_order_relational_gap_claim_count
        == 1
    )

    assert (
        review.semantic_dimensions
        .straightforward_conjunction
        == "UNASSESSED"
    )

    assert (
        review.semantic_dimensions
        .mechanism_switch
        == "UNASSESSED"
    )

    assert result.retrieval_performed is False
    assert result.model_review_performed is False
    assert result.action_policy_applied is False
    assert result.scientific_selection_changed is False


def test_distinctiveness_rejects_query_plan_drift() -> None:
    report, plan, packet = _fixture()

    drifted_packet = packet.model_copy(
        update={
            "source_query_plan_id":
                "plan:wrong"
        }
    )

    with pytest.raises(
        ValueError,
        match="source_query_plan_id",
    ):
        (
            ScientificDistinctivenessAnalyzer()
            .build(
                report,
                plan,
                drifted_packet,
            )
        )


def test_distinctiveness_rejects_unknown_work_reference() -> None:
    report, plan, packet = _fixture()

    payload = report.model_dump(
        mode="json"
    )

    payload[
        "cards"
    ][0][
        "claim_reviews"
    ][0][
        "matches"
    ][0][
        "work_id"
    ] = "work:missing"

    drifted_report = (
        ExternalNoveltyReport
        .model_validate(
            payload
        )
    )

    with pytest.raises(
        ValueError,
        match="unknown prior-art work_id",
    ):
        (
            ScientificDistinctivenessAnalyzer()
            .build(
                drifted_report,
                plan,
                packet,
            )
        )


def test_distinctiveness_output_is_deterministic() -> None:
    report, plan, packet = _fixture()

    analyzer = (
        ScientificDistinctivenessAnalyzer()
    )

    left = analyzer.build(
        report,
        plan,
        packet,
    )

    right = analyzer.build(
        report,
        plan,
        packet,
    )

    assert (
        left.model_dump(
            mode="json"
        )
        ==
        right.model_dump(
            mode="json"
        )
    )


def test_distinctiveness_recomputes_stale_derived_aggregates() -> None:
    report, plan, packet = _fixture()

    payload = report.model_dump(
        mode="json"
    )

    card = payload["cards"][0]

    card["lower_order_core_prior_art_work_ids"] = []
    card["lower_order_core_unique_work_count"] = 0
    card["lower_order_supported_core_claim_ids"] = []
    card["higher_order_relational_gap_claim_ids"] = []
    card["relational_gap_kind"] = "NONE"

    drifted_report = (
        ExternalNoveltyReport
        .model_validate(
            payload
        )
    )

    result = (
        ScientificDistinctivenessAnalyzer()
        .build(
            drifted_report,
            plan,
            packet,
        )
    )

    review = result.reviews[0]

    assert review.evidence_pattern == (
        "HIGHER_ORDER_RELATIONAL_GAP_WITH_"
        "LOWER_ORDER_PRIOR_ART"
    )

    assert (
        review.higher_order_relational_gap_claim_count
        == 1
    )

    assert (
        review.lower_order_core_unique_work_count
        == 1
    )

    assert (
        "card_lower_order_core_prior_art_work_ids_drift"
        in review.source_aggregate_warnings
    )

    assert result.source_aggregate_warning_count >= 4
