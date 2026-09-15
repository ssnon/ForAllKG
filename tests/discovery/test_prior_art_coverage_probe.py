from __future__ import annotations

import pytest

from pipeline_core.discovery.external_novelty import (
    ExternalNoveltyAssessor,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    ExternalNoveltyPolicy,
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
    RankedPriorArtWork,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from pipeline_core.discovery.prior_art_coverage_probe import (
    PriorArtPreReviewCoverageProbe,
)


class FakeRanker:
    def __init__(
        self,
        work_ids_by_claim: dict[str, list[str]],
    ) -> None:
        self.work_ids_by_claim = {
            key: list(value)
            for key, value in work_ids_by_claim.items()
        }
        self.calls: list[str] = []

    def rank(
        self,
        claim,
        packet,
        plan,
    ) -> ClaimPriorArtCandidateSet:
        self.calls.append(claim.claim_id)
        return ClaimPriorArtCandidateSet(
            hypothesis_id=claim.hypothesis_id,
            claim_id=claim.claim_id,
            ranked_works=[
                RankedPriorArtWork(
                    work_id=work_id,
                    relevance_score=0.8,
                    semantic_similarity=0.8,
                    lexical_coverage=0.8,
                    reaction_domain_relevance=1.0,
                    catalyst_scope_relevance=1.0,
                    abstract_available=bool(
                        next(
                            (
                                row.abstract
                                for row in packet.works
                                if row.work_id == work_id
                            ),
                            None,
                        )
                    ),
                )
                for work_id in self.work_ids_by_claim[
                    claim.claim_id
                ]
            ],
        )


def _portfolio() -> HypothesisPortfolio:
    card = HypothesisCard(
        hypothesis_id="hypothesis:test",
        domain_profile_id="domain:test",
        source_context_id="context:test",
        source_context_sha256="context-sha",
        source_report_id="report:test",
        source_report_sha256="report-sha",
        title="Test hypothesis",
        hypothesis_statement=(
            "A modifier may condition the relationship between X and Y."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["stmt:1"],
        gap_statement_ids=[],
        inferential_bridge="Bounded mechanistic bridge.",
        predicted_observations=[
            PredictedObservation(
                observation_id="obs:1",
                observable="response Y",
                expected_direction="unspecified",
                rationale="Bounded prediction.",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable="response Y",
                falsifying_outcome="No modifier dependence.",
            )
        ],
        assumptions=[],
        source_paper_ids=["paper:1"],
        gap_paper_ids=[],
        cross_paper_synthesis=False,
        candidate_dependency="none",
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )
    return HypothesisPortfolio(
        portfolio_id="portfolio:test",
        domain_profile_id="domain:test",
        source_context_id="context:test",
        source_context_sha256="context-sha",
        source_report_id="report:test",
        source_report_sha256="report-sha",
        hypotheses=[card],
    )


def _claim(
    claim_id: str,
    *,
    importance: str = "core",
) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:test",
        claim_rank=1 if claim_id == "claim:c1" else 2,
        kind="moderator_interaction",
        importance=importance,
        text=f"Claim text for {claim_id}",
        rationale="Novelty decomposition rationale.",
        search_concepts=["modifier", "response"],
        search_queries=[f"query for {claim_id}"],
    )


def _plan(
    *,
    c1_importance: str = "core",
    c2_importance: str = "core",
) -> LiteratureQueryPlan:
    c1 = _claim(
        "claim:c1",
        importance=c1_importance,
    )
    c2 = _claim(
        "claim:c2",
        importance=c2_importance,
    )
    return LiteratureQueryPlan(
        plan_id="plan:test",
        plan_sha256="plan-sha",
        source_portfolio_id="portfolio:test",
        queries=[
            LiteratureQuery(
                query_id="q:c1:1",
                hypothesis_id="hypothesis:test",
                claim_id="claim:c1",
                query_kind="claim_primary",
                query_text="c1 primary",
            ),
            LiteratureQuery(
                query_id="q:c1:2",
                hypothesis_id="hypothesis:test",
                claim_id="claim:c1",
                query_kind="claim_variant",
                query_text="c1 variant",
            ),
            LiteratureQuery(
                query_id="q:c2:1",
                hypothesis_id="hypothesis:test",
                claim_id="claim:c2",
                query_kind="claim_primary",
                query_text="c2 primary",
            ),
            LiteratureQuery(
                query_id="q:c2:2",
                hypothesis_id="hypothesis:test",
                claim_id="claim:c2",
                query_kind="claim_variant",
                query_text="c2 variant",
            ),
        ],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:test",
                title="Test hypothesis",
                claims=[c1, c2],
                decomposition_notes="fixture",
            )
        ],
    )


def _packet(
    *,
    abstractless_ids: set[str] | None = None,
    failed_query_ids: set[str] | None = None,
) -> PriorArtPacket:
    abstractless_ids = abstractless_ids or set()
    failed_query_ids = failed_query_ids or set()

    works = []
    query_cycle = [
        "q:c1:1",
        "q:c1:2",
        "q:c2:1",
        "q:c2:2",
    ]
    for index in range(1, 13):
        work_id = f"work:{index:02d}"
        qid = query_cycle[(index - 1) % len(query_cycle)]
        claim_id = (
            "claim:c1"
            if "c1" in qid
            else "claim:c2"
        )
        works.append(
            PriorArtWork(
                work_id=work_id,
                title=f"Work {index}",
                abstract=(
                    None
                    if work_id in abstractless_ids
                    else f"Abstract for work {index}"
                ),
                providers=["fixture"],
                retrieval_query_ids=[qid],
                retrieval_claim_ids=[claim_id],
            )
        )

    executions = [
        QueryExecution(
            query_id=qid,
            provider="fixture",
            success=(qid not in failed_query_ids),
            result_count=10,
            elapsed_seconds=0.01,
        )
        for qid in query_cycle
    ]

    return PriorArtPacket(
        packet_id="packet:test",
        packet_sha256="packet-sha",
        source_portfolio_id="portfolio:test",
        source_query_plan_id="plan:test",
        searched_at_utc="2026-09-15T00:00:00+00:00",
        providers_requested=["fixture"],
        works=works,
        executions=executions,
        raw_work_count=len(works),
        canonical_work_count=len(works),
    )


def _claim_review_from_probe(
    row,
) -> ClaimPriorArtReview:
    return ClaimPriorArtReview(
        hypothesis_id=row.hypothesis_id,
        claim_id=row.claim_id,
        claim_text=f"Claim text for {row.claim_id}",
        importance=row.importance,
        status="NO_DIRECT_MATCH_FOUND",
        matches=[],
        coverage=ClaimSearchCoverage(
            claim_id=row.claim_id,
            query_count=row.query_count,
            successful_query_count=(
                row.successful_query_count
            ),
            unique_work_count=row.ranked_work_count,
            abstract_work_count=row.abstract_work_count,
            reviewed_work_count=0,
        ),
        reason_codes=[],
        reviewer_unknown_work_ids=[],
        interpretation="fixture",
    )


def _existing_coverage(
    *,
    policy: ExternalNoveltyPolicy,
    portfolio: HypothesisPortfolio,
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
    claim_rows,
):
    assessor = ExternalNoveltyAssessor(
        decomposer=object(),
        ranker=object(),
        review_backend=object(),
        policy=policy,
        compiler=object(),
    )
    reviews = [
        _claim_review_from_probe(row)
        for row in claim_rows
    ]
    return assessor._coverage(
        portfolio.hypotheses[0],
        reviews,
        packet,
        plan,
    )


def test_probe_matches_existing_post_review_coverage_exactly() -> None:
    portfolio = _portfolio()
    plan = _plan()
    packet = _packet()
    policy = ExternalNoveltyPolicy()

    ranker = FakeRanker({
        "claim:c1": [
            "work:01",
            "work:02",
            "work:05",
            "work:06",
        ],
        "claim:c2": [
            "work:03",
            "work:04",
            "work:07",
            "work:08",
        ],
    })

    result = PriorArtPreReviewCoverageProbe(
        ranker,
        policy=policy,
    ).build(
        portfolio,
        plan,
        packet,
    )

    assert result.claim_review_llm_performed is False
    assert (
        result.prior_art_relationship_classification_performed
        is False
    )
    assert result.novelty_authority is False
    assert result.production_selection_changed is False

    hypothesis = result.hypotheses[0]
    existing = _existing_coverage(
        policy=policy,
        portfolio=portfolio,
        plan=plan,
        packet=packet,
        claim_rows=hypothesis.claim_coverages,
    )

    assert hypothesis.coverage == existing
    assert (
        hypothesis.gate_decision
        == "KEEP_FOR_CLAIM_REVIEW"
    )
    assert (
        hypothesis.coverage
        .sufficient_for_absence_based_novelty
        is True
    )
    assert result.keep_for_claim_review_count == 1
    assert result.evidence_required_count == 0


def test_probe_fails_closed_when_one_core_claim_has_only_two_abstracts() -> None:
    portfolio = _portfolio()
    plan = _plan()
    packet = _packet()

    ranker = FakeRanker({
        "claim:c1": [
            "work:01",
            "work:02",
        ],
        "claim:c2": [
            "work:03",
            "work:04",
            "work:07",
            "work:08",
        ],
    })

    result = PriorArtPreReviewCoverageProbe(
        ranker
    ).build(
        portfolio,
        plan,
        packet,
    )

    hypothesis = result.hypotheses[0]

    assert hypothesis.coverage.unique_work_count == 12
    assert hypothesis.coverage.abstract_work_count == 12
    assert (
        hypothesis.coverage.successful_query_count
        == 4
    )

    c1 = next(
        row
        for row in hypothesis.claim_coverages
        if row.claim_id == "claim:c1"
    )
    assert c1.abstract_work_count == 2
    assert c1.passes_minimum_abstract_coverage is False

    assert (
        hypothesis.coverage
        .core_claims_with_minimum_abstract_coverage
        == 1
    )
    assert (
        hypothesis.coverage
        .sufficient_for_absence_based_novelty
        is False
    )
    assert (
        hypothesis.gate_decision
        == "EVIDENCE_REQUIRED"
    )
    assert hypothesis.reason_codes == [
        "one_or_more_core_claims_below_abstract_minimum"
    ]


def test_probe_uses_all_claims_when_no_claim_is_marked_core() -> None:
    portfolio = _portfolio()
    plan = _plan(
        c1_importance="supporting",
        c2_importance="supporting",
    )
    packet = _packet()

    ranker = FakeRanker({
        "claim:c1": [
            "work:01",
            "work:02",
        ],
        "claim:c2": [
            "work:03",
            "work:04",
            "work:07",
        ],
    })

    result = PriorArtPreReviewCoverageProbe(
        ranker
    ).build(
        portfolio,
        plan,
        packet,
    )

    hypothesis = result.hypotheses[0]

    assert set(hypothesis.core_claim_ids) == {
        "claim:c1",
        "claim:c2",
    }
    assert hypothesis.coverage.core_claim_count == 2
    assert (
        hypothesis.coverage
        .core_claims_with_minimum_abstract_coverage
        == 1
    )
    assert (
        hypothesis.gate_decision
        == "EVIDENCE_REQUIRED"
    )


def test_probe_preserves_global_query_and_work_thresholds() -> None:
    portfolio = _portfolio()
    plan = _plan()
    packet = _packet(
        failed_query_ids={
            "q:c1:2",
            "q:c2:1",
            "q:c2:2",
        }
    )

    ranker = FakeRanker({
        "claim:c1": [
            "work:01",
            "work:02",
            "work:05",
        ],
        "claim:c2": [
            "work:03",
            "work:04",
            "work:07",
        ],
    })

    result = PriorArtPreReviewCoverageProbe(
        ranker
    ).build(
        portfolio,
        plan,
        packet,
    )
    hypothesis = result.hypotheses[0]

    assert (
        hypothesis.coverage.successful_query_count
        == 1
    )
    assert (
        hypothesis.gate_decision
        == "EVIDENCE_REQUIRED"
    )
    assert (
        "successful_query_count_below_minimum"
        in hypothesis.reason_codes
    )


def test_probe_rejects_candidate_set_with_unknown_work_id() -> None:
    portfolio = _portfolio()
    plan = _plan()
    packet = _packet()

    ranker = FakeRanker({
        "claim:c1": [
            "work:01",
            "work:missing",
        ],
        "claim:c2": [
            "work:03",
            "work:04",
            "work:07",
        ],
    })

    with pytest.raises(
        ValueError,
        match="unknown prior-art work IDs",
    ):
        PriorArtPreReviewCoverageProbe(
            ranker
        ).build(
            portfolio,
            plan,
            packet,
        )


def test_probe_rejects_query_plan_portfolio_mismatch_before_ranking() -> None:
    portfolio = _portfolio()
    plan = _plan().model_copy(
        update={
            "source_portfolio_id": (
                "portfolio:wrong"
            )
        }
    )
    packet = _packet()
    ranker = FakeRanker({
        "claim:c1": [],
        "claim:c2": [],
    })

    with pytest.raises(
        ValueError,
        match="query-plan portfolio mismatch",
    ):
        PriorArtPreReviewCoverageProbe(
            ranker
        ).build(
            portfolio,
            plan,
            packet,
        )

    assert ranker.calls == []
