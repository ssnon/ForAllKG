from dac_her.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    ClaimPriorArtReviewDraft,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtMatchDraft,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
    RankedPriorArtWork,
)
from dac_her.prior_art_matching import ClaimPriorArtCompiler


def _fixture():
    claim = NoveltyClaim(
        claim_id="claim:1",
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text="nitrogen coordination charge donation moderates HER activity",
        rationale="test",
        search_concepts=["nitrogen coordination", "charge donation", "HER"],
        search_queries=["nitrogen coordination charge donation HER"],
    )
    query = LiteratureQuery(
        query_id="query:1",
        hypothesis_id="hypothesis:1",
        claim_id="claim:1",
        query_kind="claim_primary",
        query_text="nitrogen coordination charge donation HER",
    )
    plan = LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="x",
        source_portfolio_id="portfolio:1",
        queries=[query],
        claims=[],
    )
    work = PriorArtWork(
        work_id="prior_art_work:valid",
        title="Charge redistribution in nitrogen coordinated dual atom HER catalysts",
        abstract="Nitrogen coordination changes charge redistribution during HER.",
        retrieval_query_ids=["query:1"],
        retrieval_claim_ids=["claim:1"],
    )
    packet = PriorArtPacket(
        packet_id="packet:1",
        packet_sha256="x",
        source_portfolio_id="portfolio:1",
        source_query_plan_id="plan:1",
        searched_at_utc="2026-08-10T00:00:00+00:00",
        providers_requested=["test"],
        works=[work],
        executions=[
            QueryExecution(
                query_id="query:1",
                provider="test",
                success=True,
                result_count=1,
            )
        ],
        raw_work_count=1,
        canonical_work_count=1,
    )
    candidates = ClaimPriorArtCandidateSet(
        hypothesis_id="hypothesis:1",
        claim_id="claim:1",
        ranked_works=[
            RankedPriorArtWork(
                work_id="prior_art_work:valid",
                relevance_score=0.8,
                semantic_similarity=0.8,
                lexical_coverage=0.7,
                reaction_domain_relevance=1.0,
                catalyst_scope_relevance=1.0,
                abstract_available=True,
            )
        ],
    )
    return claim, plan, packet, candidates


def test_unknown_reviewer_work_id_is_dropped_and_audited():
    claim, plan, packet, candidates = _fixture()
    draft = ClaimPriorArtReviewDraft(
        matches=[
            PriorArtMatchDraft(
                work_id="prior_art_work:hallucinated",
                relationship="DIRECT_PRIOR_ART",
                confidence=0.95,
                rationale="invalid id",
            ),
            PriorArtMatchDraft(
                work_id="prior_art_work:valid",
                relationship="COMPONENT_ONLY",
                confidence=0.80,
                rationale="valid bounded evidence",
            ),
        ],
        interpretation="bounded review",
    )
    review = ClaimPriorArtCompiler().compile(
        claim, candidates, draft, packet, plan
    )
    assert review.reviewer_unknown_work_ids == ["prior_art_work:hallucinated"]
    assert "reviewer_unknown_work_id_dropped" in review.reason_codes
    assert [row.work_id for row in review.matches] == ["prior_art_work:valid"]
    assert review.status == "COMPONENTS_ONLY"


def test_only_unknown_ids_becomes_insufficient_not_no_match():
    claim, plan, packet, candidates = _fixture()
    draft = ClaimPriorArtReviewDraft(
        matches=[
            PriorArtMatchDraft(
                work_id="prior_art_work:hallucinated",
                relationship="DIRECT_PRIOR_ART",
                confidence=0.95,
                rationale="invalid id",
            )
        ],
        interpretation="bounded review",
    )
    review = ClaimPriorArtCompiler().compile(
        claim, candidates, draft, packet, plan
    )
    assert review.status == "INSUFFICIENT_METADATA"
    assert "reviewer_unknown_work_id_prevents_absence_inference" in review.reason_codes
    assert review.matches == []
