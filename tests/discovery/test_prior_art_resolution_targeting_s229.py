from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
    PriorArtWork,
    RankedPriorArtWork,
)
from pipeline_core.discovery.prior_art_resolution_targeting import (
    select_pre_review_resolution_targets,
)


class FakeRanker:
    def rank(self, claim, packet, plan):
        return ClaimPriorArtCandidateSet(
            hypothesis_id=claim.hypothesis_id,
            claim_id=claim.claim_id,
            ranked_works=[
                RankedPriorArtWork(
                    work_id="w-missing",
                    relevance_score=0.9,
                    semantic_similarity=0.9,
                    lexical_coverage=0.5,
                    abstract_available=False,
                ),
                RankedPriorArtWork(
                    work_id="w-full",
                    relevance_score=0.8,
                    semantic_similarity=0.8,
                    lexical_coverage=0.4,
                    abstract_available=True,
                ),
            ],
        )


def test_pre_review_resolution_targets_core_missing_abstract_only():
    core = NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        novelty_selection_role=None,
        text="core relation",
        rationale="r",
        search_concepts=["x"],
        search_queries=["x"],
    )
    supporting = NoveltyClaim(
        claim_id="c2",
        hypothesis_id="h1",
        claim_rank=2,
        kind="mechanistic_link",
        importance="supporting",
        novelty_selection_role=None,
        text="support",
        rationale="r",
        search_concepts=["y"],
        search_queries=["y"],
    )
    group = HypothesisNoveltyClaims(
        hypothesis_id="h1",
        title="t",
        claims=[core, supporting],
        decomposition_notes="n",
    )
    plan = LiteratureQueryPlan(
        plan_id="p1",
        plan_sha256="a" * 64,
        source_portfolio_id="portfolio",
        queries=[],
        claims=[group],
    )
    packet = PriorArtPacket(
        packet_id="packet",
        packet_sha256="b" * 64,
        source_portfolio_id="portfolio",
        source_query_plan_id="p1",
        searched_at_utc="2026-01-01T00:00:00+00:00",
        works=[
            PriorArtWork(
                work_id="w-missing",
                title="Missing abstract scientific article",
                abstract=None,
            ),
            PriorArtWork(
                work_id="w-full",
                title="Full abstract scientific article",
                abstract="available",
            ),
        ],
    )

    result = select_pre_review_resolution_targets(
        plan=plan,
        packet=packet,
        ranker=FakeRanker(),
    )

    assert result["core_claim_count"] == 1
    assert result["selected_work_ids"] == ["w-missing"]
    assert result["review_outcome_consumed"] is False
