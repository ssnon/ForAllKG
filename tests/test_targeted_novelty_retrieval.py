from dac_her.external_novelty_contracts import (
    LiteratureQueryPlan,
    NoveltyClaim,
    HypothesisNoveltyClaims,
    PriorArtPacket,
)
from dac_her.novelty_refinement_contracts import NoveltyGap
from dac_her.targeted_novelty_retrieval import build_augmented_query_plan


def test_targeted_query_plan_preserves_claims_and_adds_delta():
    claim = NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text="charge donation moderates d band response",
        rationale="test",
        search_concepts=[],
        search_queries=[],
    )
    base = LiteratureQueryPlan(
        plan_id="plan",
        plan_sha256="sha",
        source_portfolio_id="p1",
        queries=[],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="h1",
                title="H",
                claims=[claim],
            )
        ],
    )
    gap = NoveltyGap(
        gap_id="g1",
        hypothesis_id="h1",
        source_external_status="INSUFFICIENT_SEARCH_EVIDENCE",
        action="targeted_search_then_refine",
        target_claim_ids=["c1"],
        differentiator=claim.text,
        targeted_queries=[
            "nitrogen coordination charge donation d band HER"
        ],
    )
    full, delta = build_augmented_query_plan(base, gap)
    assert len(full.queries) == 1
    assert len(delta.queries) == 1
    assert full.queries[0].claim_id == "c1"
    assert full.claims[0].claims[0].claim_id == "c1"
