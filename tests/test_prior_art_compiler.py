from __future__ import annotations

from domains.registry import get_domain_profile
from pipeline_core.discovery.external_novelty_contracts import (
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
from pipeline_core.discovery.prior_art_matching import ClaimPriorArtCompiler


def _plan(claim_id: str = "c1") -> LiteratureQueryPlan:
    query = LiteratureQuery(
        query_id="q1",
        hypothesis_id="h1",
        claim_id=claim_id,
        query_kind="claim_primary",
        query_text="nitrogen coordination d band HER",
    )
    return LiteratureQueryPlan(
        plan_id="p1",
        plan_sha256="x",
        source_portfolio_id="portfolio",
        queries=[query],
        claims=[],
    )


def _packet(work: PriorArtWork) -> PriorArtPacket:
    return PriorArtPacket(
        packet_id="packet",
        packet_sha256="x",
        source_portfolio_id="portfolio",
        source_query_plan_id="p1",
        searched_at_utc="2026-08-10T00:00:00+00:00",
        providers_requested=["fixture"],
        works=[work],
        executions=[
            QueryExecution(
                query_id="q1", provider="fixture", success=True, result_count=1
            )
        ],
    )


def _candidates(work: PriorArtWork, *, abstract: bool) -> ClaimPriorArtCandidateSet:
    return ClaimPriorArtCandidateSet(
        hypothesis_id="h1",
        claim_id="c1",
        ranked_works=[
            RankedPriorArtWork(
                work_id=work.work_id,
                relevance_score=0.9,
                semantic_similarity=0.9,
                lexical_coverage=0.5,
                reaction_domain_relevance=1.0,
                catalyst_scope_relevance=0.8,
                abstract_available=abstract,
            )
        ],
    )


def test_direct_and_partial_title_only_matches_are_not_substantive_prior_art() -> None:
    claim = NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text="nitrogen coordination charge donation moderates d band dependent HER activity",
        rationale="fixture",
        search_concepts=["nitrogen coordination", "charge donation", "d band", "HER"],
        search_queries=["nitrogen coordination d band HER"],
    )
    for relationship in ("DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"):
        work = PriorArtWork(
            work_id="w1",
            title="Nitrogen coordination d band HER neighboring title",
            abstract=None,
            providers=["fixture"],
            retrieval_query_ids=["q1"],
            retrieval_claim_ids=["c1"],
        )
        draft = ClaimPriorArtReviewDraft(
            matches=[
                PriorArtMatchDraft(
                    work_id="w1",
                    relationship=relationship,
                    confidence=0.95,
                    rationale="title looks relevant",
                )
            ],
            interpretation="fixture",
        )
        compiled = ClaimPriorArtCompiler(domain_profile=get_domain_profile("dac_her")).compile(
            claim, _candidates(work, abstract=False), draft, _packet(work), _plan()
        )
        assert compiled.matches[0].relationship == "TITLE_ONLY_NEIGHBOR"
        assert compiled.status == "TITLE_ONLY_NEIGHBORS"


def test_out_of_scope_d_band_counterexample_is_contextual_not_direct_conflict() -> None:
    claim = NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text=(
            "At comparable hydrogen adsorption free energies, d band position changes "
            "the relative HER activity of different nitrogen coordination environments."
        ),
        rationale="fixture",
        search_concepts=["nitrogen coordination", "d band", "HER"],
        search_queries=["nitrogen coordination d band HER"],
    )
    work = PriorArtWork(
        work_id="w1",
        title="HER on AuPt alloys is determined by adsorption sites rather than d-band properties",
        abstract=(
            "For AuPt alloys, hydrogen evolution reaction activity is determined by "
            "element-specific adsorption sites rather than d-band properties."
        ),
        providers=["fixture"],
        retrieval_query_ids=["q1"],
        retrieval_claim_ids=["c1"],
    )
    draft = ClaimPriorArtReviewDraft(
        matches=[
            PriorArtMatchDraft(
                work_id="w1",
                relationship="CONFLICTING_PRIOR_ART",
                confidence=0.92,
                rationale="d-band descriptor is challenged",
            )
        ],
        interpretation="fixture",
    )
    compiled = ClaimPriorArtCompiler(domain_profile=get_domain_profile("dac_her")).compile(
        claim, _candidates(work, abstract=True), draft, _packet(work), _plan()
    )
    assert compiled.matches[0].relationship == "CONTEXTUAL_CONFLICT"
    assert not compiled.matches[0].scope_compatible_for_conflict
    assert compiled.status == "COMPONENTS_ONLY"
    assert "conflict_downgraded_for_scope_mismatch" in compiled.reason_codes
    assert "contextual_conflict_present" in compiled.reason_codes


def test_scope_matched_nitrogen_coordination_conflict_survives() -> None:
    claim = NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="moderator_interaction",
        importance="core",
        text=(
            "d band position changes HER activity across different nitrogen coordination environments"
        ),
        rationale="fixture",
        search_concepts=["nitrogen coordination", "d band", "HER"],
        search_queries=["nitrogen coordination d band HER"],
    )
    work = PriorArtWork(
        work_id="w1",
        title="Nitrogen coordinated MN4 sites show HER independent of d-band position",
        abstract=(
            "Across nitrogen-coordinated MN4 electrocatalytic HER sites, changing the "
            "d-band position does not alter the activity ordering."
        ),
        providers=["fixture"],
        retrieval_query_ids=["q1"],
        retrieval_claim_ids=["c1"],
    )
    draft = ClaimPriorArtReviewDraft(
        matches=[
            PriorArtMatchDraft(
                work_id="w1",
                relationship="CONFLICTING_PRIOR_ART",
                confidence=0.9,
                rationale="same scoped relation is opposed",
            )
        ],
        interpretation="fixture",
    )
    compiled = ClaimPriorArtCompiler(domain_profile=get_domain_profile("dac_her")).compile(
        claim, _candidates(work, abstract=True), draft, _packet(work), _plan()
    )
    assert compiled.matches[0].relationship == "CONFLICTING_PRIOR_ART"
    assert compiled.matches[0].scope_compatible_for_conflict
    assert compiled.status == "CONFLICTING_PRIOR_ART"
