from __future__ import annotations

from types import SimpleNamespace

from domains.registry import get_domain_profile
from pipeline_core.discovery.external_novelty import ExternalNoveltyAssessor
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    ClaimPriorArtReview,
    ClaimPriorArtReviewDraft,
    ClaimSearchCoverage,
    ExternalNoveltyPolicy,
    HypothesisNoveltyClaims,
    HypothesisSearchCoverage,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtMatchDraft,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
    RankedPriorArtWork,
)
from pipeline_core.discovery.novelty_claim_decomposition import LiteratureQueryPlanner
from pipeline_core.discovery.prior_art_matching import ClaimPriorArtCompiler
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidate,
    ScientificPortfolioCandidateEvaluationDraft,
    ScientificPortfolioDimensionAssessment,
    ScientificPortfolioNormalizedSketch,
    eligible_profiles,
)


class Dummy:
    pass


def _coverage():
    return HypothesisSearchCoverage(
        hypothesis_id="h", query_count=6, successful_query_count=6,
        provider_success_count=2, unique_work_count=20, abstract_work_count=12,
        core_claim_count=1, core_claims_with_minimum_abstract_coverage=1,
        sufficient_for_absence_based_novelty=True,
    )


def _review(cid, status, importance):
    return ClaimPriorArtReview(
        hypothesis_id="h", claim_id=cid, claim_text=cid,
        importance=importance, status=status, matches=[],
        coverage=ClaimSearchCoverage(
            claim_id=cid, query_count=3, successful_query_count=3,
            unique_work_count=8, abstract_work_count=5, reviewed_work_count=5,
        ), interpretation="fixture",
    )


def test_role_bound_testing_prediction_constrains_clean_gap():
    a = ExternalNoveltyAssessor(
        decomposer=Dummy(), ranker=Dummy(), review_backend=Dummy(),
        policy=ExternalNoveltyPolicy(), compiler=Dummy(),
    )
    reviews = [
        _review("core", "COMPONENTS_ONLY", "core"),
        _review("pred", "PARTIAL_PRIOR_ART", "supporting"),
    ]
    claims = {
        "core": SimpleNamespace(novelty_selection_role="NOVELTY_BEARING"),
        "pred": SimpleNamespace(novelty_selection_role="TESTING_PREDICTION"),
    }
    status, reasons, _ = a._status(reviews, _coverage(), claims_by_id=claims)
    assert status == "LITERATURE_SUPPORTED_EXTENSION"
    assert "central_testing_prediction_has_relation_backed_prior_art" in reasons


def test_sers_first_pass_gets_domain_anchored_variant():
    claim = NoveltyClaim(
        claim_id="c1", hypothesis_id="h1", claim_rank=1,
        kind="mechanistic_link", importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="periodic confinement changes measurement variability",
        rationale="fixture",
        search_concepts=["periodic confinement", "measurement variability"],
        search_queries=["periodic confinement measurement variability"],
    )
    dec = HypothesisNoveltyClaims(hypothesis_id="h1", title="fixture", claims=[claim])
    portfolio = SimpleNamespace(
        portfolio_id="p1",
        hypotheses=[SimpleNamespace(
            hypothesis_id="h1", title="Periodic confinement",
            hypothesis_statement="Periodic confinement changes measurement variability.",
        )],
    )
    plan = LiteratureQueryPlanner(domain_profile=get_domain_profile("sers_au_ag")).build(
        portfolio, [dec]
    )
    variants = [q.query_text for q in plan.queries if q.query_kind == "claim_domain_variant"]
    assert variants
    assert any("SERS" in q or "plasmonic" in q for q in variants)


def test_sers_cross_domain_partial_loses_positive_authority():
    claim = NoveltyClaim(
        claim_id="c1", hypothesis_id="h1", claim_rank=1,
        kind="mechanistic_link", importance="core",
        novelty_selection_role="NOVELTY_BEARING",
        text="periodic confinement reduces SERS measurement variability",
        rationale="fixture", search_concepts=["periodic confinement", "SERS variability"],
        search_queries=["periodic confinement SERS variability"],
    )
    query = LiteratureQuery(
        query_id="q1", hypothesis_id="h1", claim_id="c1",
        query_kind="claim_primary", query_text="periodic confinement SERS variability",
    )
    plan = LiteratureQueryPlan(
        plan_id="plan", plan_sha256="x", source_portfolio_id="portfolio",
        queries=[query], claims=[],
    )
    work = PriorArtWork(
        work_id="w1", title="Periodic modulation and confinement in a tokamak plasma",
        abstract="Periodic modulation changes plasma confinement and fluctuation variance in a tokamak.",
        providers=["fixture"], retrieval_query_ids=["q1"], retrieval_claim_ids=["c1"],
    )
    packet = PriorArtPacket(
        packet_id="packet", packet_sha256="x", source_portfolio_id="portfolio",
        source_query_plan_id="plan", searched_at_utc="2026-10-02T00:00:00+00:00",
        providers_requested=["fixture"], works=[work],
        executions=[QueryExecution(query_id="q1", provider="fixture", success=True, result_count=1)],
    )
    candidates = ClaimPriorArtCandidateSet(
        hypothesis_id="h1", claim_id="c1",
        ranked_works=[RankedPriorArtWork(
            work_id="w1", relevance_score=0.9, semantic_similarity=0.9,
            lexical_coverage=0.5, reaction_domain_relevance=0.5,
            catalyst_scope_relevance=0.5, abstract_available=True,
        )],
    )
    draft = ClaimPriorArtReviewDraft(
        matches=[PriorArtMatchDraft(
            work_id="w1", relationship="PARTIAL_PRIOR_ART", confidence=0.95,
            rationale="generic relation overlap", evidence_spans=[],
        )], interpretation="fixture",
    )
    result = ClaimPriorArtCompiler(domain_profile=get_domain_profile("sers_au_ag")).compile(
        claim, candidates, draft, packet, plan
    )
    assert result.matches[0].relationship == "COMPONENT_ONLY"
    assert result.status == "COMPONENTS_ONLY"
    assert "positive_prior_art_downgraded_for_domain_mismatch" in result.reason_codes


def _a(level):
    return ScientificPortfolioDimensionAssessment(level=level, rationale="fixture")


def test_low_task_relevance_cannot_hide_in_other_profiles():
    candidate = ScientificPortfolioCandidate(
        candidate_id="c", origin="EVOLUTION", source_object_id="e",
        operator_id="CROSS_SOURCE_BRIDGE", idea_form="CROSS_SOURCE_BRIDGE",
        title="Adjacent downstream compensation",
        scientific_intent="Use downstream processing to compensate measurements.",
        conceptual_change_summary="changes intervention class",
        core_relations=["processing -> prediction error"],
        task_relation_mode="SUBORDINATE", conceptual_family_signature="family",
        external_literature_lineage=True, cross_source_composition=True,
    )
    draft = ScientificPortfolioCandidateEvaluationDraft(
        candidate_id="c",
        normalized_sketch=ScientificPortfolioNormalizedSketch(
            hypothesis_frame="fixture", predicted_observation="fixture",
            falsification_condition="fixture", discriminating_observation="fixture",
        ),
        task_relevance=_a("LOW"), mechanistic_coherence=_a("HIGH"),
        falsifiability=_a("HIGH"), discriminating_power=_a("HIGH"),
        operationalizability=_a("HIGH"), information_gain=_a("HIGH"),
        overall_rationale="fixture",
    )
    assert eligible_profiles(candidate, draft) == []
