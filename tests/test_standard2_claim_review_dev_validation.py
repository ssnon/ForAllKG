from __future__ import annotations

from dac_her.external_novelty_contracts import (
    ClaimPriorArtReviewDraft,
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtMatchDraft,
    PriorArtPacket,
    PriorArtWork,
    QueryExecution,
)
from dac_her.standard2_claim_review_dev_validation import (
    candidate_set_from_ranker_row,
    compile_drafts,
    reviewer_input_from_candidates,
    sha256_json,
)


def _claim() -> NoveltyClaim:
    return NoveltyClaim(
        claim_id="c1",
        hypothesis_id="h1",
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        text="Nanogap geometry regulates SERS enhancement.",
        rationale="test",
        search_concepts=["nanogap", "SERS"],
        search_queries=["nanogap SERS"],
    )


def _plan() -> LiteratureQueryPlan:
    claim = _claim()
    body = {
        "schema_version": "literature-query-plan-v1",
        "plan_id": "plan1",
        "source_portfolio_id": "portfolio1",
        "queries": [
            LiteratureQuery(
                query_id="q1",
                hypothesis_id="h1",
                claim_id="c1",
                query_kind="claim_primary",
                query_text="nanogap SERS",
            )
        ],
        "claims": [
            HypothesisNoveltyClaims(
                hypothesis_id="h1",
                title="test",
                claims=[claim],
            )
        ],
        "policy_version": "external-novelty-query-policy-v1",
    }
    body["plan_sha256"] = sha256_json(body)
    return LiteratureQueryPlan(**body)


def _packet(*, abstract: str | None) -> PriorArtPacket:
    work = PriorArtWork(
        work_id="w1",
        title="Nanogap SERS enhancement",
        abstract=abstract,
        providers=["openalex"],
        provider_ids={"openalex": "w1"},
        retrieval_query_ids=["q1"],
        retrieval_claim_ids=["c1"],
    )
    body = {
        "schema_version": "prior-art-packet-v1",
        "packet_id": "packet1",
        "source_portfolio_id": "portfolio1",
        "source_query_plan_id": "plan1",
        "searched_at_utc": "2026-08-16T00:00:00+00:00",
        "providers_requested": ["openalex"],
        "works": [work.model_dump(mode="json")],
        "executions": [
            QueryExecution(
                query_id="q1",
                provider="openalex",
                success=True,
                result_count=1,
            ).model_dump(mode="json")
        ],
        "raw_work_count": 1,
        "canonical_work_count": 1,
        "deduplicated_work_count": 0,
        "supplementary_records_collapsed": 0,
        "epistemic_usage":
            "prior_art_only_not_positive_premise",
    }
    body["packet_sha256"] = sha256_json(body)
    return PriorArtPacket(**body)


def _ranker_report() -> dict:
    return {
        "claim_reports": [
            {
                "hypothesis_id": "h1",
                "claim_id": "c1",
                "claim_rank": 1,
                "importance": "core",
                "kind": "mechanistic_link",
                "claim_text":
                    "Nanogap geometry regulates SERS enhancement.",
                "top_ranked_works": [
                    {
                        "rank": 1,
                        "work_id": "w1",
                        "title": "Nanogap SERS enhancement",
                        "year": 2024,
                        "doi": "10.1/test",
                        "providers": ["openalex"],
                        "abstract_available": True,
                        "abstract_excerpt": "",
                        "relevance_score": 0.9,
                        "semantic_similarity": 0.9,
                        "lexical_coverage": 0.8,
                        "reaction_domain_relevance": 1.0,
                        "catalyst_scope_relevance": 1.0,
                        "retrieval_query_ids": ["q1"],
                        "retrieval_claim_ids": ["c1"],
                    }
                ],
            }
        ]
    }


def _spec() -> dict:
    return {
        "compiler": {
            "domain_profile_id": "sers_au_ag",
            "policy": {
                "policy_version":
                    "external-novelty-policy-v1.1",
                "max_claims_per_hypothesis": 4,
                "max_queries_per_claim": 2,
                "max_ranked_works_per_claim": 8,
                "min_match_confidence": 0.65,
                "direct_match_confidence": 0.70,
                "min_unique_works_for_absence": 10,
                "min_abstract_works_for_absence": 5,
                "min_abstract_works_per_core_claim": 3,
                "min_successful_queries_for_absence": 2,
                "require_abstract_for_strong_match": True,
                "require_abstract_for_partial_match": True,
                "min_reaction_domain_for_conflict": 0.75,
                "min_catalyst_scope_for_conflict": 0.75,
            },
        }
    }


def test_frozen_candidate_set_preserves_rank_order():
    row = _ranker_report()["claim_reports"][0]
    candidates = candidate_set_from_ranker_row(row)
    assert [x.work_id for x in candidates.ranked_works] == ["w1"]


def test_reviewer_input_uses_canonical_abstract():
    row = _ranker_report()["claim_reports"][0]
    candidates = candidate_set_from_ranker_row(row)
    values = reviewer_input_from_candidates(
        packet=_packet(abstract="full abstract"),
        candidates=candidates,
    )
    assert values[0]["abstract"] == "full abstract"
    assert values[0]["work_id"] == "w1"


def test_compiler_downgrades_direct_match_without_abstract():
    draft = ClaimPriorArtReviewDraft(
        matches=[
            PriorArtMatchDraft(
                work_id="w1",
                relationship="DIRECT_PRIOR_ART",
                confidence=0.95,
                rationale="same relation",
            )
        ],
        interpretation="bounded review",
    )
    reviews = compile_drafts(
        spec=_spec(),
        plan=_plan(),
        packet=_packet(abstract=None),
        ranker_report=_ranker_report(),
        drafts={"c1": draft},
    )
    assert reviews[0].status == "TITLE_ONLY_NEIGHBORS"
    assert reviews[0].matches[0].relationship == "TITLE_ONLY_NEIGHBOR"


def test_compiler_drops_unknown_work_id_fail_closed():
    draft = ClaimPriorArtReviewDraft(
        matches=[
            PriorArtMatchDraft(
                work_id="invented",
                relationship="DIRECT_PRIOR_ART",
                confidence=0.95,
                rationale="invented",
            )
        ],
        interpretation="bounded review",
    )
    reviews = compile_drafts(
        spec=_spec(),
        plan=_plan(),
        packet=_packet(abstract="full abstract"),
        ranker_report=_ranker_report(),
        drafts={"c1": draft},
    )
    assert reviews[0].status == "INSUFFICIENT_METADATA"
    assert reviews[0].reviewer_unknown_work_ids == ["invented"]
