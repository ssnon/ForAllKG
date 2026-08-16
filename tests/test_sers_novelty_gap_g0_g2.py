from __future__ import annotations

import pytest
from pydantic import ValidationError

from dac_her.domains.registry import get_domain_profile
from dac_her.external_novelty import ExternalNoveltyAssessor
from dac_her.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    ExternalNoveltyPolicy,
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
)
from dac_her.hypothesis_contracts import HypothesisCard, HypothesisPortfolio
from dac_her.novelty_gap_analysis import NoveltyGapAnalyzer
from dac_her.novelty_refinement_contracts import NoveltyGap, TargetedGapQuery
from dac_her.targeted_novelty_retrieval import build_augmented_query_plan


def _claim(claim_id: str, hypothesis_id: str = "h1", text: str | None = None) -> NoveltyClaim:
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id=hypothesis_id,
        claim_rank=1,
        kind="mechanistic_link",
        importance="core",
        text=text or f"claim text {claim_id}",
        rationale="unit-test rationale",
        search_concepts=[],
        search_queries=[],
    )


def _review(
    claim_id: str,
    *,
    hypothesis_id: str = "h1",
    text: str | None = None,
    status: str = "DIRECT_PRIOR_ART",
) -> ClaimPriorArtReview:
    return ClaimPriorArtReview(
        hypothesis_id=hypothesis_id,
        claim_id=claim_id,
        claim_text=text or f"claim text {claim_id}",
        importance="core",
        status=status,
        matches=[],
        coverage=ClaimSearchCoverage(
            claim_id=claim_id,
            query_count=1,
            successful_query_count=0,
            unique_work_count=0,
            abstract_work_count=0,
            reviewed_work_count=0,
        ),
        reason_codes=[],
        reviewer_unknown_work_ids=[],
        interpretation="unit-test review",
    )


def _portfolio() -> HypothesisPortfolio:
    card = HypothesisCard.model_construct(
        hypothesis_id="h1",
        title="Unit hypothesis",
        hypothesis_statement="A bounded test hypothesis.",
    )
    return HypothesisPortfolio.model_construct(
        portfolio_id="p1",
        hypotheses=[card],
        abstention_reason=None,
    )


def _plan(claim_ids=("c1",)) -> LiteratureQueryPlan:
    claims = [_claim(claim_id) for claim_id in claim_ids]
    query = LiteratureQuery(
        query_id="q1",
        hypothesis_id="h1",
        claim_id=claim_ids[0],
        query_kind="claim_primary",
        query_text="existing query",
    )
    return LiteratureQueryPlan(
        plan_id="plan1",
        plan_sha256="unit",
        source_portfolio_id="p1",
        queries=[query],
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="h1",
                title="Unit hypothesis",
                claims=claims,
                decomposition_notes="",
            )
        ],
    )


def _packet() -> PriorArtPacket:
    return PriorArtPacket(
        packet_id="packet1",
        packet_sha256="unit",
        source_portfolio_id="p1",
        source_query_plan_id="plan1",
        searched_at_utc="2026-08-16T00:00:00Z",
        providers_requested=[],
        works=[],
        executions=[],
    )


def _assessor() -> ExternalNoveltyAssessor:
    return ExternalNoveltyAssessor(
        decomposer=object(),
        ranker=object(),
        review_backend=object(),
        policy=ExternalNoveltyPolicy(),
        compiler=object(),
    )


def test_g0_compile_report_from_frozen_claim_reviews_is_deterministic() -> None:
    assessor = _assessor()
    portfolio = _portfolio()
    plan = _plan()
    packet = _packet()
    review = _review("c1")

    first = assessor.compile_report_from_claim_reviews(portfolio, plan, packet, [review])
    second = assessor.compile_report_from_claim_reviews(portfolio, plan, packet, [review])

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.report_sha256 == second.report_sha256
    assert first.cards[0].status == "WELL_ESTABLISHED"
    assert first.source_portfolio_id == "p1"
    assert first.source_prior_art_packet_id == "packet1"


def test_g0_claim_review_set_is_exact_and_fail_closed() -> None:
    assessor = _assessor()
    portfolio = _portfolio()
    plan = _plan()
    packet = _packet()

    with pytest.raises(ValueError, match="exactly match"):
        assessor.compile_report_from_claim_reviews(portfolio, plan, packet, [])

    unexpected = _review("unexpected")
    with pytest.raises(ValueError, match="exactly match"):
        assessor.compile_report_from_claim_reviews(
            portfolio, plan, packet, [unexpected]
        )

    duplicated = _review("c1")
    with pytest.raises(ValueError, match="duplicate claim review IDs"):
        assessor.compile_report_from_claim_reviews(
            portfolio, plan, packet, [duplicated, duplicated]
        )


def test_g0_rejects_review_metadata_and_coverage_drift() -> None:
    assessor = _assessor()
    portfolio = _portfolio()
    plan = _plan()
    packet = _packet()

    with pytest.raises(ValueError, match="text drift"):
        assessor.compile_report_from_claim_reviews(
            portfolio, plan, packet, [_review("c1", text="drifted text")]
        )

    good = _review("c1")
    bad_coverage = good.model_copy(
        update={
            "coverage": good.coverage.model_copy(update={"claim_id": "other"})
        }
    )
    with pytest.raises(ValueError, match="coverage claim_id mismatch"):
        assessor.compile_report_from_claim_reviews(
            portfolio, plan, packet, [bad_coverage]
        )


def test_domain_owned_targeted_query_templates_preserve_dac_and_isolate_sers() -> None:
    core = "coordination charge transfer"
    dac = get_domain_profile("dac_her").novelty.targeted_query_variants(core)
    assert dac == (
        core,
        f"{core} hydrogen evolution reaction mechanism",
        f"{core} dual atom catalyst nitrogen coordination",
    )

    sers = get_domain_profile("sers_au_ag").novelty.targeted_query_variants(core)
    assert sers == (
        core,
        f"{core} SERS mechanism",
        f"{core} Au Ag plasmonic nanostructure",
    )
    joined = " ".join(sers).lower()
    assert "hydrogen evolution" not in joined
    assert "dual atom catalyst" not in joined
    assert "nitrogen coordination" not in joined


def test_gap_action_matrix_is_explicit_for_all_status_lattice_v2_states() -> None:
    analyzer = NoveltyGapAnalyzer(domain_profile=get_domain_profile("sers_au_ag"))
    expected = {
        "PLAUSIBLY_NOVEL": "keep",
        "NEW_COMBINATION_OF_KNOWN_EFFECTS": "keep",
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": "targeted_search_only",
        "INSUFFICIENT_SEARCH_EVIDENCE": "targeted_search_then_refine",
        "CONFLICTING_PRIOR_ART": "refine_away_from_conflict",
        "WELL_ESTABLISHED": "targeted_search_then_refine",
        "LITERATURE_SUPPORTED_EXTENSION": "targeted_search_then_refine",
    }
    for status, action in expected.items():
        card = type("Card", (), {"status": status})()
        assert analyzer._action(card) == action

    unknown = type("Card", (), {"status": "UNKNOWN_STATUS"})()
    with pytest.raises(ValueError, match="unsupported external novelty status"):
        analyzer._action(unknown)


def test_structured_targeted_queries_bind_exact_claim_ids() -> None:
    analyzer = NoveltyGapAnalyzer(
        queries_per_gap=4,
        domain_profile=get_domain_profile("sers_au_ag"),
    )
    card = type(
        "Card",
        (),
        {"hypothesis_id": "h1", "contextual_conflict_work_ids": []},
    )()
    plan = _plan(("c1", "c2"))
    # Avoid colliding with the generated query text in this focused provenance test.
    plan = plan.model_copy(update={"queries": []})
    targets = [
        _review("c1", text="gap geometry controls hotspot sampling"),
        _review("c2", text="solution drying changes analyte redistribution"),
    ]

    queries = analyzer._queries(card, targets, plan)
    assert [row.claim_id for row in queries] == ["c1", "c2", "c1", "c2"]
    assert [row.query_role for row in queries] == [
        "relation_primary",
        "relation_primary",
        "relation_variant",
        "relation_variant",
    ]


def test_novelty_gap_rejects_unbound_targeted_query_claim() -> None:
    with pytest.raises(ValidationError, match="targeted query claim_id"):
        NoveltyGap(
            gap_id="g1",
            hypothesis_id="h1",
            source_external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
            action="targeted_search_only",
            target_claim_ids=["c1"],
            differentiator="relation gap",
            targeted_queries=[
                TargetedGapQuery(
                    claim_id="c2",
                    query_role="relation_primary",
                    query_text="query",
                )
            ],
        )


def test_augmented_query_plan_uses_structured_query_claim_provenance() -> None:
    base = _plan(("c1", "c2"))
    gap = NoveltyGap(
        gap_id="g1",
        hypothesis_id="h1",
        source_external_status="KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
        action="targeted_search_only",
        target_claim_ids=["c1", "c2"],
        differentiator="relation gap",
        targeted_queries=[
            TargetedGapQuery(
                claim_id="c1",
                query_role="relation_primary",
                query_text="query one",
            ),
            TargetedGapQuery(
                claim_id="c2",
                query_role="scope_check",
                query_text="query two",
            ),
        ],
    )

    augmented, delta = build_augmented_query_plan(base, gap)
    assert [row.claim_id for row in delta.queries] == ["c1", "c2"]
    assert [row.query_text for row in delta.queries] == ["query one", "query two"]
    assert [row.claim_id for row in augmented.queries[-2:]] == ["c1", "c2"]

def test_query_budget_gives_each_target_a_primary_before_variants() -> None:
    analyzer = NoveltyGapAnalyzer(
        queries_per_gap=3,
        domain_profile=get_domain_profile("sers_au_ag"),
    )
    card = type(
        "Card",
        (),
        {"hypothesis_id": "h1", "contextual_conflict_work_ids": []},
    )()
    plan = _plan(("c1", "c2")).model_copy(update={"queries": []})
    targets = [
        _review("c1", text="gap geometry controls hotspot sampling"),
        _review("c2", text="solution drying changes analyte redistribution"),
    ]

    queries = analyzer._queries(card, targets, plan)
    assert [row.claim_id for row in queries[:2]] == ["c1", "c2"]
    assert [row.query_role for row in queries[:2]] == [
        "relation_primary",
        "relation_primary",
    ]


def test_existing_query_dedup_is_claim_scoped() -> None:
    analyzer = NoveltyGapAnalyzer(
        queries_per_gap=2,
        domain_profile=get_domain_profile("sers_au_ag"),
    )
    card = type(
        "Card",
        (),
        {"hypothesis_id": "h1", "contextual_conflict_work_ids": []},
    )()
    targets = [
        _review("c1", text="same relation text"),
        _review("c2", text="same relation text"),
    ]
    base = _plan(("c1", "c2"))
    base = base.model_copy(
        update={
            "queries": [
                LiteratureQuery(
                    query_id="existing-c1",
                    hypothesis_id="h1",
                    claim_id="c1",
                    query_kind="claim_variant",
                    query_text="same relation text",
                )
            ]
        }
    )

    queries = analyzer._queries(card, targets, base)
    assert queries
    assert queries[0].claim_id == "c2"
    assert queries[0].query_text == "same relation text"

