from __future__ import annotations

import inspect
from types import SimpleNamespace

from domains.registry import get_domain_profile

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQuery,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    LiteratureQueryPlanner,
    NoveltyClaimDecomposer,
)
from pipeline_core.discovery.novelty_gap_analysis import (
    NoveltyGapAnalyzer,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    TargetedGapQuery,
)
from pipeline_core.discovery.targeted_novelty_retrieval import (
    _targeted_query_kind,
)


def _analyzer() -> NoveltyGapAnalyzer:
    return NoveltyGapAnalyzer(
        domain_profile=get_domain_profile(
            "sers_au_ag"
        )
    )


def _gap_review(
    claim_id: str = "claim:gap",
):
    return SimpleNamespace(
        importance="core",
        status="COMPONENTS_ONLY",
        claim_id=claim_id,
        claim_text=(
            "Gold core size moderates the relationship "
            "between silver shell thickness and SERS activity, "
            "so the shell thickness associated with maximal "
            "SERS activity is core size dependent."
        ),
    )


def _gap_card(
    *,
    status: str = (
        "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
    ),
    relational_gap_kind: str = (
        "HIGHER_ORDER_RELATIONAL_GAP"
    ),
):
    review = _gap_review()

    return SimpleNamespace(
        status=status,
        relational_gap_kind=relational_gap_kind,
        higher_order_relational_gap_claim_ids=[
            review.claim_id
        ],
        claim_reviews=[
            review
        ],
    )


def test_exact_verification_contract_values_are_explicit():
    assert (
        "claim_exact_verification"
        in str(
            LiteratureQuery.model_fields[
                "query_kind"
            ].annotation
        )
    )

    assert (
        "exact_higher_order_verification"
        in str(
            TargetedGapQuery.model_fields[
                "query_role"
            ].annotation
        )
    )


def test_exact_verification_requires_kcrg_and_horg():
    analyzer = _analyzer()

    targets = (
        analyzer
        ._exact_higher_order_verification_targets(
            _gap_card()
        )
    )

    assert targets is not None
    assert [
        row.claim_id
        for row in targets
    ] == [
        "claim:gap"
    ]

    assert (
        analyzer
        ._exact_higher_order_verification_targets(
            _gap_card(
                status=(
                    "NEW_COMBINATION_OF_KNOWN_EFFECTS"
                )
            )
        )
        is None
    )

    assert (
        analyzer
        ._exact_higher_order_verification_targets(
            _gap_card(
                relational_gap_kind="NONE"
            )
        )
        is None
    )


def test_exact_verification_uses_only_recorded_core_component_gap_claims():
    analyzer = _analyzer()

    good = _gap_review(
        "claim:good"
    )

    non_core = SimpleNamespace(
        **{
            **_gap_review(
                "claim:noncore"
            ).__dict__,
            "importance": "supporting",
        }
    )

    partial = SimpleNamespace(
        **{
            **_gap_review(
                "claim:partial"
            ).__dict__,
            "status": "PARTIAL_PRIOR_ART",
        }
    )

    unrecorded = _gap_review(
        "claim:unrecorded"
    )

    card = SimpleNamespace(
        status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        relational_gap_kind=(
            "HIGHER_ORDER_RELATIONAL_GAP"
        ),
        higher_order_relational_gap_claim_ids=[
            "claim:good",
            "claim:noncore",
            "claim:partial",
        ],
        claim_reviews=[
            good,
            non_core,
            partial,
            unrecorded,
        ],
    )

    targets = (
        analyzer
        ._exact_higher_order_verification_targets(
            card
        )
    )

    assert targets is not None

    assert [
        row.claim_id
        for row in targets
    ] == [
        "claim:good"
    ]


def test_exact_verification_builds_one_full_relation_query_per_target():
    analyzer = _analyzer()

    review = _gap_review()

    rows = (
        analyzer
        ._exact_higher_order_verification_queries(
            [review],
            SimpleNamespace(
                queries=[]
            ),
        )
    )

    assert len(rows) == 1

    query = rows[0]

    assert (
        query.query_role
        == "exact_higher_order_verification"
    )

    assert query.claim_id == review.claim_id

    expected = " ".join(
        review.claim_text.split()
    )[:300]

    assert query.query_text == expected

    lower = query.query_text.lower()

    assert "gold core size" in lower
    assert "moderates the relationship" in lower
    assert "silver shell thickness" in lower
    assert "sers activity" in lower
    assert "core size dependent" in lower


def test_exact_verification_does_not_repeat_identical_existing_query():
    analyzer = _analyzer()

    review = _gap_review()

    exact = " ".join(
        review.claim_text.split()
    )[:300]

    existing_plan = SimpleNamespace(
        queries=[
            SimpleNamespace(
                claim_id=review.claim_id,
                query_text=exact,
            )
        ]
    )

    rows = (
        analyzer
        ._exact_higher_order_verification_queries(
            [review],
            existing_plan,
        )
    )

    assert rows == []


def test_exact_verification_reuses_existing_max_target_claim_bound():
    analyzer = NoveltyGapAnalyzer(
        max_target_claims=2,
        domain_profile=get_domain_profile(
            "sers_au_ag"
        ),
    )

    reviews = [
        _gap_review("claim:1"),
        _gap_review("claim:2"),
        _gap_review("claim:3"),
    ]

    card = SimpleNamespace(
        status=(
            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"
        ),
        relational_gap_kind=(
            "HIGHER_ORDER_RELATIONAL_GAP"
        ),
        higher_order_relational_gap_claim_ids=[
            row.claim_id
            for row in reviews
        ],
        claim_reviews=reviews,
    )

    targets = (
        analyzer
        ._exact_higher_order_verification_targets(
            card
        )
    )

    assert targets is not None

    assert [
        row.claim_id
        for row in targets
    ] == [
        "claim:1",
        "claim:2",
    ]


def test_exact_targeted_role_maps_to_explicit_literature_query_kind():
    assert (
        _targeted_query_kind(
            SimpleNamespace(
                query_role=(
                    "exact_higher_order_verification"
                )
            )
        )
        == "claim_exact_verification"
    )

    for role in (
        "relation_primary",
        "relation_variant",
        "scope_check",
    ):
        assert (
            _targeted_query_kind(
                SimpleNamespace(
                    query_role=role
                )
            )
            == "claim_variant"
        )


def test_exact_verification_is_not_a_first_pass_query_kind():
    planner_source = inspect.getsource(
        LiteratureQueryPlanner.build
    )

    decomposer_source = inspect.getsource(
        NoveltyClaimDecomposer.decompose
    )

    assert (
        "claim_exact_verification"
        not in planner_source
    )

    assert (
        "claim_exact_verification"
        not in decomposer_source
    )

    assert (
        "exact_higher_order_verification"
        not in planner_source
    )

    assert (
        "exact_higher_order_verification"
        not in decomposer_source
    )


def test_build_routes_exact_mode_through_exact_query_helper():
    source = inspect.getsource(
        NoveltyGapAnalyzer.build
    )

    assert (
        "_exact_higher_order_verification_targets"
        in source
    )

    assert (
        "_exact_higher_order_verification_queries"
        in source
    )
