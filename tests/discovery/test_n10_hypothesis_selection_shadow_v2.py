import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.hypothesis_selection_shadow_v2 import (
    build_hypothesis_selection_shadow_v2,
)


def _claim(
    claim_id,
    role,
    *,
    kind="mediator",
    importance="supporting",
    basis=None,
    components=None,
):
    return NoveltyClaim(
        claim_id=claim_id,
        hypothesis_id="hypothesis:1",
        claim_rank=1,
        kind=kind,
        importance=importance,
        novelty_selection_role=role,
        text=claim_id,
        rationale="Synthetic.",
        higher_order_relation_basis=(
            list(basis or [])
        ),
        higher_order_component_claim_ids=(
            list(components or [])
        ),
    )


def _plan(claims):
    return LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="plansha",
        source_portfolio_id="portfolio:1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:1",
                title="Synthetic",
                claims=claims,
            )
        ],
    )


def test_known_enabling_components_plus_pno_composite_is_eligible_shadow():
    plan = _plan(
        [
            _claim(
                "ab",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "bc",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "abc",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=[
                    "A is linked to C through B."
                ],
                components=[
                    "ab",
                    "bc",
                ],
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "ab": "SATURATED_PRIOR_ART",
            "bc": "ROUTINE_FROM_PRIOR_ART",
            "abc": "POTENTIALLY_NON_OBVIOUS",
        },
    )

    gate = result["hypotheses"][0]

    assert gate["selection_class"] == "ELIGIBLE"

    assert (
        gate[
            "shadow_positive_nonobviousness_authority"
        ]
        is True
    )

    # Shadow v2 must NEVER grant production fallback.
    assert gate["fallback_allowed"] is False
    assert gate["production_authority"] is False


def test_nested_routine_novelty_branch_blocks_composite():
    plan = _plan(
        [
            _claim(
                "ab",
                "NOVELTY_BEARING",
            ),
            _claim(
                "abc",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=[
                    "A is linked to C through B."
                ],
                components=["ab"],
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "ab": "SATURATED_PRIOR_ART",
            "abc": "POTENTIALLY_NON_OBVIOUS",
        },
    )

    gate = result["hypotheses"][0]

    assert (
        gate["selection_class"]
        == "INELIGIBLE"
    )

    assert gate["blocking_claim_ids"] == [
        "ab"
    ]

    assert (
        gate[
            "nested_novelty_bearing_component_ids"
        ]
        == ["ab"]
    )


def test_unresolved_role_downgrades_eligible_to_conditional():
    plan = _plan(
        [
            _claim(
                "known",
                "REQUIRED_ENABLING_RELATION",
            ),
            _claim(
                "unknown-role",
                None,
            ),
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "known": "SATURATED_PRIOR_ART",
            "unknown-role":
                "POTENTIALLY_NON_OBVIOUS",
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    gate = result["hypotheses"][0]

    assert (
        gate["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        gate[
            "shadow_positive_nonobviousness_authority"
        ]
        is False
    )

    assert (
        gate[
            "unresolved_selection_role_claim_ids"
        ]
        == ["unknown-role"]
    )


def test_unresolved_role_does_not_rescue_decisive_ineligible():
    plan = _plan(
        [
            _claim(
                "routine",
                "NOVELTY_BEARING",
                importance="core",
            ),
            _claim(
                "unknown-role",
                None,
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "routine":
                "ROUTINE_FROM_PRIOR_ART",
            "unknown-role":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    gate = result["hypotheses"][0]

    assert (
        gate["selection_class"]
        == "INELIGIBLE"
    )


def test_missing_atomic_outcome_is_rejected():
    plan = _plan(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing atomic nonobviousness outcome",
    ):
        build_hypothesis_selection_shadow_v2(
            query_plan=plan,
            atomic_outcomes={},
        )


def test_unknown_atomic_outcome_claim_is_rejected():
    plan = _plan(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="unknown claim_id",
    ):
        build_hypothesis_selection_shadow_v2(
            query_plan=plan,
            atomic_outcomes={
                "novel":
                    "POTENTIALLY_NON_OBVIOUS",
                "other":
                    "POTENTIALLY_NON_OBVIOUS",
            },
        )


def test_unsupported_outcome_is_rejected():
    plan = _plan(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="unsupported atomic",
    ):
        build_hypothesis_selection_shadow_v2(
            query_plan=plan,
            atomic_outcomes={
                "novel": "NO_MATCH_FOUND",
            },
        )


def test_missing_composite_basis_is_conditional_not_positive():
    plan = _plan(
        [
            _claim(
                "abc",
                "NOVELTY_BEARING",
                kind="composite",
                importance="core",
                basis=[],
            ),
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "abc":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    gate = result["hypotheses"][0]

    assert (
        gate["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        gate[
            "shadow_positive_nonobviousness_authority"
        ]
        is False
    )


def test_shadow_policy_explicitly_has_no_fallback_authority():
    plan = _plan(
        [
            _claim(
                "novel",
                "NOVELTY_BEARING",
                importance="core",
            )
        ]
    )

    result = build_hypothesis_selection_shadow_v2(
        query_plan=plan,
        atomic_outcomes={
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    assert result["shadow_only"] is True
    assert result["production_authority"] is False

    assert (
        result[
            "alpha6_original_fallback_authority"
        ]
        is False
    )

    assert (
        result["policy"][
            "fallback_allowed_always_false_in_shadow_v2"
        ]
        is True
    )
