from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.nonobviousness_dual_run_comparison import (
    build_nonobviousness_dual_run_comparison,
)


def _claim(
    claim_id,
    role,
    *,
    importance="core",
    kind="mediator",
    basis=(),
    components=(),
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
        higher_order_relation_basis=list(
            basis
        ),
        higher_order_component_claim_ids=list(
            components
        ),
    )


def _plan(claims):
    return LiteratureQueryPlan(
        plan_id="plan:1",
        plan_sha256="plan-sha",
        source_portfolio_id="portfolio:1",
        claims=[
            HypothesisNoveltyClaims(
                hypothesis_id="hypothesis:1",
                title="Synthetic",
                claims=claims,
            )
        ],
    )


def _n9(
    claims,
    outcomes,
):
    decisions = []
    full_rows = []

    for claim in claims:
        outcome = outcomes[
            claim.claim_id
        ]

        if outcome == "POTENTIALLY_NON_OBVIOUS":
            state = "READY_FOR_CLOSURE"
            full_rows.append(
                {
                    "claim_id":
                        claim.claim_id,
                    "final_verdict":
                        "POTENTIALLY_NON_OBVIOUS",
                    "final_reason_codes":
                        ["synthetic_positive"],
                }
            )

        elif outcome == "ROUTINE_FROM_PRIOR_ART":
            state = "READY_FOR_CLOSURE"
            full_rows.append(
                {
                    "claim_id":
                        claim.claim_id,
                    "final_verdict":
                        "ROUTINE_FROM_PRIOR_ART",
                    "final_reason_codes":
                        ["synthetic_routine"],
                }
            )

        elif outcome == "SATURATED_PRIOR_ART":
            state = "SATURATED_PRIOR_ART"

        elif outcome == "INSUFFICIENT_FOR_JUDGMENT":
            state = "UNRESOLVED"

        elif outcome == "NEEDS_REFINEMENT":
            state = "NEEDS_REFINEMENT"

        else:
            raise ValueError(outcome)

        decisions.append(
            {
                "claim": {
                    "claim_id":
                        claim.claim_id,
                    "importance":
                        claim.importance,
                },
                "shadow_state":
                    state,
                "specification":
                    {
                        "reason_codes": [],
                        "missing_fields": [],
                    },
            }
        )

    intake = {
        "schema_version":
            "nonobviousness-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            "portfolio:1",
        "source_query_plan_id":
            "plan:1",
        "source_external_report_id":
            "report:1",
        "hypotheses": [
            {
                "hypothesis_id":
                    "hypothesis:1",
                "claims":
                    decisions,
            }
        ],
    }

    full = {
        "schema_version":
            "nonobviousness-full-shadow-v1",
        "shadow_only": True,
        "source_portfolio_id":
            "portfolio:1",
        "claims":
            full_rows,
    }

    return intake, full


def _compile(
    claims,
    outcomes,
):
    plan = _plan(claims)

    intake, full = _n9(
        claims,
        outcomes,
    )

    return build_nonobviousness_dual_run_comparison(
        query_plan=plan,
        intake_shadow=intake,
        full_shadow=full,
    )


def test_same_positive_case_agrees_but_v2_has_no_authority():
    claims = [
        _claim(
            "novel",
            "NOVELTY_BEARING",
        )
    ]

    result = _compile(
        claims,
        {
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    row = result["comparisons"][0]

    assert row["v1"]["selection_class"] == "ELIGIBLE"

    assert (
        row["v2_candidate"]["selection_class"]
        == "ELIGIBLE"
    )

    assert (
        row[
            "observed_production_fallback_allowed"
        ]
        is True
    )

    assert (
        row["candidate_has_production_authority"]
        is False
    )

    assert result["production_authority"] is False
    assert result["authority_policy"] == "v1_only"


def test_v2_can_be_more_permissive_without_changing_production():
    claims = [
        _claim(
            "known",
            "REQUIRED_ENABLING_RELATION",
            # Deliberately core: frozen v1 blocks this,
            # while v2 interprets the orthogonal role.
            importance="core",
        ),
        _claim(
            "composite",
            "NOVELTY_BEARING",
            importance="core",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=("known",),
        ),
    ]

    result = _compile(
        claims,
        {
            "known":
                "SATURATED_PRIOR_ART",
            "composite":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    row = result["comparisons"][0]

    assert row["v1"]["selection_class"] == "INELIGIBLE"

    assert (
        row["v2_candidate"]["selection_class"]
        == "ELIGIBLE"
    )

    # Critical E1 invariant:
    # candidate disagreement does not alter production.
    assert (
        row[
            "observed_production_fallback_allowed"
        ]
        is False
    )

    assert (
        row[
            "positive_authority_candidate_changed"
        ]
        is True
    )


def test_v2_can_be_more_conservative_without_changing_production():
    claims = [
        _claim(
            "enabling",
            "REQUIRED_ENABLING_RELATION",
            importance="supporting",
        ),
        _claim(
            "composite",
            "NOVELTY_BEARING",
            importance="core",
            kind="composite",
            basis=(
                "A is linked to C through B.",
            ),
            components=("enabling",),
        ),
    ]

    result = _compile(
        claims,
        {
            "enabling":
                "INSUFFICIENT_FOR_JUDGMENT",
            "composite":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    row = result["comparisons"][0]

    # Frozen v1 ignores the supporting branch.
    assert row["v1"]["selection_class"] == "ELIGIBLE"

    # v2 correctly requires unresolved enabling coherence.
    assert (
        row["v2_candidate"]["selection_class"]
        == "CONDITIONAL"
    )

    # Still report what production v1 actually does.
    assert (
        row[
            "observed_production_fallback_allowed"
        ]
        is True
    )

    assert (
        row["candidate_has_production_authority"]
        is False
    )


def test_missing_composite_basis_is_visible_as_v2_conservatism():
    claims = [
        _claim(
            "composite",
            "NOVELTY_BEARING",
            importance="core",
            kind="composite",
            basis=(),
        ),
    ]

    result = _compile(
        claims,
        {
            "composite":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    row = result["comparisons"][0]

    assert row["v1"]["selection_class"] == "ELIGIBLE"

    assert (
        row["v2_candidate"]["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        row[
            "observed_production_fallback_allowed"
        ]
        is True
    )


def test_routine_novelty_branch_agrees_ineligible():
    claims = [
        _claim(
            "routine",
            "NOVELTY_BEARING",
            importance="core",
        )
    ]

    result = _compile(
        claims,
        {
            "routine":
                "ROUTINE_FROM_PRIOR_ART",
        },
    )

    row = result["comparisons"][0]

    assert row["v1"]["selection_class"] == "INELIGIBLE"

    assert (
        row["v2_candidate"]["selection_class"]
        == "INELIGIBLE"
    )

    assert (
        row[
            "observed_production_fallback_allowed"
        ]
        is False
    )


def test_query_plan_intake_provenance_mismatch_fails_closed():
    claims = [
        _claim(
            "novel",
            "NOVELTY_BEARING",
        )
    ]

    plan = _plan(claims)

    intake, full = _n9(
        claims,
        {
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    intake["source_query_plan_id"] = "wrong-plan"

    import pytest

    with pytest.raises(
        ValueError,
        match="plan mismatch",
    ):
        build_nonobviousness_dual_run_comparison(
            query_plan=plan,
            intake_shadow=intake,
            full_shadow=full,
        )


def test_comparison_artifact_never_becomes_authority():
    claims = [
        _claim(
            "novel",
            "NOVELTY_BEARING",
        )
    ]

    result = _compile(
        claims,
        {
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    assert result["comparison_only"] is True
    assert result["production_authority"] is False

    assert (
        result["candidate_has_production_authority"]
        is False
    )

    assert result["authority_policy"] == "v1_only"
