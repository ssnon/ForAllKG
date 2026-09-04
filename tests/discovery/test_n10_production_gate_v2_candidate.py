import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaim,
)
from pipeline_core.discovery.nonobviousness_production_gate_v2_candidate import (
    build_nonobviousness_production_gate_v2_candidate,
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
        rationale="Synthetic E2C control.",
        higher_order_relation_basis=list(
            basis
        ),
        higher_order_component_claim_ids=list(
            components
        ),
    )


def _plan(
    claims,
):
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

        specification = {
            "reason_codes": [],
            "missing_fields": [],
        }

        if (
            outcome
            == "POTENTIALLY_NON_OBVIOUS"
        ):
            state = "READY_FOR_CLOSURE"

            full_rows.append(
                {
                    "claim_id":
                        claim.claim_id,

                    "final_verdict":
                        "POTENTIALLY_NON_OBVIOUS",

                    "final_reason_codes": [
                        "synthetic_positive"
                    ],
                }
            )

        elif (
            outcome
            == "ROUTINE_FROM_PRIOR_ART"
        ):
            state = "READY_FOR_CLOSURE"

            full_rows.append(
                {
                    "claim_id":
                        claim.claim_id,

                    "final_verdict":
                        "ROUTINE_FROM_PRIOR_ART",

                    "final_reason_codes": [
                        "synthetic_routine"
                    ],
                }
            )

        elif (
            outcome
            == "SATURATED_PRIOR_ART"
        ):
            state = "SATURATED_PRIOR_ART"

        elif (
            outcome
            == "INSUFFICIENT_FOR_JUDGMENT"
        ):
            state = "UNRESOLVED"

        elif (
            outcome
            == "NEEDS_REFINEMENT"
        ):
            state = "NEEDS_REFINEMENT"

            specification = {
                "reason_codes": [
                    "atomic_specification_incomplete",
                ],
                "missing_fields": [
                    "predicted_observation",
                ],
            }

        else:
            raise ValueError(
                outcome
            )

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
                    specification,
            }
        )

    intake = {
        "schema_version":
            "nonobviousness-shadow-v1",

        "shadow_only":
            True,

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

        "shadow_only":
            True,

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
    plan = _plan(
        claims
    )

    intake, full = _n9(
        claims,
        outcomes,
    )

    return (
        build_nonobviousness_production_gate_v2_candidate(
            query_plan=plan,
            intake_shadow=intake,
            full_shadow=full,
        )
    )


def test_known_enabling_plus_positive_composite_is_candidate_allowed():
    claims = [
        _claim(
            "known",
            "REQUIRED_ENABLING_RELATION",
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
            components=(
                "known",
            ),
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

    gate = result["gates"][0]

    assert (
        gate["selection_class"]
        == "ELIGIBLE"
    )

    assert (
        gate[
            "candidate_fallback_allowed"
        ]
        is True
    )

    assert (
        result[
            "candidate_fallback_allowed_count"
        ]
        == 1
    )

    # Candidate permission is not runtime authority.
    assert (
        result[
            "production_authority"
        ]
        is False
    )

    assert (
        result[
            "alpha6_original_fallback_authority"
        ]
        is False
    )


def test_unresolved_enabling_relation_blocks_candidate_fallback():
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
                "Explicit composite relation.",
            ),
            components=(
                "enabling",
            ),
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

    gate = result["gates"][0]

    assert (
        gate["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        gate[
            "candidate_fallback_allowed"
        ]
        is False
    )


def test_routine_novelty_branch_is_candidate_blocked():
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

    gate = result["gates"][0]

    assert (
        gate["selection_class"]
        == "INELIGIBLE"
    )

    assert (
        gate[
            "candidate_fallback_allowed"
        ]
        is False
    )


def test_missing_composite_basis_is_candidate_blocked():
    claims = [
        _claim(
            "composite",
            "NOVELTY_BEARING",
            importance="core",
            kind="composite",
            basis=(),
        )
    ]

    result = _compile(
        claims,
        {
            "composite":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    gate = result["gates"][0]

    assert (
        gate["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        gate[
            "candidate_fallback_allowed"
        ]
        is False
    )


def test_insufficient_novelty_is_candidate_blocked():
    claims = [
        _claim(
            "novel",
            "NOVELTY_BEARING",
            importance="core",
        )
    ]

    result = _compile(
        claims,
        {
            "novel":
                "INSUFFICIENT_FOR_JUDGMENT",
        },
    )

    gate = result["gates"][0]

    assert (
        gate["selection_class"]
        == "CONDITIONAL"
    )

    assert (
        gate[
            "candidate_fallback_allowed"
        ]
        is False
    )


def test_no_novelty_bearing_claim_is_candidate_blocked():
    claims = [
        _claim(
            "enabling",
            "REQUIRED_ENABLING_RELATION",
            importance="core",
        )
    ]

    result = _compile(
        claims,
        {
            "enabling":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    gate = result["gates"][0]

    assert (
        gate["selection_class"]
        == "INELIGIBLE"
    )

    assert (
        gate[
            "candidate_fallback_allowed"
        ]
        is False
    )


def test_candidate_artifact_never_has_production_authority():
    claims = [
        _claim(
            "novel",
            "NOVELTY_BEARING",
            importance="core",
        )
    ]

    result = _compile(
        claims,
        {
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    gate = result["gates"][0]

    assert (
        gate[
            "candidate_fallback_allowed"
        ]
        is True
    )

    assert (
        gate[
            "production_authority"
        ]
        is False
    )

    assert (
        result["candidate_only"]
        is True
    )

    assert (
        result["production_authority"]
        is False
    )

    assert (
        result["authority_policy"]
        == "none_candidate_only"
    )


def test_query_plan_intake_mismatch_fails_closed():
    claims = [
        _claim(
            "novel",
            "NOVELTY_BEARING",
        )
    ]

    plan = _plan(
        claims
    )

    intake, full = _n9(
        claims,
        {
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    intake[
        "source_query_plan_id"
    ] = "wrong-plan"

    with pytest.raises(
        ValueError,
        match="plan mismatch",
    ):
        build_nonobviousness_production_gate_v2_candidate(
            query_plan=plan,
            intake_shadow=intake,
            full_shadow=full,
        )


def test_portfolio_mismatch_fails_closed():
    claims = [
        _claim(
            "novel",
            "NOVELTY_BEARING",
        )
    ]

    plan = _plan(
        claims
    )

    intake, full = _n9(
        claims,
        {
            "novel":
                "POTENTIALLY_NON_OBVIOUS",
        },
    )

    full[
        "source_portfolio_id"
    ] = "wrong-portfolio"

    with pytest.raises(
        ValueError,
        match="portfolio mismatch",
    ):
        build_nonobviousness_production_gate_v2_candidate(
            query_plan=plan,
            intake_shadow=intake,
            full_shadow=full,
        )
