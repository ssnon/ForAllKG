from pipeline_core.discovery.nonobviousness_shadow_action_routing import (
    route_shadow_resolution_actions,
)


def _row(
    claim_id,
    role,
    outcome,
    reasons=(),
):
    return {
        "claim_id": claim_id,
        "novelty_selection_role": role,
        "nonobviousness_outcome": outcome,
        "outcome_reason_codes": list(reasons),
    }


def test_partial_prior_art_is_not_mislabeled_as_specification_problem():
    result = route_shadow_resolution_actions(
        selection_class="CONDITIONAL",
        atomic_claims=[
            _row(
                "composite",
                "NOVELTY_BEARING",
                "NEEDS_REFINEMENT",
                (
                    "partial_prior_art_requires_resolution",
                ),
            )
        ],
        fallback_action=(
            "REFINE_NOVELTY_BEARING_SPECIFICATION"
        ),
    )

    assert result.primary_action == (
        "RESOLVE_NOVELTY_BEARING_PRIOR_ART_RELATION"
    )


def test_true_atomic_under_specification_routes_to_refinement():
    result = route_shadow_resolution_actions(
        selection_class="CONDITIONAL",
        atomic_claims=[
            _row(
                "branch",
                "NOVELTY_BEARING",
                "NEEDS_REFINEMENT",
                (
                    "atomic_specification_incomplete",
                    "missing_specification_field:"
                    "predicted_observation",
                ),
            )
        ],
        fallback_action="fallback",
    )

    assert result.primary_action == (
        "REFINE_NOVELTY_BEARING_SPECIFICATION"
    )


def test_known_enabling_relation_needs_no_resolution():
    result = route_shadow_resolution_actions(
        selection_class="ELIGIBLE",
        atomic_claims=[
            _row(
                "known",
                "REQUIRED_ENABLING_RELATION",
                "SATURATED_PRIOR_ART",
            )
        ],
        fallback_action=(
            "KEEP_ROLE_AWARE_NONOBVIOUS_CANDIDATE"
        ),
    )

    assert result.resolution_requirements == ()

    assert result.primary_action == (
        "KEEP_ROLE_AWARE_NONOBVIOUS_CANDIDATE"
    )


def test_under_specified_enabling_relation_is_distinct():
    result = route_shadow_resolution_actions(
        selection_class="CONDITIONAL",
        atomic_claims=[
            _row(
                "enabling",
                "REQUIRED_ENABLING_RELATION",
                "NEEDS_REFINEMENT",
                (
                    "atomic_specification_incomplete",
                ),
            )
        ],
        fallback_action="fallback",
    )

    assert result.primary_action == (
        "REFINE_REQUIRED_ENABLING_"
        "RELATION_SPECIFICATION"
    )


def test_insufficient_novelty_evidence_routes_to_evidence_resolution():
    result = route_shadow_resolution_actions(
        selection_class="CONDITIONAL",
        atomic_claims=[
            _row(
                "novel",
                "NOVELTY_BEARING",
                "INSUFFICIENT_FOR_JUDGMENT",
                (
                    "candidate_not_ready_for_adjudication",
                ),
            )
        ],
        fallback_action="fallback",
    )

    assert result.primary_action == (
        "RESOLVE_NOVELTY_BEARING_EVIDENCE"
    )


def test_routine_novelty_branch_remains_decisive():
    result = route_shadow_resolution_actions(
        selection_class="INELIGIBLE",
        atomic_claims=[
            _row(
                "routine",
                "NOVELTY_BEARING",
                "ROUTINE_FROM_PRIOR_ART",
            ),
            _row(
                "other",
                "NOVELTY_BEARING",
                "INSUFFICIENT_FOR_JUDGMENT",
            ),
        ],
        fallback_action="fallback",
    )

    assert result.primary_action == (
        "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH"
    )


def test_testing_prediction_does_not_become_selection_blocker():
    result = route_shadow_resolution_actions(
        selection_class="ELIGIBLE",
        atomic_claims=[
            _row(
                "prediction",
                "TESTING_PREDICTION",
                "INSUFFICIENT_FOR_JUDGMENT",
            )
        ],
        fallback_action=(
            "KEEP_ROLE_AWARE_NONOBVIOUS_CANDIDATE"
        ),
    )

    assert result.resolution_requirements == ()
