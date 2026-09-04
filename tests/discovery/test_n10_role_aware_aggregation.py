import pytest

from pipeline_core.discovery.novelty_selection_aggregation import (
    RoleAwareAtomicClaim,
    aggregate_role_aware_nonobviousness,
)


def _claim(
    claim_id,
    role,
    outcome,
):
    return RoleAwareAtomicClaim(
        claim_id=claim_id,
        novelty_selection_role=role,
        nonobviousness_outcome=outcome,
    )


def test_known_enabling_plus_pno_novelty_is_eligible():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel",
                "NOVELTY_BEARING",
                "POTENTIALLY_NON_OBVIOUS",
            ),
            _claim(
                "known",
                "REQUIRED_ENABLING_RELATION",
                "SATURATED_PRIOR_ART",
            ),
        )
    )

    assert result.selection_class == "ELIGIBLE"
    assert result.positive_nonobviousness_authority is True
    assert result.blocking_claim_ids == ()


def test_routine_enabling_relation_does_not_destroy_novelty():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel",
                "NOVELTY_BEARING",
                "POTENTIALLY_NON_OBVIOUS",
            ),
            _claim(
                "known",
                "REQUIRED_ENABLING_RELATION",
                "ROUTINE_FROM_PRIOR_ART",
            ),
        )
    )

    assert result.selection_class == "ELIGIBLE"


def test_unresolved_novelty_bearing_branch_is_conditional():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel",
                "NOVELTY_BEARING",
                "INSUFFICIENT_FOR_JUDGMENT",
            ),
            _claim(
                "known",
                "REQUIRED_ENABLING_RELATION",
                "SATURATED_PRIOR_ART",
            ),
        )
    )

    assert result.selection_class == "CONDITIONAL"
    assert result.positive_nonobviousness_authority is False
    assert result.action == (
        "RESOLVE_NOVELTY_BEARING_EVIDENCE"
    )
    assert result.unresolved_claim_ids == ("novel",)


def test_under_specified_novelty_branch_is_conditional_refinement():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel",
                "NOVELTY_BEARING",
                "NEEDS_REFINEMENT",
            ),
        )
    )

    assert result.selection_class == "CONDITIONAL"
    assert result.action == (
        "REFINE_NOVELTY_BEARING_SPECIFICATION"
    )
    assert result.positive_nonobviousness_authority is False


@pytest.mark.parametrize(
    "known_outcome",
    [
        "SATURATED_PRIOR_ART",
        "ROUTINE_FROM_PRIOR_ART",
    ],
)
def test_saturated_or_routine_novelty_branch_is_ineligible(
    known_outcome,
):
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel",
                "NOVELTY_BEARING",
                known_outcome,
            ),
        )
    )

    assert result.selection_class == "INELIGIBLE"
    assert result.blocking_claim_ids == ("novel",)
    assert result.action == (
        "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH"
    )


def test_one_positive_branch_cannot_hide_second_routine_novelty_branch():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel-a",
                "NOVELTY_BEARING",
                "POTENTIALLY_NON_OBVIOUS",
            ),
            _claim(
                "novel-b",
                "NOVELTY_BEARING",
                "SATURATED_PRIOR_ART",
            ),
        )
    )

    assert result.selection_class == "INELIGIBLE"
    assert result.blocking_claim_ids == ("novel-b",)


def test_saturated_testing_prediction_does_not_destroy_novelty():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel",
                "NOVELTY_BEARING",
                "POTENTIALLY_NON_OBVIOUS",
            ),
            _claim(
                "test",
                "TESTING_PREDICTION",
                "SATURATED_PRIOR_ART",
            ),
        )
    )

    assert result.selection_class == "ELIGIBLE"
    assert result.testing_prediction_claim_ids == ("test",)


def test_insufficient_supporting_auxiliary_does_not_destroy_novelty():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel",
                "NOVELTY_BEARING",
                "POTENTIALLY_NON_OBVIOUS",
            ),
            _claim(
                "aux",
                "AUXILIARY",
                "INSUFFICIENT_FOR_JUDGMENT",
            ),
        )
    )

    assert result.selection_class == "ELIGIBLE"
    assert result.auxiliary_claim_ids == ("aux",)


@pytest.mark.parametrize(
    "enabling_outcome",
    [
        "INSUFFICIENT_FOR_JUDGMENT",
        "NEEDS_REFINEMENT",
    ],
)
def test_unresolved_required_relation_prevents_eligible_authority(
    enabling_outcome,
):
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel",
                "NOVELTY_BEARING",
                "POTENTIALLY_NON_OBVIOUS",
            ),
            _claim(
                "required",
                "REQUIRED_ENABLING_RELATION",
                enabling_outcome,
            ),
        )
    )

    assert result.selection_class == "CONDITIONAL"
    assert result.positive_nonobviousness_authority is False
    assert result.action == (
        "RESOLVE_REQUIRED_ENABLING_RELATION"
    )
    assert result.unresolved_claim_ids == ("required",)


def test_no_novelty_bearing_claim_requires_role_refinement():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "known",
                "REQUIRED_ENABLING_RELATION",
                "SATURATED_PRIOR_ART",
            ),
            _claim(
                "test",
                "TESTING_PREDICTION",
                "POTENTIALLY_NON_OBVIOUS",
            ),
        )
    )

    assert result.selection_class == "INELIGIBLE"
    assert result.positive_nonobviousness_authority is False
    assert result.action == (
        "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION"
    )
    assert "no_novelty_bearing_claims" in result.reason_codes


def test_all_novelty_branches_insufficient_is_never_eligible():
    result = aggregate_role_aware_nonobviousness(
        (
            _claim(
                "novel-a",
                "NOVELTY_BEARING",
                "INSUFFICIENT_FOR_JUDGMENT",
            ),
            _claim(
                "novel-b",
                "NOVELTY_BEARING",
                "INSUFFICIENT_FOR_JUDGMENT",
            ),
        )
    )

    assert result.selection_class == "CONDITIONAL"
    assert result.positive_nonobviousness_authority is False


def test_empty_claim_set_fails_closed():
    result = aggregate_role_aware_nonobviousness(())

    assert result.selection_class == "INELIGIBLE"
    assert result.positive_nonobviousness_authority is False
    assert "no_atomic_claims" in result.reason_codes


def test_unknown_selection_role_is_rejected():
    malformed = RoleAwareAtomicClaim(
        claim_id="bad-role",
        novelty_selection_role="UNKNOWN_ROLE",  # type: ignore[arg-type]
        nonobviousness_outcome=(
            "POTENTIALLY_NON_OBVIOUS"
        ),
    )

    with pytest.raises(
        ValueError,
        match="unsupported novelty selection role",
    ):
        aggregate_role_aware_nonobviousness(
            (malformed,)
        )


@pytest.mark.parametrize(
    "role",
    [
        "NOVELTY_BEARING",
        "REQUIRED_ENABLING_RELATION",
        "TESTING_PREDICTION",
        "AUXILIARY",
    ],
)
def test_unknown_outcome_is_rejected_for_every_role(
    role,
):
    malformed = RoleAwareAtomicClaim(
        claim_id="bad-outcome",
        novelty_selection_role=role,
        nonobviousness_outcome="UNKNOWN_OUTCOME",  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="unsupported nonobviousness outcome",
    ):
        aggregate_role_aware_nonobviousness(
            (malformed,)
        )


def test_malformed_auxiliary_outcome_cannot_hide_behind_pno_branch():
    claims = (
        _claim(
            "novel",
            "NOVELTY_BEARING",
            "POTENTIALLY_NON_OBVIOUS",
        ),
        RoleAwareAtomicClaim(
            claim_id="malformed-aux",
            novelty_selection_role="AUXILIARY",
            nonobviousness_outcome="UNKNOWN_OUTCOME",  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(
        ValueError,
        match="unsupported nonobviousness outcome",
    ):
        aggregate_role_aware_nonobviousness(
            claims
        )


def test_duplicate_claim_id_is_rejected():
    with pytest.raises(
        ValueError,
        match="duplicate role-aware atomic claim_id",
    ):
        aggregate_role_aware_nonobviousness(
            (
                _claim(
                    "same",
                    "NOVELTY_BEARING",
                    "POTENTIALLY_NON_OBVIOUS",
                ),
                _claim(
                    "same",
                    "AUXILIARY",
                    "SATURATED_PRIOR_ART",
                ),
            )
        )
