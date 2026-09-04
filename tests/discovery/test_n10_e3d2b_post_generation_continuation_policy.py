from copy import deepcopy

import pytest

from pipeline_core.discovery.n10_post_generation_continuation_policy import (
    MAX_POST_GENERATION_CONTINUATION_DEPTH,
    post_generation_continuation_directive_from_gate_row,
)


def _row(
    *,
    selection="CONDITIONAL",
    action="REFINE_NOVELTY_BEARING_SPECIFICATION",
    base_action="REFINE_NOVELTY_BEARING_SPECIFICATION",
    positive=False,
    fallback=False,
):
    return {
        "hypothesis_id":
            "hypothesis:test",

        "selection_class":
            selection,

        "action":
            action,

        "base_aggregation_action":
            base_action,

        "positive_nonobviousness_authority":
            positive,

        "fallback_allowed":
            fallback,

        "unresolved_claim_ids":
            ["claim:1"],
    }


def test_exact_specification_repair_at_depth_zero_allows_one_continuation():
    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=_row(),
            continuation_depth=0,
        )
    )

    assert (
        result.allow_bounded_continuation
        is True
    )

    assert result.current_depth == 0
    assert result.next_depth == 1

    assert (
        result.terminal_due_to_depth_limit
        is False
    )

    assert (
        result.fresh_post_generation_n10_required
        is True
    )

    assert (
        result.reason_code
        == (
            "allow_one_post_generation_"
            "novelty_specification_repair"
        )
    )


@pytest.mark.parametrize(
    "depth",
    [
        MAX_POST_GENERATION_CONTINUATION_DEPTH,
        MAX_POST_GENERATION_CONTINUATION_DEPTH + 1,
        10,
    ],
)
def test_depth_limit_blocks_further_continuation(
    depth,
):
    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=_row(),
            continuation_depth=depth,
        )
    )

    assert (
        result.allow_bounded_continuation
        is False
    )

    assert result.next_depth is None

    assert (
        result.terminal_due_to_depth_limit
        is True
    )

    assert (
        result.fresh_post_generation_n10_required
        is False
    )

    assert (
        result.reason_code
        == (
            "post_generation_specification_"
            "repair_depth_exhausted"
        )
    )


@pytest.mark.parametrize(
    "selection",
    [
        "ELIGIBLE",
        "INELIGIBLE",
        "",
        None,
    ],
)
def test_nonconditional_states_never_continue(
    selection,
):
    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=_row(
                selection=selection,
            ),
            continuation_depth=0,
        )
    )

    assert (
        result.allow_bounded_continuation
        is False
    )

    assert (
        result.terminal_due_to_depth_limit
        is False
    )


@pytest.mark.parametrize(
    "action",
    [
        "RESOLVE_NOVELTY_BEARING_PRIOR_ART_RELATION",
        "RESOLVE_NOVELTY_BEARING_EVIDENCE",
        "REMOVE_OR_REAXIS_ROUTINE_NOVELTY_BRANCH",
        "REFINE_ATOMIC_NONOBVIOUSNESS_SPECIFICATION",
        None,
        "",
    ],
)
def test_other_actions_do_not_continue(
    action,
):
    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=_row(
                action=action,
                base_action=action,
            ),
            continuation_depth=0,
        )
    )

    assert (
        result.allow_bounded_continuation
        is False
    )

    assert (
        result.reason_code
        == (
            "post_generation_action_"
            "not_continuation_eligible"
        )
    )


def test_conflicting_action_fields_fail_closed():
    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=_row(
                action=(
                    "REFINE_NOVELTY_BEARING_SPECIFICATION"
                ),
                base_action=(
                    "RESOLVE_NOVELTY_BEARING_EVIDENCE"
                ),
            ),
            continuation_depth=0,
        )
    )

    assert (
        result.allow_bounded_continuation
        is False
    )

    assert (
        result.reason_code
        == (
            "conflicting_post_generation_"
            "resolution_actions"
        )
    )


def test_positive_conditional_corruption_fails_closed():
    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=_row(
                positive=True,
            ),
            continuation_depth=0,
        )
    )

    assert (
        result.allow_bounded_continuation
        is False
    )


def test_fallback_conditional_corruption_fails_closed():
    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=_row(
                fallback=True,
            ),
            continuation_depth=0,
        )
    )

    assert (
        result.allow_bounded_continuation
        is False
    )


@pytest.mark.parametrize(
    "depth",
    [
        -1,
        True,
        False,
        1.0,
        "0",
        None,
    ],
)
def test_invalid_depth_is_rejected(
    depth,
):
    with pytest.raises(
        ValueError,
        match="non-negative integer",
    ):
        post_generation_continuation_directive_from_gate_row(
            gate_row=_row(),
            continuation_depth=depth,
        )


def test_policy_does_not_mutate_gate_row():
    row = _row()

    before = deepcopy(
        row
    )

    post_generation_continuation_directive_from_gate_row(
        gate_row=row,
        continuation_depth=0,
    )

    assert row == before


def test_missing_base_action_can_use_matching_action():
    row = _row()

    row.pop(
        "base_aggregation_action"
    )

    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=row,
            continuation_depth=0,
        )
    )

    assert (
        result.allow_bounded_continuation
        is True
    )


def test_missing_action_can_use_matching_base_action():
    row = _row()

    row.pop(
        "action"
    )

    result = (
        post_generation_continuation_directive_from_gate_row(
            gate_row=row,
            continuation_depth=0,
        )
    )

    assert (
        result.allow_bounded_continuation
        is True
    )
