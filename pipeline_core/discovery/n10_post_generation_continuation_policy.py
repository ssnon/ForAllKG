"""Pure policy for bounded post-generation N10 continuation.

This policy does not:
- grant production authority;
- change N10 selection class;
- treat CONDITIONAL as positive;
- treat search-bounded absence as novelty;
- generate a hypothesis;
- perform retrieval;
- perform review.

It only determines whether one already-authoritative role-aware
post-generation CONDITIONAL state may receive exactly one additional
bounded specification-repair attempt.

Only REFINE_NOVELTY_BEARING_SPECIFICATION is currently continuation
eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REFINE_NOVELTY_BEARING_SPECIFICATION = (
    "REFINE_NOVELTY_BEARING_SPECIFICATION"
)

MAX_POST_GENERATION_CONTINUATION_DEPTH = 1


@dataclass(
    frozen=True,
)
class PostGenerationN10ContinuationDirective:
    allow_bounded_continuation: bool
    current_depth: int
    next_depth: int | None
    terminal_due_to_depth_limit: bool
    fresh_post_generation_n10_required: bool
    reason_code: str


def _no_continuation(
    *,
    depth: int,
    reason_code: str,
    terminal_due_to_depth_limit: bool = False,
) -> PostGenerationN10ContinuationDirective:
    return PostGenerationN10ContinuationDirective(
        allow_bounded_continuation=False,
        current_depth=depth,
        next_depth=None,
        terminal_due_to_depth_limit=(
            terminal_due_to_depth_limit
        ),
        fresh_post_generation_n10_required=False,
        reason_code=reason_code,
    )


def post_generation_continuation_directive_from_gate_row(
    *,
    gate_row: Mapping[str, Any],
    continuation_depth: int,
) -> PostGenerationN10ContinuationDirective:
    """Return a fail-closed bounded continuation directive.

    The caller is responsible for validating that ``gate_row`` came from
    an authoritative role-aware post-generation v2 gate.

    A continuation is allowed only for the exact unresolved specification
    state:

        CONDITIONAL
        + REFINE_NOVELTY_BEARING_SPECIFICATION
        + positive authority False
        + fallback False
        + depth == 0

    Any candidate produced by that continuation must receive fresh
    post-generation N10 adjudication before it can enter the final
    portfolio.
    """

    if (
        isinstance(
            continuation_depth,
            bool,
        )
        or not isinstance(
            continuation_depth,
            int,
        )
        or continuation_depth < 0
    ):
        raise ValueError(
            "continuation_depth must be a non-negative integer"
        )

    if not isinstance(
        gate_row,
        Mapping,
    ):
        return _no_continuation(
            depth=continuation_depth,
            reason_code=(
                "invalid_post_generation_gate_row"
            ),
        )

    if (
        gate_row.get(
            "selection_class"
        )
        != "CONDITIONAL"
    ):
        return _no_continuation(
            depth=continuation_depth,
            reason_code=(
                "post_generation_state_not_conditional"
            ),
        )

    if (
        gate_row.get(
            "positive_nonobviousness_authority"
        )
        is not False
    ):
        return _no_continuation(
            depth=continuation_depth,
            reason_code=(
                "conditional_state_not_fail_closed_positive"
            ),
        )

    if (
        gate_row.get(
            "fallback_allowed"
        )
        is not False
    ):
        return _no_continuation(
            depth=continuation_depth,
            reason_code=(
                "conditional_state_not_fail_closed_fallback"
            ),
        )

    action = gate_row.get(
        "action"
    )

    base_action = gate_row.get(
        "base_aggregation_action"
    )

    if (
        action is not None
        and base_action is not None
        and action != base_action
    ):
        return _no_continuation(
            depth=continuation_depth,
            reason_code=(
                "conflicting_post_generation_resolution_actions"
            ),
        )

    resolved_action = (
        base_action
        if base_action is not None
        else action
    )

    if (
        resolved_action
        != REFINE_NOVELTY_BEARING_SPECIFICATION
    ):
        return _no_continuation(
            depth=continuation_depth,
            reason_code=(
                "post_generation_action_not_continuation_eligible"
            ),
        )

    if (
        continuation_depth
        >= MAX_POST_GENERATION_CONTINUATION_DEPTH
    ):
        return _no_continuation(
            depth=continuation_depth,
            terminal_due_to_depth_limit=True,
            reason_code=(
                "post_generation_specification_repair_depth_exhausted"
            ),
        )

    return PostGenerationN10ContinuationDirective(
        allow_bounded_continuation=True,
        current_depth=continuation_depth,
        next_depth=(
            continuation_depth + 1
        ),
        terminal_due_to_depth_limit=False,
        fresh_post_generation_n10_required=True,
        reason_code=(
            "allow_one_post_generation_"
            "novelty_specification_repair"
        ),
    )
