"""Fail-closed N10-v2 resolution directives for Alpha6.

This module does not mutate the Alpha6 gap plan and does not grant
novelty authority.  It translates one already-validated role-aware
N10 production decision into a bounded refinement directive.

Only REFINE_NOVELTY_BEARING_SPECIFICATION is currently integrated.
All other actions remain behaviorally inert here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REFINE_NOVELTY_BEARING_SPECIFICATION = (
    "REFINE_NOVELTY_BEARING_SPECIFICATION"
)


@dataclass(frozen=True)
class Alpha6N10ResolutionDirective:
    """Behavioral directive derived from a validated N10-v2 gate row."""

    force_bounded_refinement: bool = False
    use_source_external_without_targeted_search: bool = False
    bypass_resolved_candidate_external_exit: bool = False
    reason_code: str | None = None


NO_N10_ALPHA6_OVERRIDE = Alpha6N10ResolutionDirective()


def alpha6_resolution_directive_from_gate_row(
    gate_row: Mapping[str, Any] | None,
) -> Alpha6N10ResolutionDirective:
    """Return the narrowly scoped Alpha6 directive for one v2 gate row.

    The caller is still responsible for validating the enclosing
    scientific-novelty gate schema and production authority.

    Fail-closed conditions:
    - missing/malformed row;
    - anything other than CONDITIONAL;
    - fallback not explicitly False;
    - positive authority not explicitly False;
    - any action other than the frozen specification-refinement action.

    In all such cases no behavioral override is granted.
    """

    if not isinstance(gate_row, Mapping):
        return NO_N10_ALPHA6_OVERRIDE

    if gate_row.get("selection_class") != "CONDITIONAL":
        return NO_N10_ALPHA6_OVERRIDE

    if gate_row.get("fallback_allowed") is not False:
        return NO_N10_ALPHA6_OVERRIDE

    if (
        gate_row.get(
            "positive_nonobviousness_authority"
        )
        is not False
    ):
        return NO_N10_ALPHA6_OVERRIDE

    if (
        gate_row.get("base_aggregation_action")
        != REFINE_NOVELTY_BEARING_SPECIFICATION
    ):
        return NO_N10_ALPHA6_OVERRIDE

    return Alpha6N10ResolutionDirective(
        force_bounded_refinement=True,
        use_source_external_without_targeted_search=True,
        bypass_resolved_candidate_external_exit=True,
        reason_code=(
            "n10_refine_novelty_bearing_specification"
        ),
    )
