"""Post-generation authority envelope for role-aware N10 v2.

This module does not change candidate selection semantics.

The existing v2 production promoter is frozen for the
``alpha6_original_fallback`` authority scope.  Post-generation Alpha6
candidates require the same promoted role-aware semantics, but they are a
different production decision point.

Accordingly, this wrapper delegates semantic promotion to the frozen v2
promoter and changes only the authority scope after validating the promoted
envelope fail-closed.
"""

from copy import deepcopy
from typing import Any

from pipeline_core.discovery.nonobviousness_production_gate_v2 import (
    build_nonobviousness_production_gate_v2,
)


ORIGINAL_ALPHA6_AUTHORITY_SCOPE = "alpha6_original_fallback"

POST_GENERATION_ALPHA6_AUTHORITY_SCOPE = (
    "alpha6_post_generation_candidate"
)

ROLE_AWARE_V2_SCHEMA = (
    "scientific-novelty-fallback-gate-v2"
)

ROLE_AWARE_V2_AUTHORITY_SOURCE = (
    "n10_role_aware_nonobviousness_v2"
)

ROLE_AWARE_POSITIVE_REQUIREMENT = (
    "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS"
)


def _validate_original_v2_envelope(
    promoted: dict[str, Any],
) -> None:
    """Validate the frozen v2 promoter result before scope rebinding."""

    if (
        promoted.get("schema_version")
        != ROLE_AWARE_V2_SCHEMA
    ):
        raise ValueError(
            "unexpected role-aware v2 production schema"
        )

    if promoted.get("production_authority") is not True:
        raise ValueError(
            "role-aware v2 gate lacks production authority"
        )

    if (
        promoted.get("authority_scope")
        != ORIGINAL_ALPHA6_AUTHORITY_SCOPE
    ):
        raise ValueError(
            "unexpected source authority scope for "
            "post-generation rebinding"
        )

    if (
        promoted.get("authority_source")
        != ROLE_AWARE_V2_AUTHORITY_SOURCE
    ):
        raise ValueError(
            "unexpected role-aware v2 authority source"
        )

    if (
        promoted.get("positive_authority_requires")
        != ROLE_AWARE_POSITIVE_REQUIREMENT
    ):
        raise ValueError(
            "unexpected role-aware positive-authority contract"
        )

    if promoted.get("conditional_is_positive") is not False:
        raise ValueError(
            "CONDITIONAL must not be positive authority"
        )

    if promoted.get("absence_is_novelty") is not False:
        raise ValueError(
            "search-bounded absence must not become novelty"
        )

    if (
        promoted.get("candidate_semantics_preserved")
        is not True
    ):
        raise ValueError(
            "candidate semantics must be preserved"
        )

    if not isinstance(
        promoted.get("gates"),
        list,
    ):
        raise ValueError(
            "role-aware v2 gates must be a list"
        )


def build_nonobviousness_post_generation_production_gate_v2(
    *,
    candidate_gate: dict[str, Any],
) -> dict[str, Any]:
    """Promote a role-aware candidate gate for post-generation authority.

    Selection classes, actions, positive-authority booleans, fallback
    booleans, resolution diagnostics, and all source provenance are inherited
    unchanged from the frozen v2 production promoter.

    Only the authority scope changes.
    """

    promoted = build_nonobviousness_production_gate_v2(
        candidate_gate=candidate_gate,
    )

    _validate_original_v2_envelope(
        promoted,
    )

    result = deepcopy(promoted)

    result["authority_scope"] = (
        POST_GENERATION_ALPHA6_AUTHORITY_SCOPE
    )

    return result
