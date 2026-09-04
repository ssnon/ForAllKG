from __future__ import annotations

from copy import deepcopy
from typing import Any


_CANDIDATE_SCHEMA = (
    "scientific-novelty-fallback-"
    "gate-v2-candidate"
)

_PRODUCTION_SCHEMA = (
    "scientific-novelty-fallback-"
    "gate-v2"
)

_ALLOWED_SELECTION_CLASSES = {
    "ELIGIBLE",
    "CONDITIONAL",
    "INELIGIBLE",
}

_AUTHORITY_SOURCE = (
    "n10_role_aware_nonobviousness_v2"
)


def build_nonobviousness_production_gate_v2(
    *,
    candidate_gate: dict[str, Any],
) -> dict[str, Any]:
    """Promote frozen role-aware candidate semantics to v2 authority.

    This function does NOT perform scientific re-aggregation.

    It accepts only a valid candidate-only N10-v2 artifact and copies
    its already-frozen fallback decision into the authoritative field.

    Scientific policy:
    - ELIGIBLE requires explicit positive role-aware authority.
    - CONDITIONAL is never positive authority.
    - INELIGIBLE is never positive authority.
    - Absence/no-match cannot create positive authority.
    - Any malformed or internally inconsistent candidate fails closed
      by refusing compilation.

    Runtime consumption is intentionally outside this compiler.
    """

    if not isinstance(
        candidate_gate,
        dict,
    ):
        raise ValueError(
            "v2 production compiler requires "
            "candidate gate object"
        )

    if (
        candidate_gate.get(
            "schema_version"
        )
        != _CANDIDATE_SCHEMA
    ):
        raise ValueError(
            "unexpected v2 candidate schema"
        )

    if (
        candidate_gate.get(
            "candidate_only"
        )
        is not True
    ):
        raise ValueError(
            "v2 production compiler requires "
            "candidate-only source"
        )

    if (
        candidate_gate.get(
            "production_authority"
        )
        is not False
    ):
        raise ValueError(
            "v2 candidate unexpectedly already "
            "has production authority"
        )

    if (
        candidate_gate.get(
            "alpha6_original_fallback_authority"
        )
        is not False
    ):
        raise ValueError(
            "v2 candidate unexpectedly already "
            "has Alpha6 authority"
        )

    if (
        candidate_gate.get(
            "authority_policy"
        )
        != "none_candidate_only"
    ):
        raise ValueError(
            "unexpected v2 candidate authority policy"
        )

    gates = candidate_gate.get(
        "gates"
    )

    if not isinstance(
        gates,
        list,
    ):
        raise ValueError(
            "v2 candidate gates must be a list"
        )

    declared_gate_count = (
        candidate_gate.get(
            "gate_count"
        )
    )

    if (
        not isinstance(
            declared_gate_count,
            int,
        )
        or declared_gate_count
        != len(gates)
    ):
        raise ValueError(
            "v2 candidate gate_count mismatch"
        )

    seen: set[str] = set()

    production_rows: list[
        dict[str, Any]
    ] = []

    allowed_count = 0

    selection_counts = {
        "ELIGIBLE": 0,
        "CONDITIONAL": 0,
        "INELIGIBLE": 0,
    }

    for row in gates:
        if not isinstance(
            row,
            dict,
        ):
            raise ValueError(
                "v2 candidate gate row "
                "must be an object"
            )

        hypothesis_id = str(
            row.get(
                "hypothesis_id"
            )
            or ""
        ).strip()

        if not hypothesis_id:
            raise ValueError(
                "v2 candidate gate row "
                "missing hypothesis_id"
            )

        if hypothesis_id in seen:
            raise ValueError(
                "duplicate v2 candidate "
                "hypothesis_id: "
                + hypothesis_id
            )

        seen.add(
            hypothesis_id
        )

        selection_class = str(
            row.get(
                "selection_class"
            )
            or ""
        ).strip()

        if (
            selection_class
            not in _ALLOWED_SELECTION_CLASSES
        ):
            raise ValueError(
                "unsupported v2 selection class: "
                f"{selection_class!r}"
            )

        positive = row.get(
            "candidate_positive_nonobviousness_authority"
        )

        if not isinstance(
            positive,
            bool,
        ):
            raise ValueError(
                "v2 candidate positive authority "
                "must be boolean"
            )

        candidate_allowed = row.get(
            "candidate_fallback_allowed"
        )

        if not isinstance(
            candidate_allowed,
            bool,
        ):
            raise ValueError(
                "v2 candidate fallback permission "
                "must be boolean"
            )

        if (
            row.get(
                "production_authority"
            )
            is not False
        ):
            raise ValueError(
                "v2 candidate row unexpectedly "
                "has production authority"
            )

        # Strong promotion invariant:
        # an ELIGIBLE label without positive scientific authority
        # is corrupt rather than merely fallback-negative.
        if (
            selection_class
            == "ELIGIBLE"
            and positive
            is not True
        ):
            raise ValueError(
                "ELIGIBLE v2 candidate lacks "
                "positive nonobviousness authority"
            )

        # Positive authority cannot exist on a non-eligible state.
        if (
            selection_class
            != "ELIGIBLE"
            and positive
            is True
        ):
            raise ValueError(
                "non-ELIGIBLE v2 candidate "
                "claims positive authority"
            )

        expected_candidate_allowed = bool(
            selection_class
            == "ELIGIBLE"
            and positive
            is True
        )

        if (
            candidate_allowed
            is not expected_candidate_allowed
        ):
            raise ValueError(
                "v2 candidate fallback decision "
                "is internally inconsistent"
            )

        # No semantic reinterpretation occurs here.
        fallback_allowed = (
            candidate_allowed
        )

        if fallback_allowed:
            allowed_count += 1

        selection_counts[
            selection_class
        ] += 1

        production_rows.append(
            {
                "hypothesis_id":
                    hypothesis_id,

                "selection_class":
                    selection_class,

                "fallback_allowed":
                    fallback_allowed,

                "positive_nonobviousness_authority":
                    positive,

                "authority_source":
                    _AUTHORITY_SOURCE,

                "action":
                    row.get(
                        "action"
                    ),

                "base_aggregation_action":
                    row.get(
                        "base_aggregation_action"
                    ),

                "blocking_claim_ids":
                    deepcopy(
                        row.get(
                            "blocking_claim_ids",
                            [],
                        )
                    ),

                "unresolved_claim_ids":
                    deepcopy(
                        row.get(
                            "unresolved_claim_ids",
                            [],
                        )
                    ),

                "unresolved_selection_role_claim_ids":
                    deepcopy(
                        row.get(
                            "unresolved_selection_role_claim_ids",
                            [],
                        )
                    ),

                "structurally_unresolved_claim_ids":
                    deepcopy(
                        row.get(
                            "structurally_unresolved_claim_ids",
                            [],
                        )
                    ),

                "resolution_requirements":
                    deepcopy(
                        row.get(
                            "resolution_requirements",
                            [],
                        )
                    ),

                "reason_codes":
                    deepcopy(
                        row.get(
                            "reason_codes",
                            [],
                        )
                    ),
            }
        )

    declared_allowed = (
        candidate_gate.get(
            "candidate_fallback_allowed_count"
        )
    )

    if (
        not isinstance(
            declared_allowed,
            int,
        )
        or declared_allowed
        != allowed_count
    ):
        raise ValueError(
            "v2 candidate allowed-count mismatch"
        )

    declared_blocked = (
        candidate_gate.get(
            "candidate_fallback_blocked_count"
        )
    )

    blocked_count = (
        len(gates)
        - allowed_count
    )

    if (
        not isinstance(
            declared_blocked,
            int,
        )
        or declared_blocked
        != blocked_count
    ):
        raise ValueError(
            "v2 candidate blocked-count mismatch"
        )

    declared_selection_counts = (
        candidate_gate.get(
            "selection_counts"
        )
    )

    if (
        not isinstance(
            declared_selection_counts,
            dict,
        )
        or {
            key:
                int(
                    declared_selection_counts.get(
                        key,
                        0,
                    )
                )
            for key
            in selection_counts
        }
        != selection_counts
    ):
        raise ValueError(
            "v2 candidate selection-count mismatch"
        )

    return {
        "schema_version":
            _PRODUCTION_SCHEMA,

        "production_authority":
            True,

        "authority_scope":
            "alpha6_original_fallback",

        "authority_source":
            _AUTHORITY_SOURCE,

        "source_candidate_schema":
            _CANDIDATE_SCHEMA,

        "source_portfolio_id":
            candidate_gate.get(
                "source_portfolio_id"
            ),

        "source_query_plan_id":
            candidate_gate.get(
                "source_query_plan_id"
            ),

        "gate_count":
            len(production_rows),

        "fallback_allowed_count":
            allowed_count,

        "fallback_blocked_count":
            blocked_count,

        "selection_counts":
            selection_counts,

        "positive_authority_requires":
            (
                "ELIGIBLE_AND_ROLE_AWARE_"
                "POSITIVE_NONOBVIOUSNESS"
            ),

        "conditional_is_positive":
            False,

        "absence_is_novelty":
            False,

        "candidate_semantics_preserved":
            True,

        "gates":
            production_rows,
    }
