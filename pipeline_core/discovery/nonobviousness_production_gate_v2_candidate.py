from __future__ import annotations

from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_selection_shadow_v2 import (
    build_hypothesis_selection_shadow_v2,
)
from pipeline_core.discovery.nonobviousness_outcome_adapter import (
    build_atomic_outcomes_from_n9,
)


_SCHEMA = (
    "scientific-novelty-fallback-"
    "gate-v2-candidate"
)

_ALLOWED_SELECTION_CLASSES = {
    "ELIGIBLE",
    "CONDITIONAL",
    "INELIGIBLE",
}


def build_nonobviousness_production_gate_v2_candidate(
    *,
    query_plan: LiteratureQueryPlan,
    intake_shadow: dict[str, Any],
    full_shadow: dict[str, Any],
) -> dict[str, Any]:
    """Compile role-aware v2 semantics into candidate fallback booleans.

    This artifact intentionally has NO production authority.

    Candidate fallback permission is granted only when the frozen
    shadow-v2 scientific-selection result is ELIGIBLE and that result
    independently carries positive non-obviousness authority.

    CONDITIONAL is never positive.
    INELIGIBLE is never positive.
    Absence is never novelty.
    """

    source_portfolio_id = (
        query_plan.source_portfolio_id
    )

    if (
        intake_shadow.get(
            "source_portfolio_id"
        )
        != source_portfolio_id
    ):
        raise ValueError(
            "v2 candidate query-plan/intake "
            "portfolio mismatch"
        )

    if (
        full_shadow.get(
            "source_portfolio_id"
        )
        != source_portfolio_id
    ):
        raise ValueError(
            "v2 candidate query-plan/full "
            "portfolio mismatch"
        )

    if (
        intake_shadow.get(
            "source_query_plan_id"
        )
        != query_plan.plan_id
    ):
        raise ValueError(
            "v2 candidate query-plan/intake "
            "plan mismatch"
        )

    atomic_outcomes = (
        build_atomic_outcomes_from_n9(
            intake_shadow=intake_shadow,
            full_shadow=full_shadow,
        )
    )

    selection = (
        build_hypothesis_selection_shadow_v2(
            query_plan=query_plan,
            atomic_outcomes=atomic_outcomes,
        )
    )

    if (
        selection.get(
            "production_authority"
        )
        is not False
    ):
        raise RuntimeError(
            "shadow-v2 unexpectedly has "
            "production authority"
        )

    if (
        selection.get(
            "alpha6_original_fallback_authority"
        )
        is not False
    ):
        raise RuntimeError(
            "shadow-v2 unexpectedly has "
            "Alpha6 fallback authority"
        )

    gates: list[
        dict[str, Any]
    ] = []

    candidate_allowed_count = 0

    for row in selection.get(
        "hypotheses",
        [],
    ):
        hypothesis_id = str(
            row.get(
                "hypothesis_id"
            )
            or ""
        ).strip()

        if not hypothesis_id:
            raise ValueError(
                "v2 candidate hypothesis_id empty"
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
                "unsupported v2 candidate "
                "selection class: "
                f"{selection_class!r}"
            )

        shadow_positive = bool(
            row.get(
                "shadow_positive_nonobviousness_authority"
            )
        )

        # Fail closed on an internally inconsistent positive flag.
        if (
            shadow_positive
            and selection_class
            != "ELIGIBLE"
        ):
            raise RuntimeError(
                "non-ELIGIBLE hypothesis "
                "unexpectedly has positive "
                "shadow authority"
            )

        candidate_allowed = bool(
            selection_class
            == "ELIGIBLE"
            and shadow_positive
        )

        if candidate_allowed:
            candidate_allowed_count += 1

        if candidate_allowed:
            reason_codes = [
                "role_aware_v2_eligible_and_positive",
            ]

        elif selection_class == "CONDITIONAL":
            reason_codes = [
                "role_aware_v2_conditional_fail_closed",
            ]

        elif selection_class == "INELIGIBLE":
            reason_codes = [
                "role_aware_v2_ineligible",
            ]

        else:
            # Defensive only; enum validation above should make
            # this unreachable.
            reason_codes = [
                "role_aware_v2_unknown_fail_closed",
            ]

        gates.append(
            {
                "hypothesis_id":
                    hypothesis_id,

                "selection_class":
                    selection_class,

                "candidate_fallback_allowed":
                    candidate_allowed,

                "candidate_positive_nonobviousness_authority":
                    shadow_positive,

                "action":
                    row.get(
                        "action"
                    ),

                "base_aggregation_action":
                    row.get(
                        "base_aggregation_action"
                    ),

                "blocking_claim_ids":
                    list(
                        row.get(
                            "blocking_claim_ids",
                            [],
                        )
                    ),

                "unresolved_claim_ids":
                    list(
                        row.get(
                            "unresolved_claim_ids",
                            [],
                        )
                    ),

                "unresolved_selection_role_claim_ids":
                    list(
                        row.get(
                            "unresolved_selection_role_claim_ids",
                            [],
                        )
                    ),

                "structurally_unresolved_claim_ids":
                    list(
                        row.get(
                            "structurally_unresolved_claim_ids",
                            [],
                        )
                    ),

                "resolution_requirements":
                    list(
                        row.get(
                            "resolution_requirements",
                            [],
                        )
                    ),

                "reason_codes":
                    reason_codes,

                # This field is deliberately always false until
                # a later explicit promotion commit.
                "production_authority":
                    False,
            }
        )

    return {
        "schema_version":
            _SCHEMA,

        "candidate_only":
            True,

        "production_authority":
            False,

        "alpha6_original_fallback_authority":
            False,

        "authority_policy":
            "none_candidate_only",

        "candidate_policy":
            (
                "role-aware-v2-"
                "eligible-and-positive-only"
            ),

        "source_portfolio_id":
            source_portfolio_id,

        "source_query_plan_id":
            query_plan.plan_id,

        "gate_count":
            len(gates),

        "candidate_fallback_allowed_count":
            candidate_allowed_count,

        "candidate_fallback_blocked_count":
            (
                len(gates)
                - candidate_allowed_count
            ),

        "selection_counts":
            dict(
                selection.get(
                    "selection_counts",
                    {}
                )
            ),

        "policy": {
            "eligible_requires_positive_authority":
                True,

            "conditional_is_positive":
                False,

            "ineligible_is_positive":
                False,

            "absence_is_novelty":
                False,

            "unknown_is_novelty":
                False,

            "topology_overrides_role":
                False,

            "known_required_enabling_is_blocking":
                False,

            "candidate_has_runtime_authority":
                False,
        },

        "gates":
            gates,
    }
