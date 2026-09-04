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
from pipeline_core.discovery.nonobviousness_production_gate import (
    build_nonobviousness_fallback_gate,
)


_SCHEMA = "nonobviousness-dual-run-comparison-v1"


def build_nonobviousness_dual_run_comparison(
    *,
    query_plan: LiteratureQueryPlan,
    intake_shadow: dict[str, Any],
    full_shadow: dict[str, Any],
) -> dict[str, Any]:
    """Compare authoritative v1 and candidate v2 without promotion.

    This compiler is observational only.

    Production authority remains with the frozen v1 gate. The v2
    result is computed solely to expose semantic divergence before
    any production integration decision is made.
    """

    source_portfolio_id = str(
        intake_shadow.get(
            "source_portfolio_id"
        )
        or ""
    )

    if (
        query_plan.source_portfolio_id
        != source_portfolio_id
    ):
        raise ValueError(
            "dual-run query-plan/intake "
            "portfolio mismatch"
        )

    if (
        full_shadow.get(
            "source_portfolio_id"
        )
        != source_portfolio_id
    ):
        raise ValueError(
            "dual-run intake/full "
            "portfolio mismatch"
        )

    if (
        intake_shadow.get(
            "source_query_plan_id"
        )
        != query_plan.plan_id
    ):
        raise ValueError(
            "dual-run query-plan/intake "
            "plan mismatch"
        )

    v1 = build_nonobviousness_fallback_gate(
        intake_shadow=intake_shadow,
        full_shadow=full_shadow,
    )

    atomic = build_atomic_outcomes_from_n9(
        intake_shadow=intake_shadow,
        full_shadow=full_shadow,
    )

    v2 = build_hypothesis_selection_shadow_v2(
        query_plan=query_plan,
        atomic_outcomes=atomic,
    )

    v1_by_id = {
        row["hypothesis_id"]: row
        for row in v1.get(
            "gates",
            []
        )
    }

    v2_by_id = {
        row["hypothesis_id"]: row
        for row in v2.get(
            "hypotheses",
            []
        )
    }

    if set(v1_by_id) != set(v2_by_id):
        raise ValueError(
            "dual-run v1/v2 hypothesis set mismatch"
        )

    rows: list[
        dict[str, Any]
    ] = []

    transition_counts: dict[
        str,
        int,
    ] = {}

    positive_authority_divergence_count = 0

    for hypothesis_id in sorted(
        v1_by_id
    ):
        old = v1_by_id[hypothesis_id]
        new = v2_by_id[hypothesis_id]

        old_selection = str(
            old.get("selection_class")
            or ""
        )

        new_selection = str(
            new.get("selection_class")
            or ""
        )

        transition = (
            old_selection
            + " -> "
            + new_selection
        )

        transition_counts[
            transition
        ] = (
            transition_counts.get(
                transition,
                0,
            )
            + 1
        )

        v1_positive = bool(
            old.get(
                "fallback_allowed"
            )
        )

        v2_positive = bool(
            new.get(
                "shadow_positive_nonobviousness_authority"
            )
        )

        positive_changed = (
            v1_positive
            != v2_positive
        )

        if positive_changed:
            positive_authority_divergence_count += 1

        rows.append(
            {
                "hypothesis_id":
                    hypothesis_id,

                "v1": {
                    "selection_class":
                        old_selection,
                    "fallback_allowed":
                        v1_positive,
                    "action":
                        old.get("action"),
                    "reason_codes":
                        list(
                            old.get(
                                "reason_codes",
                                []
                            )
                        ),
                    "selection_relevant_claim_ids":
                        list(
                            old.get(
                                "selection_relevant_claim_ids",
                                [],
                            )
                        ),
                },

                "v2_candidate": {
                    "selection_class":
                        new_selection,
                    "shadow_positive_nonobviousness_authority":
                        v2_positive,
                    "action":
                        new.get("action"),
                    "base_aggregation_action":
                        new.get(
                            "base_aggregation_action"
                        ),
                    "blocking_claim_ids":
                        list(
                            new.get(
                                "blocking_claim_ids",
                                [],
                            )
                        ),
                    "unresolved_claim_ids":
                        list(
                            new.get(
                                "unresolved_claim_ids",
                                [],
                            )
                        ),
                    "resolution_requirements":
                        list(
                            new.get(
                                "resolution_requirements",
                                [],
                            )
                        ),
                },

                "selection_transition":
                    transition,

                "selection_changed":
                    old_selection
                    != new_selection,

                "positive_authority_candidate_changed":
                    positive_changed,

                # The comparison artifact itself cannot alter runtime
                # authority. This is deliberately the observed v1
                # decision, not a merged or promoted decision.
                "observed_production_fallback_allowed":
                    v1_positive,

                "candidate_has_production_authority":
                    False,
            }
        )

    return {
        "schema_version":
            _SCHEMA,

        "comparison_only":
            True,

        "production_authority":
            False,

        "authoritative_policy":
            "scientific-novelty-fallback-gate-v1",

        "candidate_policy":
            "hypothesis-selection-shadow-v2",

        "candidate_has_production_authority":
            False,

        "authority_policy":
            "v1_only",

        "source_portfolio_id":
            source_portfolio_id,

        "source_query_plan_id":
            query_plan.plan_id,

        "hypothesis_count":
            len(rows),

        "selection_transition_counts":
            dict(
                sorted(
                    transition_counts.items()
                )
            ),

        "positive_authority_divergence_count":
            positive_authority_divergence_count,

        "comparisons":
            rows,
    }
