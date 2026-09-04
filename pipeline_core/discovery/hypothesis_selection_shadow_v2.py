from __future__ import annotations

from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    LiteratureQueryPlan,
)
from pipeline_core.discovery.novelty_selection_aggregation import (
    NonobviousnessOutcome,
)
from pipeline_core.discovery.novelty_selection_topology_aggregation import (
    TopologyAwareAtomicClaim,
    aggregate_topology_aware_nonobviousness,
)
from pipeline_core.discovery.nonobviousness_shadow_action_routing import (
    route_shadow_resolution_actions,
)


_SCHEMA = "hypothesis-selection-shadow-v2"


_ALLOWED_OUTCOMES = {
    "POTENTIALLY_NON_OBVIOUS",
    "SATURATED_PRIOR_ART",
    "ROUTINE_FROM_PRIOR_ART",
    "INSUFFICIENT_FOR_JUDGMENT",
    "NEEDS_REFINEMENT",
}


def _normalize_outcomes(
    atomic_outcomes: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Normalize explicit atomic adjudication input fail closed.

    This compiler does NOT infer outcomes from prior-art status,
    search absence, external novelty status, or claim wording.
    """

    normalized: dict[str, dict[str, Any]] = {}

    for raw_claim_id, raw_value in atomic_outcomes.items():
        claim_id = str(raw_claim_id).strip()

        if not claim_id:
            raise ValueError(
                "atomic outcome requires claim_id"
            )

        if claim_id in normalized:
            raise ValueError(
                "duplicate atomic outcome claim_id: "
                + claim_id
            )

        if isinstance(raw_value, str):
            outcome = raw_value
            reason_codes: list[str] = []
        elif isinstance(raw_value, dict):
            outcome = str(
                raw_value.get(
                    "nonobviousness_outcome"
                )
                or ""
            ).strip()

            reasons = raw_value.get(
                "reason_codes",
                [],
            )

            if not isinstance(
                reasons,
                (list, tuple),
            ):
                raise ValueError(
                    "atomic outcome reason_codes "
                    "must be a list or tuple"
                )

            reason_codes = [
                str(value)
                for value in reasons
                if str(value).strip()
            ]
        else:
            raise ValueError(
                "atomic outcome must be string "
                "or object"
            )

        if outcome not in _ALLOWED_OUTCOMES:
            raise ValueError(
                "unsupported atomic "
                "nonobviousness outcome: "
                + outcome
            )

        normalized[claim_id] = {
            "nonobviousness_outcome":
                outcome,
            "reason_codes":
                list(
                    dict.fromkeys(
                        reason_codes
                    )
                ),
        }

    return normalized


def build_hypothesis_selection_shadow_v2(
    *,
    query_plan: LiteratureQueryPlan,
    atomic_outcomes: dict[str, Any],
) -> dict[str, Any]:
    """Compile role/topology-aware hypothesis selection in shadow only.

    Important:
    - This artifact has NO production authority.
    - It does not set Alpha6 original-fallback permission.
    - It does not infer N9 outcomes.
    - A separate adapter may later translate frozen N9 artifacts into
      the explicit atomic_outcomes input, but that adapter is outside
      this compiler.
    """

    outcomes = _normalize_outcomes(
        atomic_outcomes
    )

    canonical_claim_ids = [
        claim.claim_id
        for group in query_plan.claims
        for claim in group.claims
    ]

    if len(
        canonical_claim_ids
    ) != len(
        set(canonical_claim_ids)
    ):
        raise ValueError(
            "duplicate canonical claim_id "
            "in query plan"
        )

    canonical_set = set(
        canonical_claim_ids
    )

    outcome_set = set(
        outcomes
    )

    unknown_outcomes = sorted(
        outcome_set
        - canonical_set
    )

    if unknown_outcomes:
        raise ValueError(
            "atomic outcomes contain unknown claim_id: "
            + ", ".join(unknown_outcomes)
        )

    missing_outcomes = [
        claim_id
        for claim_id in canonical_claim_ids
        if claim_id not in outcome_set
    ]

    if missing_outcomes:
        raise ValueError(
            "missing atomic nonobviousness outcome: "
            + ", ".join(missing_outcomes)
        )

    hypothesis_rows: list[
        dict[str, Any]
    ] = []

    selection_counts = {
        "ELIGIBLE": 0,
        "CONDITIONAL": 0,
        "INELIGIBLE": 0,
    }

    for group in query_plan.claims:
        topology_claims: list[
            TopologyAwareAtomicClaim
        ] = []

        unresolved_role_claim_ids: list[
            str
        ] = []

        atomic_rows: list[
            dict[str, Any]
        ] = []

        for claim in group.claims:
            outcome_row = outcomes[
                claim.claim_id
            ]

            role = (
                claim.novelty_selection_role
            )

            atomic_rows.append(
                {
                    "claim_id":
                        claim.claim_id,
                    "kind":
                        claim.kind,
                    "importance":
                        claim.importance,
                    "novelty_selection_role":
                        role,
                    "nonobviousness_outcome":
                        outcome_row[
                            "nonobviousness_outcome"
                        ],
                    "outcome_reason_codes":
                        list(
                            outcome_row[
                                "reason_codes"
                            ]
                        ),
                    "higher_order_relation_basis":
                        list(
                            claim
                            .higher_order_relation_basis
                        ),
                    "higher_order_component_claim_ids":
                        list(
                            claim
                            .higher_order_component_claim_ids
                        ),
                }
            )

            if role is None:
                unresolved_role_claim_ids.append(
                    claim.claim_id
                )
                continue

            topology_claims.append(
                TopologyAwareAtomicClaim(
                    claim_id=claim.claim_id,
                    claim_kind=claim.kind,
                    novelty_selection_role=role,
                    nonobviousness_outcome=(
                        outcome_row[
                            "nonobviousness_outcome"
                        ]
                    ),
                    higher_order_relation_basis=tuple(
                        claim
                        .higher_order_relation_basis
                    ),
                    higher_order_component_claim_ids=tuple(
                        claim
                        .higher_order_component_claim_ids
                    ),
                )
            )

        # A missing role is epistemically unresolved. We may still
        # calculate the known-role base state for diagnostics, but it
        # can never yield positive authority.
        if topology_claims:
            aggregate = (
                aggregate_topology_aware_nonobviousness(
                    tuple(topology_claims)
                )
            )

            selection_class = (
                aggregate.selection_class
            )
            action = aggregate.action
            reason_codes = list(
                aggregate.reason_codes
            )

            blocking_claim_ids = list(
                aggregate.blocking_claim_ids
            )

            unresolved_claim_ids = list(
                aggregate.unresolved_claim_ids
            )

            structurally_unresolved_claim_ids = list(
                aggregate
                .structurally_unresolved_claim_ids
            )

            topology_edges = [
                list(edge)
                for edge
                in aggregate.topology_edges
            ]

            novelty_bearing_claim_ids = list(
                aggregate
                .novelty_bearing_claim_ids
            )

            required_enabling_claim_ids = list(
                aggregate
                .required_enabling_claim_ids
            )

            testing_prediction_claim_ids = list(
                aggregate
                .testing_prediction_claim_ids
            )

            auxiliary_claim_ids = list(
                aggregate
                .auxiliary_claim_ids
            )

            nested_novelty_ids = list(
                aggregate
                .nested_novelty_bearing_component_ids
            )
        else:
            selection_class = "INELIGIBLE"
            action = (
                "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION"
            )
            reason_codes = [
                "no_resolved_selection_roles",
            ]
            blocking_claim_ids = []
            unresolved_claim_ids = []
            structurally_unresolved_claim_ids = []
            topology_edges = []
            novelty_bearing_claim_ids = []
            required_enabling_claim_ids = []
            testing_prediction_claim_ids = []
            auxiliary_claim_ids = []
            nested_novelty_ids = []

        if unresolved_role_claim_ids:
            reason_codes.append(
                "unresolved_novelty_selection_roles"
            )

            for claim_id in (
                unresolved_role_claim_ids
            ):
                reason_codes.append(
                    "selection_role_unresolved:"
                    + claim_id
                )

            # Unknown selection role never becomes positive authority.
            # Preserve an already decisive INELIGIBLE result; otherwise
            # downgrade to CONDITIONAL.
            if selection_class != "INELIGIBLE":
                selection_class = (
                    "CONDITIONAL"
                )
                action = (
                    "REFINE_NOVELTY_SELECTION_ROLE_SPECIFICATION"
                )

        shadow_positive = bool(
            selection_class == "ELIGIBLE"
            and not unresolved_role_claim_ids
        )

        routed = route_shadow_resolution_actions(
            selection_class=selection_class,
            atomic_claims=atomic_rows,
            fallback_action=action,
        )

        routed_action = (
            routed.primary_action
        )

        resolution_requirements = [
            {
                "claim_id":
                    row.claim_id,
                "novelty_selection_role":
                    row.novelty_selection_role,
                "nonobviousness_outcome":
                    row.nonobviousness_outcome,
                "action":
                    row.action,
                "reason_codes":
                    list(
                        row.reason_codes
                    ),
            }
            for row
            in routed.resolution_requirements
        ]

        selection_counts[
            selection_class
        ] += 1

        hypothesis_rows.append(
            {
                "hypothesis_id":
                    group.hypothesis_id,
                "selection_class":
                    selection_class,
                "action":
                    routed_action,
                "base_aggregation_action":
                    action,
                "resolution_requirements":
                    resolution_requirements,
                "shadow_positive_nonobviousness_authority":
                    shadow_positive,
                # Critical: v2 is observational only.
                "fallback_allowed":
                    False,
                "production_authority":
                    False,
                "reason_codes":
                    list(
                        dict.fromkeys(
                            reason_codes
                        )
                    ),
                "blocking_claim_ids":
                    blocking_claim_ids,
                "unresolved_claim_ids":
                    unresolved_claim_ids,
                "unresolved_selection_role_claim_ids":
                    list(
                        unresolved_role_claim_ids
                    ),
                "structurally_unresolved_claim_ids":
                    structurally_unresolved_claim_ids,
                "novelty_bearing_claim_ids":
                    novelty_bearing_claim_ids,
                "required_enabling_claim_ids":
                    required_enabling_claim_ids,
                "testing_prediction_claim_ids":
                    testing_prediction_claim_ids,
                "auxiliary_claim_ids":
                    auxiliary_claim_ids,
                "nested_novelty_bearing_component_ids":
                    nested_novelty_ids,
                "topology_edges":
                    topology_edges,
                "atomic_claims":
                    atomic_rows,
            }
        )

    return {
        "schema_version":
            _SCHEMA,
        "shadow_only": True,
        "production_authority": False,
        "authority_scope":
            "observational_hypothesis_selection_only",
        "alpha6_original_fallback_authority":
            False,
        "source_portfolio_id":
            query_plan.source_portfolio_id,
        "source_query_plan_id":
            query_plan.plan_id,
        "source_query_plan_sha256":
            query_plan.plan_sha256,
        "policy": {
            "absence_is_not_novelty": True,
            "unknown_is_not_novelty": True,
            "topology_does_not_override_role": True,
            "nested_novelty_bearing_components_remain_selection_relevant":
                True,
            "known_required_enabling_relation_is_nonblocking":
                True,
            "composite_positive_authority_requires_explicit_source_basis":
                True,
            "conditional_is_not_positive_authority":
                True,
            "fallback_allowed_always_false_in_shadow_v2":
                True,
        },
        "hypothesis_count":
            len(hypothesis_rows),
        "selection_counts":
            selection_counts,
        "hypotheses":
            hypothesis_rows,
    }
