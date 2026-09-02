from __future__ import annotations

import hashlib
from typing import Any

from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)


_SCHEMA = "nonobviousness-opportunity-shadow-v1"

_CANDIDATE_LIKE_EXTERNAL = {
    "NEW_COMBINATION_OF_KNOWN_EFFECTS",
    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
    "PLAUSIBLY_NOVEL",
}

_REPEAT_CONDITION_KINDS = {
    "moderator_interaction",
    "context_condition",
    "descriptor_interaction",
}


def _stable_id(
    prefix: str,
    *parts: object,
) -> str:
    raw = "|".join(
        str(x)
        for x in parts
    ).encode("utf-8")

    return (
        prefix
        + ":"
        + hashlib.sha256(raw).hexdigest()[:20]
    )


def _sorted_unique(
    values,
) -> list[str]:
    return sorted(
        {
            str(x)
            for x in values
            if str(x).strip()
        }
    )


def _positive_mechanism_statement_ids(
    dual: DualHypothesisContext,
) -> set[str]:
    """Positive mechanism evidence usable by N11 opportunity planning.

    Unresolved statements, navigation notes and verification-required
    candidate material cannot become positive support merely because
    they occur in a mechanism route.
    """

    return {
        str(row.statement_id)
        for row
        in dual.grounded_context.evidence_statements
        if (
            row.eligible_as_premise
            and not row.requires_verification
            and row.epistemic_role
            in {
                "reported",
                "evidence_synthesis",
            }
            and row.claim_kind
            == "mechanism"
        )
    }


def _raw_mechanistic_branches(
    dual: DualHypothesisContext,
) -> list[dict[str, Any]]:
    positive_ids = (
        _positive_mechanism_statement_ids(
            dual
        )
    )

    branches: list[
        dict[str, Any]
    ] = []

    for route in (
        dual.grounded_context.mechanism_routes
    ):
        if route.requires_verification:
            continue

        support = _sorted_unique(
            set(
                map(
                    str,
                    route.statement_ids,
                )
            )
            & positive_ids
        )

        if not support:
            continue

        branches.append(
            {
                "source_refs": [
                    "route:"
                    + str(route.route_id)
                ],
                "support_statement_ids":
                    support,
                "paper_ids":
                    _sorted_unique(
                        route.paper_ids
                    ),
            }
        )

    for motif in (
        dual.grounded_context.mechanistic_motifs
    ):
        support = _sorted_unique(
            set(
                map(
                    str,
                    motif.statement_ids,
                )
            )
            & positive_ids
        )

        if not support:
            continue

        branches.append(
            {
                "source_refs": [
                    "motif:"
                    + str(motif.motif_id)
                ],
                "support_statement_ids":
                    support,
                "paper_ids":
                    _sorted_unique(
                        motif.paper_ids
                    ),
            }
        )

    return branches


def _collapse_overlapping_branches(
    branches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse wrappers that share positive mechanism evidence.

    v1.1 deliberately uses a conservative definition of mechanistic
    independence: two branches are treated as distinct only when their
    positive mechanism statement sets do not overlap.

    A route and motif over the same scientific statements therefore
    cannot masquerade as two pathways.
    """

    groups: list[
        dict[str, set[str]]
    ] = []

    for branch in branches:
        support = set(
            map(
                str,
                branch[
                    "support_statement_ids"
                ],
            )
        )

        if not support:
            continue

        refs = set(
            map(
                str,
                branch.get(
                    "source_refs",
                    [],
                ),
            )
        )

        papers = set(
            map(
                str,
                branch.get(
                    "paper_ids",
                    [],
                ),
            )
        )

        touching = [
            index
            for index, group
            in enumerate(groups)
            if (
                support
                & group[
                    "support_statement_ids"
                ]
            )
        ]

        if not touching:
            groups.append(
                {
                    "support_statement_ids":
                        set(support),
                    "source_refs":
                        set(refs),
                    "paper_ids":
                        set(papers),
                }
            )
            continue

        first = touching[0]

        groups[first][
            "support_statement_ids"
        ].update(support)

        groups[first][
            "source_refs"
        ].update(refs)

        groups[first][
            "paper_ids"
        ].update(papers)

        # Overlap can be transitive. Merge every other touching group
        # into the first one.
        for index in reversed(
            touching[1:]
        ):
            groups[first][
                "support_statement_ids"
            ].update(
                groups[index][
                    "support_statement_ids"
                ]
            )
            groups[first][
                "source_refs"
            ].update(
                groups[index][
                    "source_refs"
                ]
            )
            groups[first][
                "paper_ids"
            ].update(
                groups[index][
                    "paper_ids"
                ]
            )
            del groups[index]

    result: list[
        dict[str, Any]
    ] = []

    for group in groups:
        support = sorted(
            group[
                "support_statement_ids"
            ]
        )

        refs = sorted(
            group[
                "source_refs"
            ]
        )

        papers = sorted(
            group[
                "paper_ids"
            ]
        )

        result.append(
            {
                "branch_id":
                    _stable_id(
                        "n11_mechanism_branch",
                        ",".join(support),
                    ),
                "source_refs":
                    refs,
                "support_statement_ids":
                    support,
                "paper_ids":
                    papers,
            }
        )

    return sorted(
        result,
        key=lambda row: (
            row["branch_id"]
        ),
    )


def _project_branches_to_hypothesis(
    *,
    branches: list[dict[str, Any]],
    premise_statement_ids: list[str],
) -> list[dict[str, Any]]:
    """Keep only mechanism evidence actually available to this hypothesis."""

    premise_ids = set(
        map(
            str,
            premise_statement_ids,
        )
    )

    projected: list[
        dict[str, Any]
    ] = []

    for branch in branches:
        support = _sorted_unique(
            set(
                map(
                    str,
                    branch[
                        "support_statement_ids"
                    ],
                )
            )
            & premise_ids
        )

        if not support:
            continue

        projected.append(
            {
                "source_refs":
                    list(
                        branch.get(
                            "source_refs",
                            [],
                        )
                    ),
                "support_statement_ids":
                    support,
                "paper_ids":
                    list(
                        branch.get(
                            "paper_ids",
                            [],
                        )
                    ),
            }
        )

    return (
        _collapse_overlapping_branches(
            projected
        )
    )


def _hypothesis_bound_gap_ids(
    *,
    dual: DualHypothesisContext,
    gap_statement_ids: list[str],
) -> list[str]:
    eligible_gap_ids = {
        str(row.statement_id)
        for row
        in dual.grounded_context.evidence_statements
        if (
            row.epistemic_role
            == "unresolved"
            and row.eligible_as_gap
        )
    }

    return _sorted_unique(
        set(
            map(
                str,
                gap_statement_ids,
            )
        )
        & eligible_gap_ids
    )


def _design_lever_support(
    *,
    dual: DualHypothesisContext,
    premise_statement_ids: list[str],
) -> list[str]:
    premise_ids = set(
        map(
            str,
            premise_statement_ids,
        )
    )

    positive_ids = {
        str(row.statement_id)
        for row
        in dual.grounded_context.evidence_statements
        if (
            row.eligible_as_premise
            and not row.requires_verification
        )
    }

    support: set[str] = set()

    for lever in (
        dual.grounded_context
        .reported_design_levers
    ):
        support.update(
            set(
                map(
                    str,
                    lever.statement_ids,
                )
            )
            & premise_ids
            & positive_ids
        )

    return sorted(
        support
    )


def _higher_order_operator_rows(
    *,
    mechanistic_branches:
        list[dict[str, Any]],
    bound_gap_statement_ids:
        list[str],
    design_lever_statement_ids:
        list[str],
) -> tuple[
    list[dict[str, Any]],
    list[str],
]:
    """Compile conservative relation-search eligibility.

    Positive support and unresolved gap IDs stay explicitly separate.
    The gap indicates where inference may be explored; it is never
    promoted into positive evidence.
    """

    eligible: list[
        dict[str, Any]
    ] = []

    unsupported: list[str] = []

    branch_support = _sorted_unique(
        sid
        for branch
        in mechanistic_branches
        for sid
        in branch[
            "support_statement_ids"
        ]
    )

    has_distinct_mechanisms = (
        len(
            mechanistic_branches
        )
        >= 2
    )

    has_bound_gap = bool(
        bound_gap_statement_ids
    )

    if (
        has_distinct_mechanisms
        and has_bound_gap
    ):
        eligible.append(
            {
                "operator":
                    "PATHWAY_COMPETITION",
                "support_statement_ids":
                    branch_support,
                "gap_statement_ids":
                    list(
                        bound_gap_statement_ids
                    ),
                "reason_codes": [
                    "multiple_distinct_grounded_mechanism_branches",
                    "hypothesis_bound_unresolved_gap",
                    "positive_support_separate_from_unresolved_gap",
                ],
            }
        )
    else:
        unsupported.append(
            "PATHWAY_COMPETITION"
        )

    if (
        has_distinct_mechanisms
        and has_bound_gap
        and bool(
            design_lever_statement_ids
        )
    ):
        eligible.append(
            {
                "operator":
                    "MECHANISM_SWITCH",
                "support_statement_ids":
                    _sorted_unique(
                        [
                            *branch_support,
                            *design_lever_statement_ids,
                        ]
                    ),
                "gap_statement_ids":
                    list(
                        bound_gap_statement_ids
                    ),
                "reason_codes": [
                    "multiple_distinct_grounded_mechanism_branches",
                    "grounded_design_lever_available",
                    "hypothesis_bound_unresolved_gap",
                    "positive_support_separate_from_unresolved_gap",
                ],
            }
        )
    else:
        unsupported.append(
            "MECHANISM_SWITCH"
        )

    # These operators require stronger explicit scientific structure
    # than the current HypothesisContext contract exposes
    # deterministically. Do not infer them from free text.
    unsupported.extend(
        [
            "REGIME_OR_THRESHOLD",
            "REVERSAL_OR_RANKING_CHANGE",
            "NON_MONOTONIC_RESPONSE",
        ]
    )

    return (
        sorted(
            eligible,
            key=lambda row:
                row["operator"],
        ),
        sorted(
            set(
                unsupported
            )
        ),
    )


def _claim_index(
    plan: LiteratureQueryPlan,
) -> dict[str, Any]:
    return {
        str(claim.claim_id): claim
        for group
        in plan.claims
        for claim
        in group.claims
    }


def build_nonobviousness_opportunity_shadow(
    *,
    dual: DualHypothesisContext,
    portfolio: HypothesisPortfolio,
    query_plan: LiteratureQueryPlan,
    external_report: ExternalNoveltyReport,
    full_shadow: dict[str, Any],
    production_gate: dict[str, Any],
) -> dict[str, Any]:
    """Build an N11 shadow-only scientific opportunity artifact.

    N11 does not change scientific selection and does not generate a
    hypothesis. It only identifies relation operators that have enough
    grounded structural prerequisites to justify bounded exploration
    after N10 blocks an original candidate.
    """

    if (
        query_plan.source_portfolio_id
        != portfolio.portfolio_id
    ):
        raise ValueError(
            "N11 query-plan / portfolio provenance mismatch"
        )

    if (
        external_report.source_portfolio_id
        != portfolio.portfolio_id
    ):
        raise ValueError(
            "N11 external-report / portfolio provenance mismatch"
        )

    if (
        full_shadow.get(
            "source_portfolio_id"
        )
        != portfolio.portfolio_id
    ):
        raise ValueError(
            "N11 full-shadow / portfolio provenance mismatch"
        )

    if (
        production_gate.get(
            "source_portfolio_id"
        )
        != portfolio.portfolio_id
    ):
        raise ValueError(
            "N11 production-gate / portfolio provenance mismatch"
        )

    cards = {
        str(card.hypothesis_id): card
        for card
        in portfolio.hypotheses
    }

    external_by_id = {
        str(card.hypothesis_id): card
        for card
        in external_report.cards
    }

    claims = _claim_index(
        query_plan
    )

    raw_branches = (
        _raw_mechanistic_branches(
            dual
        )
    )

    opportunities: list[
        dict[str, Any]
    ] = []

    for gate in production_gate.get(
        "gates",
        [],
    ):
        hypothesis_id = str(
            gate.get(
                "hypothesis_id"
            )
            or ""
        )

        if not hypothesis_id:
            continue

        if (
            gate.get(
                "fallback_allowed"
            )
            is True
        ):
            continue

        card = cards.get(
            hypothesis_id
        )

        external = (
            external_by_id.get(
                hypothesis_id
            )
        )

        if (
            card is None
            or external is None
        ):
            continue

        if (
            external.status
            not in _CANDIDATE_LIKE_EXTERNAL
        ):
            continue

        relevant_claim_ids = set(
            map(
                str,
                gate.get(
                    "selection_relevant_claim_ids",
                    [],
                ),
            )
        )

        blocked_claims: list[
            dict[str, Any]
        ] = []

        for atomic in gate.get(
            "atomic_claims",
            [],
        ):
            claim_id = str(
                atomic.get(
                    "claim_id"
                )
                or ""
            )

            if (
                relevant_claim_ids
                and claim_id
                not in relevant_claim_ids
            ):
                continue

            outcome = str(
                atomic.get(
                    "nonobviousness_outcome"
                )
                or ""
            )

            if (
                outcome
                == "POTENTIALLY_NON_OBVIOUS"
            ):
                continue

            claim = claims.get(
                claim_id
            )

            if claim is None:
                continue

            blocked_claims.append(
                {
                    "claim_id":
                        claim_id,
                    "kind":
                        claim.kind,
                    "importance":
                        claim.importance,
                    "text":
                        claim.text,
                    "n10_outcome":
                        outcome,
                    "n10_reason_codes":
                        list(
                            atomic.get(
                                "reason_codes"
                            )
                            or []
                        ),
                }
            )

        blocked_kinds = {
            str(row["kind"])
            for row
            in blocked_claims
        }

        branches = (
            _project_branches_to_hypothesis(
                branches=raw_branches,
                premise_statement_ids=(
                    card.premise_statement_ids
                ),
            )
        )

        bound_gaps = (
            _hypothesis_bound_gap_ids(
                dual=dual,
                gap_statement_ids=(
                    card.gap_statement_ids
                ),
            )
        )

        lever_support = (
            _design_lever_support(
                dual=dual,
                premise_statement_ids=(
                    card.premise_statement_ids
                ),
            )
        )

        (
            higher_order,
            unsupported,
        ) = _higher_order_operator_rows(
            mechanistic_branches=branches,
            bound_gap_statement_ids=(
                bound_gaps
            ),
            design_lever_statement_ids=(
                lever_support
            ),
        )

        eligible = list(
            higher_order
        )

        deprioritized: list[
            dict[str, Any]
        ] = []

        if lever_support:
            if (
                blocked_kinds
                & _REPEAT_CONDITION_KINDS
            ):
                deprioritized.append(
                    {
                        "operator":
                            "CONDITIONED_DEPENDENCY",
                        "support_statement_ids":
                            lever_support,
                        "gap_statement_ids":
                            bound_gaps,
                        "reason_codes": [
                            "grounded_design_lever_available",
                            "same_relation_operator_already_blocked_by_n10",
                        ],
                    }
                )
            elif bound_gaps:
                eligible.append(
                    {
                        "operator":
                            "CONDITIONED_DEPENDENCY",
                        "support_statement_ids":
                            lever_support,
                        "gap_statement_ids":
                            bound_gaps,
                        "reason_codes": [
                            "grounded_design_lever_available",
                            "hypothesis_bound_unresolved_gap",
                        ],
                    }
                )
            else:
                unsupported.append(
                    "CONDITIONED_DEPENDENCY"
                )
        else:
            unsupported.append(
                "CONDITIONED_DEPENDENCY"
            )

        opportunities.append(
            {
                "hypothesis_id":
                    hypothesis_id,
                "source_external_status":
                    external.status,
                "n10_gate_action":
                    gate.get(
                        "action"
                    ),
                "n10_selection_class":
                    gate.get(
                        "selection_class"
                    ),
                "blocked_claims":
                    blocked_claims,
                "bound_gap_statement_ids":
                    bound_gaps,
                "mechanistic_branch_count":
                    len(
                        branches
                    ),
                "mechanistic_branches":
                    branches,
                "design_lever_statement_ids":
                    lever_support,
                "eligible_operators":
                    sorted(
                        eligible,
                        key=lambda row:
                            row["operator"],
                    ),
                "deprioritized_repeat_operators":
                    sorted(
                        deprioritized,
                        key=lambda row:
                            row["operator"],
                    ),
                "unsupported_operators":
                    sorted(
                        set(
                            unsupported
                        )
                    ),
            }
        )

    return {
        "schema_version":
            _SCHEMA,
        "shadow_only":
            True,
        "scientific_selection_changed":
            False,
        "source_portfolio_id":
            portfolio.portfolio_id,
        "source_dual_context_id":
            dual.dual_context_id,
        "source_query_plan_id":
            query_plan.plan_id,
        "source_external_report_id":
            external_report.report_id,
        "source_nonobviousness_full_schema":
            full_shadow.get(
                "schema_version"
            ),
        "source_production_gate_schema":
            production_gate.get(
                "schema_version"
            ),
        "opportunity_count":
            len(
                opportunities
            ),
        "opportunities":
            opportunities,
        "epistemic_policy": {
            "shadow_only":
                True,
            "n10_remains_production_authority":
                True,
            "unresolved_gap_is_not_positive_evidence":
                True,
            "operator_requires_grounded_structural_prerequisites":
                True,
            "route_motif_overlap_does_not_create_distinct_mechanisms":
                True,
            "mechanistic_branches_are_hypothesis_premise_conditioned":
                True,
        },
    }
