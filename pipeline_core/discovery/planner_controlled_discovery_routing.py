from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ControlledRuntimeRoute = Literal[
    "KEEP_ORIGINAL",
    "GAP_SHARPEN",
    "FRESH_CONTEXT_REAXIS",
    "HOLD_FOR_EVIDENCE",
]

ControlledRoutingExecutionStatus = Literal[
    "MATCH",
    "DIVERGED",
    "NOT_OBSERVED",
    "DISABLED",
]


_ALLOWED_OPERATORS = {
    "MODERATOR",
    "INTERACTION",
    "RESIDUAL",
    "BOUNDARY",
    "PROXY_DECOUPLING",
    "COMPENSATION_LIMIT",
}


class PlannerControlledRoutingDirective(StrictModel):
    hypothesis_id: str
    gap_id: str

    source_actionable_action: str
    source_gap_action: str

    runtime_route: ControlledRuntimeRoute

    operator_candidates: list[str] = Field(default_factory=list)
    preferred_operator: str | None = None

    evidence_reaxis_capacity_present: bool = False
    reason_codes: list[str] = Field(default_factory=list)

    experimental_authority_scope: Literal[
        "ALPHA6_EXPERIMENTAL_RUN_ONLY"
    ] = "ALPHA6_EXPERIMENTAL_RUN_ONLY"

    production_default_changed: Literal[False] = False
    novelty_authority_created: Literal[False] = False


class PlannerControlledDiscoveryRoutingPlan(StrictModel):
    schema_version: Literal[
        "planner-controlled-discovery-routing-plan-v1"
    ] = "planner-controlled-discovery-routing-plan-v1"

    plan_id: str
    plan_sha256: str

    enabled: bool

    source_actionable_refinement_plan_id: str
    source_gap_plan_id: str
    source_portfolio_id: str

    directives: list[PlannerControlledRoutingDirective] = Field(
        default_factory=list
    )

    directive_count: int = 0
    route_counts: dict[str, int] = Field(default_factory=dict)

    experimental_authority_scope: Literal[
        "ALPHA6_EXPERIMENTAL_RUN_ONLY"
    ] = "ALPHA6_EXPERIMENTAL_RUN_ONLY"

    external_prior_art_as_positive_premise: Literal[False] = False
    operator_is_reasoning_template_not_evidence: Literal[True] = True
    production_default_changed: Literal[False] = False
    novelty_authority_created: Literal[False] = False


class PlannerControlledRoutingExecutionRow(StrictModel):
    hypothesis_id: str
    gap_id: str
    requested_route: ControlledRuntimeRoute

    observed_decisions: list[str] = Field(default_factory=list)
    observed_actions: list[str] = Field(default_factory=list)
    observed_generation_modes: list[str] = Field(default_factory=list)
    observed_reason_codes: list[str] = Field(default_factory=list)

    status: ControlledRoutingExecutionStatus
    route_exercised: bool = False
    interpretation: str


class PlannerControlledRoutingExecutionReport(StrictModel):
    schema_version: Literal[
        "planner-controlled-discovery-routing-execution-v1"
    ] = "planner-controlled-discovery-routing-execution-v1"

    report_id: str
    report_sha256: str

    source_routing_plan_id: str
    source_refinement_report_id: str

    enabled: bool
    runtime_consumed_plan: bool

    rows: list[PlannerControlledRoutingExecutionRow] = Field(
        default_factory=list
    )

    row_count: int = 0
    match_count: int = 0
    diverged_count: int = 0
    not_observed_count: int = 0
    disabled_count: int = 0

    production_default_changed: Literal[False] = False
    production_selection_authority: Literal[False] = False


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _route_for_target(
    target: Any,
    gap: Any,
) -> PlannerControlledRoutingDirective:
    action = str(target.primary_action)
    reasons: list[str] = []

    operators = [
        str(value)
        for value in getattr(
            target,
            "operator_candidates",
            [],
        )
    ]
    preferred = getattr(
        target,
        "preferred_operator",
        None,
    )
    preferred = (
        None
        if preferred is None
        else str(preferred)
    )

    unknown_operators = sorted(
        set(operators) - _ALLOWED_OPERATORS
    )
    if unknown_operators:
        raise ValueError(
            "actionable refinement plan contains unsupported "
            f"operators: {unknown_operators}"
        )

    if action == "KEEP_TESTABLE_GAP":
        route: ControlledRuntimeRoute = "KEEP_ORIGINAL"
        reasons.append(
            "bounded_higher_order_gap_kept_without_extra_novelty_optimization"
        )

    elif action == "STRUCTURAL_SHARPEN":
        if (
            getattr(gap, "target_claim_ids", None)
            and operators
        ):
            route = "GAP_SHARPEN"
            reasons.extend(
                [
                    "structural_sharpen_uses_existing_gap_sharpen_runtime",
                    "action_plan_operators_forwarded_to_prompt",
                ]
            )
        else:
            route = "HOLD_FOR_EVIDENCE"
            reasons.append(
                "structural_sharpen_missing_claim_binding_or_operator"
            )

    elif action == "EVIDENCE_REAXIS":
        if bool(
            getattr(
                target,
                "evidence_reaxis_capacity_present",
                False,
            )
        ):
            route = "FRESH_CONTEXT_REAXIS"
            reasons.append(
                "safe_unused_grounded_evidence_capacity_present"
            )
        else:
            route = "HOLD_FOR_EVIDENCE"
            reasons.append(
                "evidence_reaxis_requested_without_safe_capacity"
            )

    elif action in {
        "RETRIEVE_MECHANISM_SUPPORT",
        "HOLD_UNRESOLVED",
        "CONFLICT_REVIEW",
        "ABSTAIN",
    }:
        route = "HOLD_FOR_EVIDENCE"
        reasons.append(
            "generation_forbidden_until_evidence_or_conflict_resolution"
        )

    else:
        route = "HOLD_FOR_EVIDENCE"
        reasons.append(
            "unrecognized_action_fail_closed"
        )

    return PlannerControlledRoutingDirective(
        hypothesis_id=str(target.hypothesis_id),
        gap_id=str(gap.gap_id),
        source_actionable_action=action,
        source_gap_action=str(gap.action),
        runtime_route=route,
        operator_candidates=operators,
        preferred_operator=preferred,
        evidence_reaxis_capacity_present=bool(
            getattr(
                target,
                "evidence_reaxis_capacity_present",
                False,
            )
        ),
        reason_codes=reasons,
    )


def build_planner_controlled_routing_plan(
    *,
    actionable_refinement_plan: Any,
    gap_plan: Any,
    enabled: bool,
) -> PlannerControlledDiscoveryRoutingPlan:
    gap_by_hypothesis = {
        str(row.hypothesis_id): row
        for row in getattr(
            gap_plan,
            "gaps",
            [],
        )
    }

    directives: list[
        PlannerControlledRoutingDirective
    ] = []

    for target in getattr(
        actionable_refinement_plan,
        "targets",
        [],
    ):
        hypothesis_id = str(
            target.hypothesis_id
        )
        gap = gap_by_hypothesis.get(
            hypothesis_id
        )
        if gap is None:
            raise ValueError(
                "actionable refinement target has no matching "
                f"NoveltyGap: {hypothesis_id}"
            )

        directives.append(
            _route_for_target(
                target,
                gap,
            )
        )

    route_counts = Counter(
        row.runtime_route
        for row in directives
    )

    body = {
        "schema_version":
            "planner-controlled-discovery-routing-plan-v1",
        "enabled": bool(enabled),
        "source_actionable_refinement_plan_id": str(
            actionable_refinement_plan.plan_id
        ),
        "source_gap_plan_id": str(
            gap_plan.plan_id
        ),
        "source_portfolio_id": str(
            gap_plan.source_portfolio_id
        ),
        "directives": [
            row.model_dump(mode="json")
            for row in directives
        ],
        "directive_count": len(directives),
        "route_counts": dict(
            sorted(route_counts.items())
        ),
        "experimental_authority_scope":
            "ALPHA6_EXPERIMENTAL_RUN_ONLY",
        "external_prior_art_as_positive_premise":
            False,
        "operator_is_reasoning_template_not_evidence":
            True,
        "production_default_changed":
            False,
        "novelty_authority_created":
            False,
    }

    plan_id = _stable_id(
        "planner_controlled_discovery_routing",
        body[
            "source_actionable_refinement_plan_id"
        ],
        body["source_gap_plan_id"],
        str(bool(enabled)),
        *[
            (
                f"{row.hypothesis_id}:"
                f"{row.runtime_route}:"
                f"{row.preferred_operator}"
            )
            for row in directives
        ],
    )

    return PlannerControlledDiscoveryRoutingPlan(
        **body,
        plan_id=plan_id,
        plan_sha256=_sha256(
            {
                **body,
                "plan_id": plan_id,
            }
        ),
    )


def directive_map(
    plan: PlannerControlledDiscoveryRoutingPlan,
) -> dict[str, PlannerControlledRoutingDirective]:
    result = {
        row.hypothesis_id: row
        for row in plan.directives
    }
    if len(result) != len(plan.directives):
        raise ValueError(
            "controlled routing plan contains duplicate hypothesis directives"
        )
    return result


def structural_sharpen_gap(
    gap: Any,
    directive: PlannerControlledRoutingDirective,
) -> Any:
    if directive.runtime_route != "GAP_SHARPEN":
        return gap

    if not getattr(gap, "target_claim_ids", None):
        raise ValueError(
            "controlled GAP_SHARPEN requires target_claim_ids"
        )
    if not directive.operator_candidates:
        raise ValueError(
            "controlled GAP_SHARPEN requires operator candidates"
        )

    update = {
        "action": "gap_sharpen",
        "sharpening_operators": list(
            directive.operator_candidates
        ),
        "reason_codes": sorted(
            set(
                [
                    *list(
                        getattr(
                            gap,
                            "reason_codes",
                            [],
                        )
                    ),
                    "planner_controlled_discovery_routing",
                    (
                        "planner_preferred_operator:"
                        + str(
                            directive.preferred_operator
                        )
                    ),
                ]
            )
        ),
    }

    if hasattr(gap, "model_dump") and hasattr(
        type(gap),
        "model_validate",
    ):
        payload = gap.model_dump(mode="json")
        payload.update(update)
        return type(gap).model_validate(payload)

    return gap.model_copy(update=update)


def _route_exercised(
    *,
    route: ControlledRuntimeRoute,
    attempts: list[Any],
) -> bool:
    if route == "KEEP_ORIGINAL":
        return any(
            str(row.decision) == "kept_original"
            and "planner_controlled_route:keep_original"
            in list(
                getattr(
                    row,
                    "reason_codes",
                    [],
                )
            )
            for row in attempts
        )

    if route == "HOLD_FOR_EVIDENCE":
        return any(
            str(row.decision) == "held_for_evidence"
            and any(
                str(reason).startswith(
                    "planner_controlled_route:"
                )
                for reason in getattr(
                    row,
                    "reason_codes",
                    [],
                )
            )
            for row in attempts
        )

    if route == "GAP_SHARPEN":
        return any(
            str(row.action) == "gap_sharpen"
            and (
                str(
                    getattr(
                        row,
                        "generation_mode",
                        "none",
                    )
                )
                == "same_premise_refinement"
                or bool(
                    getattr(
                        row,
                        "refinement_generated",
                        False,
                    )
                )
            )
            for row in attempts
        )

    if route == "FRESH_CONTEXT_REAXIS":
        return any(
            str(
                getattr(
                    row,
                    "generation_mode",
                    "none",
                )
            )
            == "fresh_context_reaxis"
            for row in attempts
        )

    return False


def build_planner_controlled_routing_execution(
    *,
    routing_plan: PlannerControlledDiscoveryRoutingPlan,
    refinement_report: Any,
    runtime_consumed_plan: bool,
) -> PlannerControlledRoutingExecutionReport:
    attempts_by_gap: dict[str, list[Any]] = {}

    for attempt in getattr(
        refinement_report,
        "attempts",
        [],
    ):
        attempts_by_gap.setdefault(
            str(attempt.gap_id),
            [],
        ).append(attempt)

    rows: list[
        PlannerControlledRoutingExecutionRow
    ] = []

    for directive in routing_plan.directives:
        attempts = attempts_by_gap.get(
            directive.gap_id,
            [],
        )

        if not routing_plan.enabled:
            status: ControlledRoutingExecutionStatus = (
                "DISABLED"
            )
            exercised = False
            interpretation = (
                "Experimental planner-controlled routing was disabled; "
                "legacy Alpha6 behavior remained authoritative."
            )
        elif not runtime_consumed_plan:
            status = "NOT_OBSERVED"
            exercised = False
            interpretation = (
                "The routing plan was enabled but the runtime did not "
                "consume it."
            )
        elif not attempts:
            status = "NOT_OBSERVED"
            exercised = False
            interpretation = (
                "No refinement attempt was recorded for the controlled "
                "routing directive."
            )
        else:
            exercised = _route_exercised(
                route=directive.runtime_route,
                attempts=attempts,
            )
            if exercised:
                status = "MATCH"
                interpretation = (
                    "The experimental Alpha6 runtime exercised the route "
                    "requested by the actionable refinement plan."
                )
            else:
                status = "DIVERGED"
                interpretation = (
                    "The runtime produced an observable trajectory but did "
                    "not exercise the requested experimental route."
                )

        rows.append(
            PlannerControlledRoutingExecutionRow(
                hypothesis_id=
                    directive.hypothesis_id,
                gap_id=
                    directive.gap_id,
                requested_route=
                    directive.runtime_route,
                observed_decisions=[
                    str(row.decision)
                    for row in attempts
                ],
                observed_actions=[
                    str(row.action)
                    for row in attempts
                ],
                observed_generation_modes=[
                    str(
                        getattr(
                            row,
                            "generation_mode",
                            "none",
                        )
                    )
                    for row in attempts
                ],
                observed_reason_codes=sorted(
                    {
                        str(reason)
                        for row in attempts
                        for reason in getattr(
                            row,
                            "reason_codes",
                            [],
                        )
                    }
                ),
                status=status,
                route_exercised=exercised,
                interpretation=interpretation,
            )
        )

    counts = Counter(
        row.status
        for row in rows
    )

    body = {
        "schema_version":
            "planner-controlled-discovery-routing-execution-v1",
        "source_routing_plan_id":
            routing_plan.plan_id,
        "source_refinement_report_id":
            str(refinement_report.report_id),
        "enabled":
            routing_plan.enabled,
        "runtime_consumed_plan":
            bool(runtime_consumed_plan),
        "rows": [
            row.model_dump(mode="json")
            for row in rows
        ],
        "row_count":
            len(rows),
        "match_count":
            counts.get("MATCH", 0),
        "diverged_count":
            counts.get("DIVERGED", 0),
        "not_observed_count":
            counts.get("NOT_OBSERVED", 0),
        "disabled_count":
            counts.get("DISABLED", 0),
        "production_default_changed":
            False,
        "production_selection_authority":
            False,
    }

    report_id = _stable_id(
        "planner_controlled_discovery_routing_execution",
        routing_plan.plan_id,
        body["source_refinement_report_id"],
        str(bool(runtime_consumed_plan)),
        *[
            (
                f"{row.hypothesis_id}:"
                f"{row.requested_route}:"
                f"{row.status}"
            )
            for row in rows
        ],
    )

    return PlannerControlledRoutingExecutionReport(
        **body,
        report_id=report_id,
        report_sha256=_sha256(
            {
                **body,
                "report_id": report_id,
            }
        ),
    )


__all__ = [
    "PlannerControlledRoutingDirective",
    "PlannerControlledDiscoveryRoutingPlan",
    "PlannerControlledRoutingExecutionRow",
    "PlannerControlledRoutingExecutionReport",
    "build_planner_controlled_routing_plan",
    "directive_map",
    "structural_sharpen_gap",
    "build_planner_controlled_routing_execution",
]
