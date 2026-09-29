from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


PlannerAction = Literal[
    "RETRIEVE_MORE",
    "KEEP_RESOLVED",
    "SAME_PREMISE_SHARPEN",
    "EVIDENCE_REAXIS",
    "ABSTAIN",
]

ExecutionConformance = Literal[
    "MATCH",
    "MATCH_AFTER_FALLBACK",
    "DIVERGED",
    "NOT_COMPARABLE",
]


class EvidenceNeighborhoodAllocation(StrictModel):
    original_premise_statement_ids: list[str] = Field(default_factory=list)
    portfolio_shared_core_statement_ids: list[str] = Field(default_factory=list)
    portfolio_unique_statement_ids: list[str] = Field(default_factory=list)
    globally_unused_safe_statement_ids: list[str] = Field(default_factory=list)
    ranked_candidate_statement_ids: list[str] = Field(default_factory=list)
    candidate_scores: dict[str, float] = Field(default_factory=dict)

    candidate_capacity_present: bool = False
    allocation_basis: Literal[
        "NO_SAFE_UNUSED_EVIDENCE",
        "LEXICAL_GAP_OVERLAP_ONLY",
    ] = "NO_SAFE_UNUSED_EVIDENCE"

    scientific_relevance_established: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False


class OperatorAllocation(StrictModel):
    allowed_operators: list[str] = Field(default_factory=list)
    preferred_operator: str | None = None
    operator_scores: dict[str, int] = Field(default_factory=dict)

    allocation_basis: Literal[
        "NO_EXPLICIT_OPERATOR",
        "EXPLICIT_GAP_OPERATOR_WITH_LEXICAL_CUE",
        "EXPLICIT_GAP_OPERATOR_NO_PREFERENCE",
    ] = "NO_EXPLICIT_OPERATOR"

    inferred_operator_outside_allowed_set: Literal[False] = False
    operator_is_scientific_evidence: Literal[False] = False


class NegativeSpacePlannerTarget(StrictModel):
    gap_id: str
    hypothesis_id: str
    source_external_status: str
    source_gap_action: str

    initial_action: PlannerAction
    if_resolved_candidate: PlannerAction
    if_known_or_extension: PlannerAction
    if_insufficient_evidence: PlannerAction
    if_conflicting_prior_art: PlannerAction

    evidence_allocation: EvidenceNeighborhoodAllocation
    operator_allocation: OperatorAllocation

    conceptual_knownness_advisory: Literal[
        "NOT_AVAILABLE",
        "HOLD_COVERAGE",
        "REAXIS_IF_SUPPORTED",
        "KEEP_DEPTH",
    ] = "NOT_AVAILABLE"

    novelty_depth_class: str | None = None
    novelty_depth_advisory: str | None = None

    actionable_refinement_action: str | None = None
    actionable_operator_candidates: list[str] = Field(default_factory=list)
    actionable_preferred_operator: str | None = None

    policy_reason_codes: list[str] = Field(default_factory=list)

    external_prior_art_as_positive_premise: Literal[False] = False
    unused_evidence_relevance_inferred: Literal[False] = False
    production_selection_authority: Literal[False] = False


class NegativeSpacePlannerPlan(StrictModel):
    schema_version: Literal[
        "negative-space-discovery-planner-plan-v1"
    ] = "negative-space-discovery-planner-plan-v1"

    plan_id: str
    plan_sha256: str
    source_context_id: str
    source_portfolio_id: str
    source_gap_plan_id: str

    targets: list[NegativeSpacePlannerTarget] = Field(default_factory=list)

    target_count: int = 0
    retrieve_first_count: int = 0
    evidence_reaxis_available_count: int = 0
    operator_sharpen_available_count: int = 0

    novelty_depth_profile_consumed: bool = False
    novelty_depth_advisory_counts: dict[str, int] = Field(
        default_factory=dict
    )

    actionable_refinement_plan_consumed: bool = False
    actionable_refinement_action_counts: dict[str, int] = Field(
        default_factory=dict
    )
    actionable_operator_target_count: int = 0

    planner_policy_version: Literal[
        "negative-space-discovery-planner-policy-v1"
    ] = "negative-space-discovery-planner-policy-v1"

    diagnostic_only: Literal[True] = True
    generation_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False


class PlannerObservedAttempt(StrictModel):
    decision: str
    generation_mode: str
    targeted_external_status: str | None = None
    final_external_status: str | None = None
    observed_action: PlannerAction
    reason_codes: list[str] = Field(default_factory=list)


class NegativeSpacePlannerExecutionRow(StrictModel):
    gap_id: str
    hypothesis_id: str

    effective_external_status: str | None = None
    planned_effective_action: PlannerAction | None = None

    observed_attempts: list[PlannerObservedAttempt] = Field(default_factory=list)
    observed_actions: list[PlannerAction] = Field(default_factory=list)

    conformance: ExecutionConformance
    interpretation: str


class NegativeSpacePlannerExecutionReport(StrictModel):
    schema_version: Literal[
        "negative-space-discovery-planner-execution-v1"
    ] = "negative-space-discovery-planner-execution-v1"

    report_id: str
    report_sha256: str

    source_planner_plan_id: str
    source_refinement_report_id: str
    source_final_portfolio_id: str

    rows: list[NegativeSpacePlannerExecutionRow] = Field(default_factory=list)

    row_count: int = 0
    match_count: int = 0
    match_after_fallback_count: int = 0
    diverged_count: int = 0
    not_comparable_count: int = 0

    planned_action_counts: dict[str, int] = Field(default_factory=dict)
    observed_action_counts: dict[str, int] = Field(default_factory=dict)

    planner_changed_runtime_behavior: Literal[False] = False
    ranking_computed: Literal[False] = False
    production_selection_changed: Literal[False] = False


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)?")

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "under", "with", "without", "when",
    "which", "while",
}

_OPERATOR_CUES: dict[str, tuple[str, ...]] = {
    "MODERATOR": (
        "moderate", "moderates", "modulate", "modulates", "condition",
        "conditions", "depend", "depends", "context", "orientation",
    ),
    "INTERACTION": (
        "interaction", "interact", "joint", "jointly", "combined",
        "combination", "synergy", "synergistic",
    ),
    "RESIDUAL": (
        "residual", "remaining", "after", "accounting", "normalized",
        "normalization", "correction", "corrected",
    ),
    "BOUNDARY": (
        "boundary", "regime", "threshold", "limit", "weakens", "fails",
        "failure", "reverse", "reverses", "transition",
    ),
    "PROXY_DECOUPLING": (
        "proxy", "track", "tracks", "correlate", "correlation", "decouple",
        "decoupling", "diverge", "divergence",
    ),
    "COMPENSATION_LIMIT": (
        "compensation", "compensate", "mitigation", "mitigate",
        "correction", "effective", "effectiveness", "limit",
    ),
}


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


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(str(text))
        if token.lower() not in _STOP
    }


def _gap_text(gap: Any) -> str:
    return " ".join(
        [
            str(getattr(gap, "differentiator", "") or ""),
            *[
                str(x)
                for x in getattr(gap, "unresolved_boundary", [])
            ],
        ]
    )


def _safe_unused_statement_ids(*, context: Any, portfolio: Any) -> list[str]:
    used = {
        str(statement_id)
        for card in getattr(portfolio, "hypotheses", [])
        for statement_id in getattr(card, "premise_statement_ids", [])
    }

    result: list[str] = []
    for row in getattr(context, "evidence_statements", []):
        sid = str(getattr(row, "statement_id", "") or "")
        if not sid or sid in used:
            continue
        if not bool(getattr(row, "eligible_as_premise", False)):
            continue
        if list(getattr(row, "premise_restrictions", []) or []):
            continue
        result.append(sid)

    return result


def _evidence_topology(
    *,
    context: Any,
    portfolio: Any,
    hypothesis_id: str,
    gap: Any,
) -> EvidenceNeighborhoodAllocation:
    cards = list(getattr(portfolio, "hypotheses", []) or [])
    by_id = {str(card.hypothesis_id): card for card in cards}
    card = by_id[hypothesis_id]

    premise_sets = {
        str(row.hypothesis_id): set(
            map(str, getattr(row, "premise_statement_ids", []))
        )
        for row in cards
    }

    original = list(
        dict.fromkeys(map(str, getattr(card, "premise_statement_ids", [])))
    )

    if len(cards) >= 2:
        shared_core = set.intersection(*premise_sets.values())
    else:
        shared_core = set()

    usage = Counter(
        sid
        for values in premise_sets.values()
        for sid in values
    )

    unique = [sid for sid in original if usage[sid] == 1]

    safe_unused = _safe_unused_statement_ids(
        context=context,
        portfolio=portfolio,
    )

    statement_by_id = {
        str(row.statement_id): row
        for row in getattr(context, "evidence_statements", [])
        if getattr(row, "statement_id", None)
    }

    gap_tokens = _tokens(_gap_text(gap))
    scored: list[tuple[float, str]] = []

    for sid in safe_unused:
        row = statement_by_id[sid]
        statement_tokens = _tokens(str(getattr(row, "text", "")))
        if not gap_tokens or not statement_tokens:
            score = 0.0
        else:
            inter = len(gap_tokens & statement_tokens)
            union = len(gap_tokens | statement_tokens)
            score = inter / union if union else 0.0
        scored.append((round(float(score), 6), sid))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    ranked = [sid for score, sid in scored if score > 0][:4]
    candidate_scores = {sid: score for score, sid in scored if score > 0}

    return EvidenceNeighborhoodAllocation(
        original_premise_statement_ids=original,
        portfolio_shared_core_statement_ids=[
            sid for sid in original if sid in shared_core
        ],
        portfolio_unique_statement_ids=unique,
        globally_unused_safe_statement_ids=safe_unused,
        ranked_candidate_statement_ids=ranked,
        candidate_scores=candidate_scores,
        candidate_capacity_present=bool(safe_unused),
        allocation_basis=(
            "LEXICAL_GAP_OVERLAP_ONLY"
            if ranked
            else "NO_SAFE_UNUSED_EVIDENCE"
        ),
    )


def _operator_allocation(gap: Any) -> OperatorAllocation:
    allowed = [
        str(x)
        for x in getattr(gap, "sharpening_operators", [])
    ]

    if not allowed:
        return OperatorAllocation()

    text_tokens = _tokens(_gap_text(gap))
    scores: dict[str, int] = {}

    for operator in allowed:
        cues = _OPERATOR_CUES.get(operator, ())
        scores[operator] = int(
            sum(cue in text_tokens for cue in cues)
        )

    preferred = None
    if scores:
        best = max(scores.values())
        if best > 0:
            preferred = sorted(
                operator
                for operator, score in scores.items()
                if score == best
            )[0]

    return OperatorAllocation(
        allowed_operators=allowed,
        preferred_operator=preferred,
        operator_scores=scores,
        allocation_basis=(
            "EXPLICIT_GAP_OPERATOR_WITH_LEXICAL_CUE"
            if preferred is not None
            else "EXPLICIT_GAP_OPERATOR_NO_PREFERENCE"
        ),
    )


def _policy_actions(
    *,
    gap_action: str,
    evidence_capacity: bool,
    operator_capacity: bool,
) -> tuple[
    PlannerAction,
    PlannerAction,
    PlannerAction,
    PlannerAction,
    PlannerAction,
    list[str],
]:
    reasons: list[str] = []

    if gap_action == "keep":
        return (
            "KEEP_RESOLVED",
            "KEEP_RESOLVED",
            "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            ["source_gap_action_keep"],
        )

    if gap_action == "reject":
        return (
            "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            ["source_gap_action_reject"],
        )

    initial: PlannerAction = "RETRIEVE_MORE"

    if gap_action == "gap_sharpen":
        return (
            initial,
            "SAME_PREMISE_SHARPEN",
            "SAME_PREMISE_SHARPEN",
            "ABSTAIN",
            "SAME_PREMISE_SHARPEN",
            [
                "explicit_gap_sharpen_action",
                "resolved_status_does_not_short_circuit_explicit_sharpen",
            ],
        )

    if gap_action == "targeted_search_only":
        return (
            initial,
            "KEEP_RESOLVED",
            "EVIDENCE_REAXIS" if evidence_capacity else "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            ["retrieval_first_non_destructive"],
        )

    if gap_action == "targeted_search_then_refine":
        if evidence_capacity:
            known_action: PlannerAction = "EVIDENCE_REAXIS"
            reasons.append("safe_unused_evidence_capacity_present")
        else:
            known_action = "SAME_PREMISE_SHARPEN"
            reasons.append("no_safe_unused_evidence_capacity")

        reasons.append("retrieval_before_regeneration")
        return (
            initial,
            "KEEP_RESOLVED",
            known_action,
            "ABSTAIN",
            "SAME_PREMISE_SHARPEN" if operator_capacity else "ABSTAIN",
            reasons,
        )

    if gap_action == "refine_away_from_conflict":
        return (
            initial,
            "KEEP_RESOLVED",
            (
                "EVIDENCE_REAXIS"
                if evidence_capacity
                else (
                    "SAME_PREMISE_SHARPEN"
                    if operator_capacity
                    else "ABSTAIN"
                )
            ),
            "ABSTAIN",
            "SAME_PREMISE_SHARPEN" if operator_capacity else "ABSTAIN",
            ["conflict_requires_bounded_redirection"],
        )

    return (
        initial,
        "KEEP_RESOLVED",
        "ABSTAIN",
        "ABSTAIN",
        "ABSTAIN",
        ["unrecognized_gap_action_fail_closed"],
    )


def _apply_novelty_depth_advisory(
    *,
    advisory: str | None,
    evidence_capacity: bool,
    operator_capacity: bool,
    resolved: PlannerAction,
    known: PlannerAction,
    insufficient: PlannerAction,
    conflict: PlannerAction,
) -> tuple[
    PlannerAction,
    PlannerAction,
    PlannerAction,
    PlannerAction,
    list[str],
]:
    reasons: list[str] = []

    if advisory is None:
        return resolved, known, insufficient, conflict, reasons

    reasons.append("novelty_depth_advisory:" + advisory.lower())

    if advisory == "KEEP_TESTABLE_GAP":
        return (
            "KEEP_RESOLVED",
            "EVIDENCE_REAXIS" if evidence_capacity else "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            reasons,
        )

    if advisory == "SHARPEN_LOCAL_EXTENSION":
        sharpen = (
            "SAME_PREMISE_SHARPEN"
            if operator_capacity
            else "ABSTAIN"
        )
        return sharpen, sharpen, "ABSTAIN", sharpen, reasons

    if advisory in {
        "RETRIEVE_MECHANISM_SUPPORT",
        "REAXIS_OR_ABSTAIN",
    }:
        reaxis = (
            "EVIDENCE_REAXIS"
            if evidence_capacity
            else "ABSTAIN"
        )
        return reaxis, reaxis, "ABSTAIN", "ABSTAIN", reasons

    if advisory in {
        "HOLD_UNRESOLVED",
        "KNOWN_RELATION",
        "CONFLICT_REVIEW",
    }:
        return "ABSTAIN", "ABSTAIN", "ABSTAIN", "ABSTAIN", reasons

    return resolved, known, insufficient, conflict, reasons


def _apply_actionable_refinement_action(
    *,
    action: str | None,
    resolved: PlannerAction,
    known: PlannerAction,
    insufficient: PlannerAction,
    conflict: PlannerAction,
) -> tuple[
    PlannerAction,
    PlannerAction,
    PlannerAction,
    PlannerAction,
    list[str],
]:
    reasons: list[str] = []

    if action is None:
        return resolved, known, insufficient, conflict, reasons

    reasons.append(
        "actionable_refinement_plan:" + action.lower()
    )

    if action == "KEEP_TESTABLE_GAP":
        return (
            "KEEP_RESOLVED",
            "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            reasons,
        )

    if action == "STRUCTURAL_SHARPEN":
        return (
            "SAME_PREMISE_SHARPEN",
            "SAME_PREMISE_SHARPEN",
            "ABSTAIN",
            "SAME_PREMISE_SHARPEN",
            reasons,
        )

    if action == "EVIDENCE_REAXIS":
        return (
            "EVIDENCE_REAXIS",
            "EVIDENCE_REAXIS",
            "ABSTAIN",
            "ABSTAIN",
            reasons,
        )

    if action in {
        "RETRIEVE_MECHANISM_SUPPORT",
        "HOLD_UNRESOLVED",
        "CONFLICT_REVIEW",
        "ABSTAIN",
    }:
        return (
            "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            "ABSTAIN",
            reasons,
        )

    return resolved, known, insufficient, conflict, reasons


def build_negative_space_planner_plan(
    *,
    context: Any,
    portfolio: Any,
    gap_plan: Any,
    novelty_depth_profile: Any | None = None,
    actionable_refinement_plan: Any | None = None,
) -> NegativeSpacePlannerPlan:
    portfolio_ids = {
        str(row.hypothesis_id)
        for row in getattr(portfolio, "hypotheses", [])
    }

    depth_by_hypothesis = (
        {
            str(row.hypothesis_id): row
            for row in getattr(
                novelty_depth_profile,
                "profiles",
                [],
            )
        }
        if novelty_depth_profile is not None
        else {}
    )

    actionable_by_hypothesis = (
        {
            str(row.hypothesis_id): row
            for row in getattr(
                actionable_refinement_plan,
                "targets",
                [],
            )
        }
        if actionable_refinement_plan is not None
        else {}
    )

    targets: list[NegativeSpacePlannerTarget] = []

    for gap in getattr(gap_plan, "gaps", []):
        hypothesis_id = str(gap.hypothesis_id)
        if hypothesis_id not in portfolio_ids:
            raise ValueError(
                "planner gap hypothesis missing from source portfolio: "
                f"{hypothesis_id}"
            )

        evidence = _evidence_topology(
            context=context,
            portfolio=portfolio,
            hypothesis_id=hypothesis_id,
            gap=gap,
        )
        operators = _operator_allocation(gap)

        (
            initial,
            resolved,
            known,
            insufficient,
            conflict,
            reasons,
        ) = _policy_actions(
            gap_action=str(gap.action),
            evidence_capacity=evidence.candidate_capacity_present,
            operator_capacity=bool(operators.allowed_operators),
        )

        depth_row = depth_by_hypothesis.get(hypothesis_id)
        depth_class = (
            None
            if depth_row is None
            else str(depth_row.novelty_depth_class)
        )
        depth_advisory = (
            None
            if depth_row is None
            else str(depth_row.planner_advisory)
        )

        (
            resolved,
            known,
            insufficient,
            conflict,
            depth_reasons,
        ) = _apply_novelty_depth_advisory(
            advisory=depth_advisory,
            evidence_capacity=evidence.candidate_capacity_present,
            operator_capacity=bool(operators.allowed_operators),
            resolved=resolved,
            known=known,
            insufficient=insufficient,
            conflict=conflict,
        )
        reasons.extend(depth_reasons)

        actionable_row = actionable_by_hypothesis.get(
            hypothesis_id
        )
        actionable_action = (
            None
            if actionable_row is None
            else str(actionable_row.primary_action)
        )
        actionable_operator_candidates = (
            []
            if actionable_row is None
            else [
                str(value)
                for value in getattr(
                    actionable_row,
                    "operator_candidates",
                    [],
                )
            ]
        )
        actionable_preferred_operator = (
            None
            if actionable_row is None
            or getattr(
                actionable_row,
                "preferred_operator",
                None,
            )
            is None
            else str(
                actionable_row.preferred_operator
            )
        )

        (
            resolved,
            known,
            insufficient,
            conflict,
            actionable_reasons,
        ) = _apply_actionable_refinement_action(
            action=actionable_action,
            resolved=resolved,
            known=known,
            insufficient=insufficient,
            conflict=conflict,
        )
        reasons.extend(actionable_reasons)

        targets.append(
            NegativeSpacePlannerTarget(
                gap_id=str(gap.gap_id),
                hypothesis_id=hypothesis_id,
                source_external_status=str(gap.source_external_status),
                source_gap_action=str(gap.action),
                initial_action=initial,
                if_resolved_candidate=resolved,
                if_known_or_extension=known,
                if_insufficient_evidence=insufficient,
                if_conflicting_prior_art=conflict,
                evidence_allocation=evidence,
                operator_allocation=operators,
                novelty_depth_class=depth_class,
                novelty_depth_advisory=depth_advisory,
                actionable_refinement_action=actionable_action,
                actionable_operator_candidates=
                    actionable_operator_candidates,
                actionable_preferred_operator=
                    actionable_preferred_operator,
                policy_reason_codes=reasons,
            )
        )

    body = {
        "schema_version": "negative-space-discovery-planner-plan-v1",
        "source_context_id": str(context.context_id),
        "source_portfolio_id": str(portfolio.portfolio_id),
        "source_gap_plan_id": str(gap_plan.plan_id),
        "targets": [
            row.model_dump(mode="json") for row in targets
        ],
        "target_count": len(targets),
        "retrieve_first_count": sum(
            row.initial_action == "RETRIEVE_MORE" for row in targets
        ),
        "evidence_reaxis_available_count": sum(
            row.if_known_or_extension == "EVIDENCE_REAXIS"
            for row in targets
        ),
        "operator_sharpen_available_count": sum(
            (
                row.if_resolved_candidate == "SAME_PREMISE_SHARPEN"
                or row.if_known_or_extension == "SAME_PREMISE_SHARPEN"
                or row.if_conflicting_prior_art == "SAME_PREMISE_SHARPEN"
            )
            for row in targets
        ),
        "novelty_depth_profile_consumed":
            novelty_depth_profile is not None,
        "novelty_depth_advisory_counts":
            dict(
                sorted(
                    Counter(
                        row.novelty_depth_advisory
                        for row in targets
                        if row.novelty_depth_advisory is not None
                    ).items()
                )
            ),
        "actionable_refinement_plan_consumed":
            actionable_refinement_plan is not None,
        "actionable_refinement_action_counts":
            dict(
                sorted(
                    Counter(
                        row.actionable_refinement_action
                        for row in targets
                        if row.actionable_refinement_action is not None
                    ).items()
                )
            ),
        "actionable_operator_target_count":
            sum(
                bool(row.actionable_operator_candidates)
                for row in targets
            ),
        "planner_policy_version":
            "negative-space-discovery-planner-policy-v1",
        "diagnostic_only": True,
        "generation_authority_created": False,
        "novelty_authority_created": False,
        "production_selection_authority": False,
    }

    plan_id = _stable_id(
        "negative_space_discovery_planner",
        body["source_context_id"],
        body["source_portfolio_id"],
        body["source_gap_plan_id"],
        *[
            (
                f"{row.gap_id}:"
                f"{row.initial_action}:"
                f"{row.if_resolved_candidate}:"
                f"{row.if_known_or_extension}"
            )
            for row in targets
        ],
    )

    return NegativeSpacePlannerPlan(
        **body,
        plan_id=plan_id,
        plan_sha256=_sha256({**body, "plan_id": plan_id}),
    )


_RESOLVED = {
    "PLAUSIBLY_NOVEL",
    "NEW_COMBINATION_OF_KNOWN_EFFECTS",
    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
}

_KNOWN_OR_EXTENSION = {
    "WELL_ESTABLISHED",
    "LITERATURE_SUPPORTED_EXTENSION",
}

_CONFLICT = {"CONFLICTING_PRIOR_ART"}


def planned_action_for_external_status(
    target: NegativeSpacePlannerTarget,
    external_status: str | None,
) -> PlannerAction | None:
    if external_status is None:
        return None

    status = str(external_status)

    if status in _RESOLVED:
        return target.if_resolved_candidate
    if status in _KNOWN_OR_EXTENSION:
        return target.if_known_or_extension
    if status == "INSUFFICIENT_SEARCH_EVIDENCE":
        return target.if_insufficient_evidence
    if status in _CONFLICT:
        return target.if_conflicting_prior_art
    return None


def _observed_action(attempt: Any) -> PlannerAction:
    generation_mode = str(
        getattr(attempt, "generation_mode", "none") or "none"
    )
    decision = str(getattr(attempt, "decision", "") or "")

    if generation_mode == "fresh_context_reaxis":
        return "EVIDENCE_REAXIS"
    if generation_mode == "same_premise_refinement":
        return "SAME_PREMISE_SHARPEN"
    if decision == "accepted_reaxis":
        return "EVIDENCE_REAXIS"
    if decision == "accepted_refinement":
        return "SAME_PREMISE_SHARPEN"
    if decision == "kept_original":
        return "KEEP_RESOLVED"
    return "ABSTAIN"


def evaluate_negative_space_planner_execution(
    *,
    plan: NegativeSpacePlannerPlan,
    refinement_report: Any,
) -> NegativeSpacePlannerExecutionReport:
    attempts_by_gap: dict[str, list[Any]] = {}
    for attempt in getattr(refinement_report, "attempts", []):
        attempts_by_gap.setdefault(str(attempt.gap_id), []).append(attempt)

    rows: list[NegativeSpacePlannerExecutionRow] = []

    for target in plan.targets:
        attempts = attempts_by_gap.get(target.gap_id, [])

        observed: list[PlannerObservedAttempt] = []
        for attempt in attempts:
            observed.append(
                PlannerObservedAttempt(
                    decision=str(attempt.decision),
                    generation_mode=str(
                        getattr(attempt, "generation_mode", "none") or "none"
                    ),
                    targeted_external_status=(
                        None
                        if getattr(attempt, "targeted_external_status", None)
                        is None
                        else str(attempt.targeted_external_status)
                    ),
                    final_external_status=(
                        None
                        if getattr(attempt, "final_external_status", None)
                        is None
                        else str(attempt.final_external_status)
                    ),
                    observed_action=_observed_action(attempt),
                    reason_codes=[
                        str(x)
                        for x in getattr(attempt, "reason_codes", [])
                    ],
                )
            )

        effective_status = None
        for row in observed:
            if row.targeted_external_status is not None:
                effective_status = row.targeted_external_status
        if effective_status is None:
            effective_status = target.source_external_status

        planned = planned_action_for_external_status(
            target,
            effective_status,
        )

        observed_actions = [row.observed_action for row in observed]

        if not observed:
            conformance: ExecutionConformance = "NOT_COMPARABLE"
            interpretation = "No Alpha6 attempt was available for this planner target."
        elif planned is None:
            conformance = "NOT_COMPARABLE"
            interpretation = "The effective external status has no planner policy mapping."
        elif observed_actions and observed_actions[0] == planned:
            conformance = "MATCH"
            interpretation = "The first observed Alpha6 action matches the planner policy."
        elif planned in observed_actions:
            conformance = "MATCH_AFTER_FALLBACK"
            interpretation = (
                "The planner action occurred after an earlier bounded Alpha6 attempt."
            )
        else:
            conformance = "DIVERGED"
            interpretation = (
                "Observed Alpha6 behavior differs from the planner's descriptive "
                "counterfactual policy. Runtime behavior remains authoritative; "
                "the planner has no selection authority."
            )

        rows.append(
            NegativeSpacePlannerExecutionRow(
                gap_id=target.gap_id,
                hypothesis_id=target.hypothesis_id,
                effective_external_status=effective_status,
                planned_effective_action=planned,
                observed_attempts=observed,
                observed_actions=observed_actions,
                conformance=conformance,
                interpretation=interpretation,
            )
        )

    planned_counts = Counter(
        row.planned_effective_action
        for row in rows
        if row.planned_effective_action is not None
    )
    observed_counts = Counter(
        action
        for row in rows
        for action in row.observed_actions
    )

    body = {
        "schema_version":
            "negative-space-discovery-planner-execution-v1",
        "source_planner_plan_id": plan.plan_id,
        "source_refinement_report_id": str(refinement_report.report_id),
        "source_final_portfolio_id": str(refinement_report.final_portfolio_id),
        "rows": [row.model_dump(mode="json") for row in rows],
        "row_count": len(rows),
        "match_count": sum(row.conformance == "MATCH" for row in rows),
        "match_after_fallback_count": sum(
            row.conformance == "MATCH_AFTER_FALLBACK" for row in rows
        ),
        "diverged_count": sum(row.conformance == "DIVERGED" for row in rows),
        "not_comparable_count": sum(
            row.conformance == "NOT_COMPARABLE" for row in rows
        ),
        "planned_action_counts": {
            str(key): value
            for key, value in sorted(
                planned_counts.items(),
                key=lambda item: str(item[0]),
            )
        },
        "observed_action_counts": dict(sorted(observed_counts.items())),
        "planner_changed_runtime_behavior": False,
        "ranking_computed": False,
        "production_selection_changed": False,
    }

    report_id = _stable_id(
        "negative_space_discovery_planner_execution",
        plan.plan_id,
        body["source_refinement_report_id"],
        *[
            (
                f"{row.gap_id}:"
                f"{row.planned_effective_action}:"
                f"{row.conformance}"
            )
            for row in rows
        ],
    )

    return NegativeSpacePlannerExecutionReport(
        **body,
        report_id=report_id,
        report_sha256=_sha256({**body, "report_id": report_id}),
    )


__all__ = [
    "EvidenceNeighborhoodAllocation",
    "OperatorAllocation",
    "NegativeSpacePlannerTarget",
    "NegativeSpacePlannerPlan",
    "PlannerObservedAttempt",
    "NegativeSpacePlannerExecutionRow",
    "NegativeSpacePlannerExecutionReport",
    "build_negative_space_planner_plan",
    "planned_action_for_external_status",
    "evaluate_negative_space_planner_execution",
]
