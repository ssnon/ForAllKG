from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ClosureMode = Literal[
    "OBSERVE_ONLY",
    "TARGETED_SEARCH_ONLY",
    "TARGETED_SEARCH_THEN_REGENERATE",
    "SAME_PREMISE_OPERATOR_SHARPEN",
    "CONFLICT_AVOIDING_REFINEMENT",
    "NO_REGENERATION",
]


class NegativeSpaceEvidenceAllocation(StrictModel):
    hypothesis_id: str
    original_premise_statement_ids: list[str] = Field(default_factory=list)
    shared_core_statement_ids: list[str] = Field(default_factory=list)
    portfolio_unique_premise_statement_ids: list[str] = Field(default_factory=list)

    globally_unused_eligible_statement_ids: list[str] = Field(default_factory=list)
    candidate_alternative_premise_ids: list[str] = Field(default_factory=list)

    alternative_capacity_present: bool = False
    alternative_relevance_established: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False


class NegativeSpaceOperatorAllocation(StrictModel):
    operators: list[str] = Field(default_factory=list)
    operator_source: Literal[
        "EXPLICIT_NOVELTY_GAP",
        "NONE",
    ] = "NONE"

    operator_inferred_from_hypothesis_text: Literal[False] = False
    operator_is_scientific_evidence: Literal[False] = False


class NoveltyGuidedNegativeSpaceTarget(StrictModel):
    gap_id: str
    hypothesis_id: str
    source_external_status: str
    source_gap_action: str

    closure_mode: ClosureMode
    eligible_for_guided_regeneration: bool

    target_claim_ids: list[str] = Field(default_factory=list)
    differentiator: str = ""
    already_known_boundary: list[str] = Field(default_factory=list)
    unresolved_boundary: list[str] = Field(default_factory=list)
    targeted_query_count: int = 0

    evidence_allocation: NegativeSpaceEvidenceAllocation
    operator_allocation: NegativeSpaceOperatorAllocation

    reason_codes: list[str] = Field(default_factory=list)

    external_prior_art_as_positive_premise: Literal[False] = False
    scientific_relevance_of_unused_evidence_inferred: Literal[False] = False
    production_selection_authority: Literal[False] = False


class NoveltyGuidedNegativeSpacePlan(StrictModel):
    schema_version: Literal[
        "novelty-guided-negative-space-plan-v1"
    ] = "novelty-guided-negative-space-plan-v1"

    plan_id: str
    plan_sha256: str

    source_context_id: str
    source_portfolio_id: str
    source_gap_plan_id: str
    source_evidence_diversity_report_id: str

    targets: list[NoveltyGuidedNegativeSpaceTarget] = Field(default_factory=list)

    target_count: int = 0
    regeneration_eligible_count: int = 0
    operator_target_count: int = 0
    operator_opportunity_count: int = 0
    evidence_alternative_capacity_target_count: int = 0

    diagnostic_only: Literal[True] = True
    generation_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False


class NoveltyGuidedRegenerationTrace(StrictModel):
    gap_id: str
    hypothesis_id: str

    attempt_observed: bool = False
    decision: str | None = None
    generation_mode: str | None = None
    candidate_hypothesis_id: str | None = None
    final_hypothesis_id: str | None = None

    refinement_generated: bool = False
    context_grounding_valid: bool = False
    grounding_preserved: bool = False

    targeted_external_status: str | None = None
    final_external_status: str | None = None

    fresh_external_verification_observed: bool = False
    accepted_regeneration: bool = False
    closed_loop_observed: bool = False

    post_generation_semantic_stable: bool | None = None
    post_generation_scientific_action: str | None = None
    post_generation_selection_class: str | None = None

    reason_codes: list[str] = Field(default_factory=list)


class NoveltyGuidedDiscoveryClosureReport(StrictModel):
    schema_version: Literal[
        "novelty-guided-discovery-closure-report-v1"
    ] = "novelty-guided-discovery-closure-report-v1"

    report_id: str
    report_sha256: str

    source_negative_space_plan_id: str
    source_refinement_report_id: str
    source_final_portfolio_id: str

    traces: list[NoveltyGuidedRegenerationTrace] = Field(default_factory=list)

    target_count: int = 0
    regeneration_attempted_count: int = 0
    regeneration_generated_count: int = 0
    accepted_regeneration_count: int = 0
    fresh_external_verification_count: int = 0
    closed_loop_observed_count: int = 0

    gap_action_counts: dict[str, int] = Field(default_factory=dict)
    generation_mode_counts: dict[str, int] = Field(default_factory=dict)
    decision_counts: dict[str, int] = Field(default_factory=dict)

    conceptual_knownness_required_for_closure: Literal[False] = False
    external_prior_art_as_positive_premise: Literal[False] = False
    ranking_computed: Literal[False] = False
    candidate_survival_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False


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


def _closure_mode(action: str) -> ClosureMode:
    mapping: dict[str, ClosureMode] = {
        "keep": "OBSERVE_ONLY",
        "targeted_search_only": "TARGETED_SEARCH_ONLY",
        "targeted_search_then_refine": "TARGETED_SEARCH_THEN_REGENERATE",
        "gap_sharpen": "SAME_PREMISE_OPERATOR_SHARPEN",
        "refine_away_from_conflict": "CONFLICT_AVOIDING_REFINEMENT",
        "reject": "NO_REGENERATION",
    }
    return mapping.get(str(action), "OBSERVE_ONLY")


def _regeneration_eligible(action: str) -> bool:
    return str(action) in {
        "targeted_search_then_refine",
        "gap_sharpen",
        "refine_away_from_conflict",
    }


def _by_hypothesis(rows: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        hid = str(getattr(row, "hypothesis_id", "") or "")
        if hid:
            result[hid] = row
    return result


def _safe_unused_eligible_ids(
    *,
    context: Any,
    evidence_diversity: Any,
) -> list[str]:
    statement_index = {
        str(getattr(row, "statement_id")): row
        for row in getattr(context, "evidence_statements", [])
        if getattr(row, "statement_id", None)
    }

    result: list[str] = []
    for raw in getattr(
        evidence_diversity,
        "unused_eligible_statement_ids",
        [],
    ):
        sid = str(raw)
        statement = statement_index.get(sid)
        if statement is None:
            continue

        if not bool(getattr(statement, "eligible_as_premise", False)):
            continue

        restrictions = list(
            getattr(statement, "premise_restrictions", []) or []
        )
        if restrictions:
            continue

        if sid not in result:
            result.append(sid)

    return result


def build_novelty_guided_negative_space_plan(
    *,
    context: Any,
    portfolio: Any,
    gap_plan: Any,
    evidence_diversity: Any,
) -> NoveltyGuidedNegativeSpacePlan:
    portfolio_by_hypothesis = _by_hypothesis(
        list(getattr(portfolio, "hypotheses", []) or [])
    )
    evidence_card_by_hypothesis = _by_hypothesis(
        list(getattr(evidence_diversity, "cards", []) or [])
    )

    shared_core = [
        str(x)
        for x in getattr(
            evidence_diversity,
            "shared_core_statement_ids",
            [],
        )
    ]

    safe_unused = _safe_unused_eligible_ids(
        context=context,
        evidence_diversity=evidence_diversity,
    )

    targets: list[NoveltyGuidedNegativeSpaceTarget] = []

    for gap in list(getattr(gap_plan, "gaps", []) or []):
        hypothesis_id = str(gap.hypothesis_id)
        card = portfolio_by_hypothesis.get(hypothesis_id)
        if card is None:
            raise ValueError(
                "NoveltyGap references hypothesis outside source "
                f"portfolio: {hypothesis_id}"
            )

        evidence_card = evidence_card_by_hypothesis.get(hypothesis_id)

        original_premises = [
            str(x)
            for x in getattr(
                card,
                "premise_statement_ids",
                [],
            )
        ]

        unique_premises = (
            [
                str(x)
                for x in getattr(
                    evidence_card,
                    "portfolio_unique_premise_statement_ids",
                    [],
                )
            ]
            if evidence_card is not None
            else []
        )

        alternatives = [
            sid
            for sid in safe_unused
            if sid not in set(original_premises)
        ]

        operators = [
            str(x)
            for x in getattr(
                gap,
                "sharpening_operators",
                [],
            )
        ]

        allocation = NegativeSpaceEvidenceAllocation(
            hypothesis_id=hypothesis_id,
            original_premise_statement_ids=original_premises,
            shared_core_statement_ids=shared_core,
            portfolio_unique_premise_statement_ids=unique_premises,
            globally_unused_eligible_statement_ids=safe_unused,
            candidate_alternative_premise_ids=alternatives,
            alternative_capacity_present=bool(alternatives),
        )

        operator = NegativeSpaceOperatorAllocation(
            operators=operators,
            operator_source=(
                "EXPLICIT_NOVELTY_GAP"
                if operators
                else "NONE"
            ),
        )

        action = str(gap.action)

        targets.append(
            NoveltyGuidedNegativeSpaceTarget(
                gap_id=str(gap.gap_id),
                hypothesis_id=hypothesis_id,
                source_external_status=str(
                    gap.source_external_status
                ),
                source_gap_action=action,
                closure_mode=_closure_mode(action),
                eligible_for_guided_regeneration=_regeneration_eligible(action),
                target_claim_ids=[
                    str(x)
                    for x in getattr(
                        gap,
                        "target_claim_ids",
                        [],
                    )
                ],
                differentiator=str(
                    getattr(gap, "differentiator", "") or ""
                ),
                already_known_boundary=[
                    str(x)
                    for x in getattr(
                        gap,
                        "already_known_boundary",
                        [],
                    )
                ],
                unresolved_boundary=[
                    str(x)
                    for x in getattr(
                        gap,
                        "unresolved_boundary",
                        [],
                    )
                ],
                targeted_query_count=len(
                    list(
                        getattr(
                            gap,
                            "targeted_queries",
                            [],
                        )
                        or []
                    )
                ),
                evidence_allocation=allocation,
                operator_allocation=operator,
                reason_codes=[
                    str(x)
                    for x in getattr(
                        gap,
                        "reason_codes",
                        [],
                    )
                ],
            )
        )

    body = {
        "schema_version": "novelty-guided-negative-space-plan-v1",
        "source_context_id": str(getattr(context, "context_id")),
        "source_portfolio_id": str(getattr(portfolio, "portfolio_id")),
        "source_gap_plan_id": str(getattr(gap_plan, "plan_id")),
        "source_evidence_diversity_report_id":
            str(getattr(evidence_diversity, "report_id")),
        "targets": [
            row.model_dump(mode="json")
            for row in targets
        ],
        "target_count": len(targets),
        "regeneration_eligible_count": sum(
            row.eligible_for_guided_regeneration
            for row in targets
        ),
        "operator_target_count": sum(
            bool(row.operator_allocation.operators)
            for row in targets
        ),
        "operator_opportunity_count": sum(
            len(row.operator_allocation.operators)
            for row in targets
        ),
        "evidence_alternative_capacity_target_count": sum(
            row.evidence_allocation.alternative_capacity_present
            for row in targets
        ),
        "diagnostic_only": True,
        "generation_authority_created": False,
        "novelty_authority_created": False,
        "production_selection_authority": False,
    }

    plan_id = _stable_id(
        "novelty_guided_negative_space_plan",
        body["source_context_id"],
        body["source_portfolio_id"],
        body["source_gap_plan_id"],
        *[
            f"{row.gap_id}:{row.closure_mode}"
            for row in targets
        ],
    )

    return NoveltyGuidedNegativeSpacePlan(
        **body,
        plan_id=plan_id,
        plan_sha256=_sha256(
            {
                **body,
                "plan_id": plan_id,
            }
        ),
    )


def bind_novelty_guided_regeneration_closure(
    *,
    negative_space_plan: NoveltyGuidedNegativeSpacePlan,
    refinement_report: Any,
) -> NoveltyGuidedDiscoveryClosureReport:
    attempts = list(
        getattr(refinement_report, "attempts", []) or []
    )

    attempts_by_gap: dict[str, list[Any]] = {}
    for attempt in attempts:
        attempts_by_gap.setdefault(
            str(getattr(attempt, "gap_id", "")),
            [],
        ).append(attempt)

    traces: list[NoveltyGuidedRegenerationTrace] = []

    for target in negative_space_plan.targets:
        candidates = attempts_by_gap.get(
            target.gap_id,
            [],
        )

        matching = [
            row
            for row in candidates
            if str(
                getattr(
                    row,
                    "original_hypothesis_id",
                    "",
                )
            )
            == target.hypothesis_id
        ]

        if len(matching) > 1:
            raise ValueError(
                "multiple Alpha6 attempts bind to one negative-space "
                f"target: {target.gap_id}"
            )

        if not matching:
            traces.append(
                NoveltyGuidedRegenerationTrace(
                    gap_id=target.gap_id,
                    hypothesis_id=target.hypothesis_id,
                )
            )
            continue

        attempt = matching[0]

        decision = str(
            getattr(attempt, "decision", "") or ""
        )
        generation_mode = str(
            getattr(attempt, "generation_mode", "") or ""
        )

        refinement_generated = bool(
            getattr(
                attempt,
                "refinement_generated",
                False,
            )
        )

        final_external_status = getattr(
            attempt,
            "final_external_status",
            None,
        )

        fresh_external = (
            refinement_generated
            and final_external_status is not None
        )

        accepted = decision in {
            "accepted_refinement",
            "accepted_reaxis",
        }

        traces.append(
            NoveltyGuidedRegenerationTrace(
                gap_id=target.gap_id,
                hypothesis_id=target.hypothesis_id,
                attempt_observed=True,
                decision=decision or None,
                generation_mode=generation_mode or None,
                candidate_hypothesis_id=getattr(
                    attempt,
                    "candidate_hypothesis_id",
                    None,
                ),
                final_hypothesis_id=getattr(
                    attempt,
                    "final_hypothesis_id",
                    None,
                ),
                refinement_generated=refinement_generated,
                context_grounding_valid=bool(
                    getattr(
                        attempt,
                        "context_grounding_valid",
                        False,
                    )
                ),
                grounding_preserved=bool(
                    getattr(
                        attempt,
                        "grounding_preserved",
                        False,
                    )
                ),
                targeted_external_status=(
                    None
                    if getattr(
                        attempt,
                        "targeted_external_status",
                        None,
                    ) is None
                    else str(
                        attempt.targeted_external_status
                    )
                ),
                final_external_status=(
                    None
                    if final_external_status is None
                    else str(final_external_status)
                ),
                fresh_external_verification_observed=fresh_external,
                accepted_regeneration=accepted,
                closed_loop_observed=(
                    refinement_generated
                    and fresh_external
                ),
                post_generation_semantic_stable=getattr(
                    attempt,
                    "post_generation_semantic_stable",
                    None,
                ),
                post_generation_scientific_action=getattr(
                    attempt,
                    "post_generation_scientific_action",
                    None,
                ),
                post_generation_selection_class=getattr(
                    attempt,
                    "post_generation_selection_class",
                    None,
                ),
                reason_codes=[
                    str(x)
                    for x in getattr(
                        attempt,
                        "reason_codes",
                        [],
                    )
                ],
            )
        )

    action_counts = Counter(
        row.source_gap_action
        for row in negative_space_plan.targets
    )
    mode_counts = Counter(
        row.generation_mode or "none"
        for row in traces
    )
    decision_counts = Counter(
        row.decision or "NO_ATTEMPT"
        for row in traces
    )

    body = {
        "schema_version":
            "novelty-guided-discovery-closure-report-v1",
        "source_negative_space_plan_id":
            negative_space_plan.plan_id,
        "source_refinement_report_id":
            str(getattr(refinement_report, "report_id")),
        "source_final_portfolio_id":
            str(
                getattr(
                    refinement_report,
                    "final_portfolio_id",
                )
            ),
        "traces": [
            row.model_dump(mode="json")
            for row in traces
        ],
        "target_count": len(traces),
        "regeneration_attempted_count":
            sum(row.attempt_observed for row in traces),
        "regeneration_generated_count":
            sum(row.refinement_generated for row in traces),
        "accepted_regeneration_count":
            sum(row.accepted_regeneration for row in traces),
        "fresh_external_verification_count":
            sum(
                row.fresh_external_verification_observed
                for row in traces
            ),
        "closed_loop_observed_count":
            sum(row.closed_loop_observed for row in traces),
        "gap_action_counts":
            dict(sorted(action_counts.items())),
        "generation_mode_counts":
            dict(sorted(mode_counts.items())),
        "decision_counts":
            dict(sorted(decision_counts.items())),
        "conceptual_knownness_required_for_closure": False,
        "external_prior_art_as_positive_premise": False,
        "ranking_computed": False,
        "candidate_survival_authority_created": False,
        "novelty_authority_created": False,
        "production_selection_changed": False,
    }

    report_id = _stable_id(
        "novelty_guided_discovery_closure",
        negative_space_plan.plan_id,
        body["source_refinement_report_id"],
        body["source_final_portfolio_id"],
        *[
            (
                f"{row.gap_id}:"
                f"{row.generation_mode}:"
                f"{row.decision}:"
                f"{row.final_external_status}"
            )
            for row in traces
        ],
    )

    return NoveltyGuidedDiscoveryClosureReport(
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
    "NegativeSpaceEvidenceAllocation",
    "NegativeSpaceOperatorAllocation",
    "NoveltyGuidedNegativeSpaceTarget",
    "NoveltyGuidedNegativeSpacePlan",
    "NoveltyGuidedRegenerationTrace",
    "NoveltyGuidedDiscoveryClosureReport",
    "build_novelty_guided_negative_space_plan",
    "bind_novelty_guided_regeneration_closure",
]
