from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FusionState = Literal[
    "HIGHER_ORDER_GAP_DEPTH_ONLY",
    "HIGHER_ORDER_GAP_CONCEPTUALLY_BOUNDED",
    "SHALLOW_EXTENSION_DEPTH_ONLY",
    "SHALLOW_EXACT_GAP_CONFIRMED",
    "DEPTH_CONCEPTUAL_MISMATCH",
    "WEAK_BRIDGE_UNRESOLVED",
    "KNOWN_RELATION",
    "COVERAGE_UNRESOLVED",
    "CONFLICTED",
]

ConceptualSignal = Literal[
    "NOT_AVAILABLE",
    "INSUFFICIENT_COVERAGE",
    "L3_EXACT_GAP",
    "L1_L2_BROADER_GAP",
    "NO_GAP_OBSERVED",
    "UNCLASSIFIED",
]

RefinementAction = Literal[
    "KEEP_TESTABLE_GAP",
    "STRUCTURAL_SHARPEN",
    "EVIDENCE_REAXIS",
    "RETRIEVE_MECHANISM_SUPPORT",
    "HOLD_UNRESOLVED",
    "CONFLICT_REVIEW",
    "ABSTAIN",
]

ReasoningOperator = Literal[
    "MODERATOR",
    "INTERACTION",
    "RESIDUAL",
    "BOUNDARY",
    "PROXY_DECOUPLING",
    "COMPENSATION_LIMIT",
]


class KnownnessDepthFusionRow(StrictModel):
    hypothesis_id: str
    title: str

    source_external_status: str
    novelty_depth_class: str
    edge_graph_advisory: str

    known_backbone_claim_ids: list[str] = Field(default_factory=list)
    novelty_bearing_claim_ids: list[str] = Field(default_factory=list)
    weak_bridge_claim_ids: list[str] = Field(default_factory=list)

    conceptual_signal: ConceptualSignal
    conceptual_first_gap_level: str | None = None
    conceptual_coverage_sufficient: bool | None = None

    fusion_state: FusionState
    fusion_confidence_basis: Literal[
        "EDGE_GRAPH_ONLY",
        "EDGE_GRAPH_PLUS_CONCEPTUAL",
        "EDGE_GRAPH_PLUS_INSUFFICIENT_CONCEPTUAL",
    ]

    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str

    foundational_knownness_checked: Literal[False] = False
    conceptual_knownness_is_search_bounded: Literal[True] = True
    novelty_authority_created: Literal[False] = False


class KnownnessDepthFusionReport(StrictModel):
    schema_version: Literal[
        "knownness-depth-fusion-report-v1"
    ] = "knownness-depth-fusion-report-v1"

    report_id: str
    report_sha256: str

    source_edge_graph_report_id: str
    source_portfolio_id: str
    source_external_report_id: str

    rows: list[KnownnessDepthFusionRow] = Field(default_factory=list)

    hypothesis_count: int = 0
    fusion_state_counts: dict[str, int] = Field(default_factory=dict)
    conceptual_signal_counts: dict[str, int] = Field(default_factory=dict)
    conceptual_available_count: int = 0
    conceptual_sufficient_count: int = 0

    foundational_knownness_checked: Literal[False] = False
    conceptual_knownness_used_as_diagnostic: Literal[True] = True
    conceptual_knownness_has_selection_authority: Literal[False] = False
    ranking_computed: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    production_selection_changed: Literal[False] = False


class ActionableRefinementTarget(StrictModel):
    hypothesis_id: str
    title: str

    fusion_state: FusionState
    novelty_depth_class: str
    conceptual_signal: ConceptualSignal

    primary_action: RefinementAction

    operator_candidates: list[ReasoningOperator] = Field(default_factory=list)
    preferred_operator: ReasoningOperator | None = None
    operator_basis: Literal[
        "NONE",
        "HIGHER_ORDER_GAP_STRUCTURE",
        "SHALLOW_LOCAL_EXTENSION_STRUCTURE",
    ] = "NONE"

    novelty_bearing_claim_ids: list[str] = Field(default_factory=list)
    known_backbone_claim_ids: list[str] = Field(default_factory=list)
    weak_bridge_claim_ids: list[str] = Field(default_factory=list)

    safe_unused_evidence_statement_ids: list[str] = Field(default_factory=list)
    evidence_reaxis_capacity_present: bool = False

    requires_additional_retrieval: bool = False
    generation_candidate: bool = False

    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str

    operator_is_reasoning_template_not_evidence: Literal[True] = True
    external_prior_art_as_positive_premise: Literal[False] = False
    unused_evidence_relevance_inferred: Literal[False] = False
    generation_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ActionableRefinementPlan(StrictModel):
    schema_version: Literal[
        "actionable-refinement-plan-v1"
    ] = "actionable-refinement-plan-v1"

    plan_id: str
    plan_sha256: str

    source_fusion_report_id: str
    source_edge_graph_report_id: str
    source_gap_plan_id: str
    source_context_id: str
    source_portfolio_id: str

    targets: list[ActionableRefinementTarget] = Field(default_factory=list)

    target_count: int = 0
    action_counts: dict[str, int] = Field(default_factory=dict)
    structural_sharpen_count: int = 0
    evidence_reaxis_count: int = 0
    retrieval_support_count: int = 0
    keep_testable_gap_count: int = 0
    abstain_or_hold_count: int = 0

    operator_candidate_counts: dict[str, int] = Field(default_factory=dict)
    preferred_operator_counts: dict[str, int] = Field(default_factory=dict)

    planner_only: Literal[True] = True
    alpha6_runtime_behavior_changed: Literal[False] = False
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


def _conceptual_signal(row: Any) -> ConceptualSignal:
    overlay = getattr(row, "conceptual_knownness", None)

    if overlay is None or not bool(
        getattr(overlay, "available", False)
    ):
        return "NOT_AVAILABLE"

    sufficient = getattr(
        overlay,
        "coverage_sufficient",
        None,
    )
    first_gap = getattr(
        overlay,
        "first_gap_level",
        None,
    )
    disposition = getattr(
        overlay,
        "disposition",
        None,
    )

    if sufficient is not True:
        return "INSUFFICIENT_COVERAGE"

    if disposition == "NO_GAP_OBSERVED":
        return "NO_GAP_OBSERVED"

    if str(first_gap) == "L3_EXACT":
        return "L3_EXACT_GAP"

    if str(first_gap) in {
        "L1_BROAD",
        "L2_INTERMEDIATE",
    }:
        return "L1_L2_BROADER_GAP"

    if first_gap is None:
        return "NO_GAP_OBSERVED"

    return "UNCLASSIFIED"


def _fuse_row(row: Any) -> KnownnessDepthFusionRow:
    depth = str(row.novelty_depth_class)
    signal = _conceptual_signal(row)
    first_gap = getattr(
        getattr(row, "conceptual_knownness", None),
        "first_gap_level",
        None,
    )
    sufficient = getattr(
        getattr(row, "conceptual_knownness", None),
        "coverage_sufficient",
        None,
    )

    reasons: list[str] = []

    if signal == "NOT_AVAILABLE":
        basis = "EDGE_GRAPH_ONLY"
    elif signal == "INSUFFICIENT_COVERAGE":
        basis = "EDGE_GRAPH_PLUS_INSUFFICIENT_CONCEPTUAL"
        reasons.append(
            "conceptual_knownness_present_but_coverage_insufficient"
        )
    else:
        basis = "EDGE_GRAPH_PLUS_CONCEPTUAL"
        reasons.append(
            "conceptual_knownness_coverage_sufficient"
        )

    if depth == "CONFLICTED_DEPTH":
        state: FusionState = "CONFLICTED"
        interpretation = (
            "The relation graph contains conflicting core prior art; "
            "refinement should not proceed as a novelty optimization."
        )
    elif depth == "UNRESOLVED_DEPTH":
        state = "COVERAGE_UNRESOLVED"
        interpretation = (
            "External or conceptual coverage is unresolved; more evidence "
            "is required before choosing a structural novelty action."
        )
    elif depth == "KNOWN_RELATION":
        state = "KNOWN_RELATION"
        interpretation = (
            "The core relation is already known in the bounded evidence; "
            "novelty-oriented generation should abstain."
        )
    elif depth == "UNSUPPORTED_BRIDGE_DEPTH_UNKNOWN":
        state = "WEAK_BRIDGE_UNRESOLVED"
        reasons.append(
            "weak_novelty_bearing_bridge_not_promoted_by_knownness"
        )
        interpretation = (
            "The novelty-bearing bridge remains weakly grounded. "
            "Conceptual absence cannot convert missing mechanistic support "
            "into deep novelty."
        )
    elif depth == "HIGHER_ORDER_INTERACTION_GAP":
        if signal == "L3_EXACT_GAP":
            state = "HIGHER_ORDER_GAP_CONCEPTUALLY_BOUNDED"
            reasons.append(
                "broad_intermediate_known_exact_gap"
            )
            interpretation = (
                "Known lower-order backbone plus sufficient conceptual "
                "coverage localizes the remaining gap at the exact "
                "higher-order relation."
            )
        elif signal in {
            "L1_L2_BROADER_GAP",
            "NO_GAP_OBSERVED",
        }:
            state = "DEPTH_CONCEPTUAL_MISMATCH"
            reasons.append(
                "edge_depth_and_conceptual_gap_level_disagree"
            )
            interpretation = (
                "Edge-graph depth and conceptual-knownness depth do not "
                "agree. The discrepancy is retained as unresolved rather "
                "than forcing either signal to dominate."
            )
        else:
            state = "HIGHER_ORDER_GAP_DEPTH_ONLY"
            interpretation = (
                "The relation graph supports a higher-order gap over a "
                "known lower-order backbone, but conceptual knownness is "
                "not sufficiently available to further localize depth."
            )
    elif depth == "SHALLOW_LOCAL_EXTENSION":
        if signal == "L3_EXACT_GAP":
            state = "SHALLOW_EXACT_GAP_CONFIRMED"
            reasons.append(
                "shallow_edge_depth_agrees_with_exact_conceptual_gap"
            )
            interpretation = (
                "The local extension is consistent with a search-bounded "
                "exact-level gap over broader known structure."
            )
        elif signal == "L1_L2_BROADER_GAP":
            state = "DEPTH_CONCEPTUAL_MISMATCH"
            reasons.append(
                "shallow_edge_depth_but_broader_conceptual_gap"
            )
            interpretation = (
                "The relation graph suggests a shallow extension while "
                "conceptual knownness indicates an earlier broader gap. "
                "The mismatch is held for further evidence."
            )
        elif signal == "NO_GAP_OBSERVED":
            state = "DEPTH_CONCEPTUAL_MISMATCH"
            reasons.append(
                "shallow_edge_depth_but_no_conceptual_gap"
            )
            interpretation = (
                "The relation graph suggests a local gap but sufficiently "
                "covered conceptual knownness reports no gap. "
                "Automatic sharpening is withheld."
            )
        else:
            state = "SHALLOW_EXTENSION_DEPTH_ONLY"
            interpretation = (
                "The relation graph localizes novelty to an extension of a "
                "known backbone. Conceptual knownness is unavailable or "
                "insufficient, so the classification remains depth-only."
            )
    else:
        state = "COVERAGE_UNRESOLVED"
        reasons.append("unrecognized_depth_class_fail_closed")
        interpretation = (
            "The novelty depth class is not recognized by the fusion "
            "policy, so the row is held unresolved."
        )

    return KnownnessDepthFusionRow(
        hypothesis_id=str(row.hypothesis_id),
        title=str(row.title),
        source_external_status=str(row.source_external_status),
        novelty_depth_class=depth,
        edge_graph_advisory=str(row.planner_advisory),
        known_backbone_claim_ids=list(
            getattr(row, "known_backbone_claim_ids", [])
        ),
        novelty_bearing_claim_ids=list(
            getattr(row, "novelty_bearing_claim_ids", [])
        ),
        weak_bridge_claim_ids=list(
            getattr(row, "weak_bridge_claim_ids", [])
        ),
        conceptual_signal=signal,
        conceptual_first_gap_level=(
            None if first_gap is None else str(first_gap)
        ),
        conceptual_coverage_sufficient=sufficient,
        fusion_state=state,
        fusion_confidence_basis=basis,
        reason_codes=sorted(set(reasons)),
        interpretation=interpretation,
    )


def build_knownness_depth_fusion(
    *,
    edge_graph_report: Any,
) -> KnownnessDepthFusionReport:
    rows = [
        _fuse_row(row)
        for row in getattr(
            edge_graph_report,
            "profiles",
            [],
        )
    ]

    fusion_counts = Counter(
        row.fusion_state
        for row in rows
    )
    conceptual_counts = Counter(
        row.conceptual_signal
        for row in rows
    )

    body = {
        "schema_version": "knownness-depth-fusion-report-v1",
        "source_edge_graph_report_id": str(
            edge_graph_report.report_id
        ),
        "source_portfolio_id": str(
            edge_graph_report.source_portfolio_id
        ),
        "source_external_report_id": str(
            edge_graph_report.source_external_report_id
        ),
        "rows": [
            row.model_dump(mode="json")
            for row in rows
        ],
        "hypothesis_count": len(rows),
        "fusion_state_counts": dict(
            sorted(fusion_counts.items())
        ),
        "conceptual_signal_counts": dict(
            sorted(conceptual_counts.items())
        ),
        "conceptual_available_count": sum(
            row.conceptual_signal != "NOT_AVAILABLE"
            for row in rows
        ),
        "conceptual_sufficient_count": sum(
            row.conceptual_coverage_sufficient is True
            for row in rows
        ),
        "foundational_knownness_checked": False,
        "conceptual_knownness_used_as_diagnostic": True,
        "conceptual_knownness_has_selection_authority": False,
        "ranking_computed": False,
        "novelty_authority_created": False,
        "production_selection_changed": False,
    }

    report_id = _stable_id(
        "knownness_depth_fusion",
        body["source_edge_graph_report_id"],
        *[
            (
                f"{row.hypothesis_id}:"
                f"{row.fusion_state}:"
                f"{row.conceptual_signal}"
            )
            for row in rows
        ],
    )

    return KnownnessDepthFusionReport(
        **body,
        report_id=report_id,
        report_sha256=_sha256(
            {
                **body,
                "report_id": report_id,
            }
        ),
    )


def _safe_unused_statement_ids(
    *,
    context: Any,
    portfolio: Any,
) -> list[str]:
    used = {
        str(statement_id)
        for card in getattr(
            portfolio,
            "hypotheses",
            [],
        )
        for statement_id in getattr(
            card,
            "premise_statement_ids",
            [],
        )
    }

    result: list[str] = []

    for row in getattr(
        context,
        "evidence_statements",
        [],
    ):
        sid = str(
            getattr(
                row,
                "statement_id",
                "",
            )
            or ""
        )

        if not sid or sid in used:
            continue

        if not bool(
            getattr(
                row,
                "eligible_as_premise",
                False,
            )
        ):
            continue

        if list(
            getattr(
                row,
                "premise_restrictions",
                [],
            )
            or []
        ):
            continue

        result.append(sid)

    return sorted(set(result))


def _action_for_row(
    *,
    fusion: KnownnessDepthFusionRow,
    safe_unused: list[str],
) -> ActionableRefinementTarget:
    state = fusion.fusion_state
    reasons: list[str] = []

    action: RefinementAction
    operators: list[ReasoningOperator] = []
    preferred: ReasoningOperator | None = None
    basis = "NONE"
    retrieval = False
    generation_candidate = False

    if state in {
        "HIGHER_ORDER_GAP_DEPTH_ONLY",
        "HIGHER_ORDER_GAP_CONCEPTUALLY_BOUNDED",
    }:
        action = "KEEP_TESTABLE_GAP"
        reasons.append(
            "higher_order_gap_over_known_backbone_is_already_testable"
        )
        interpretation = (
            "Keep the bounded higher-order gap. No novelty inflation by "
            "adding another condition is requested."
        )
    elif state in {
        "SHALLOW_EXTENSION_DEPTH_ONLY",
        "SHALLOW_EXACT_GAP_CONFIRMED",
    }:
        action = "STRUCTURAL_SHARPEN"
        operators = [
            "RESIDUAL",
            "BOUNDARY",
        ]
        preferred = "RESIDUAL"
        basis = "SHALLOW_LOCAL_EXTENSION_STRUCTURE"
        generation_candidate = True
        reasons.extend(
            [
                "shallow_local_extension_should_change_relation_topology",
                "residual_preferred_to_additional_modifier",
            ]
        )
        interpretation = (
            "The hypothesis is a shallow extension of a known backbone. "
            "Plan a same-premise structural sharpen, preferring a residual "
            "question after accounting for the known relation; a boundary "
            "form is the bounded alternative. Operators are reasoning "
            "templates, not evidence."
        )
    elif state == "WEAK_BRIDGE_UNRESOLVED":
        if safe_unused:
            action = "EVIDENCE_REAXIS"
            generation_candidate = True
            reasons.append(
                "weak_bridge_with_safe_unused_grounded_evidence_capacity"
            )
            interpretation = (
                "The novelty-bearing bridge is weak. A fresh grounded "
                "evidence neighborhood exists, so re-axis is preferable "
                "to strengthening the unsupported bridge."
            )
        else:
            action = "RETRIEVE_MECHANISM_SUPPORT"
            retrieval = True
            reasons.append(
                "weak_bridge_without_safe_reaxis_capacity"
            )
            interpretation = (
                "The novelty-bearing bridge is weak and no safe unused "
                "grounded premise is available. Retrieve mechanism support "
                "or abstain rather than generate a stronger unsupported "
                "relation."
            )
    elif state == "KNOWN_RELATION":
        action = "ABSTAIN"
        reasons.append("known_relation_not_novelty_generation_target")
        interpretation = (
            "The relation is already known in the bounded evidence; "
            "novelty-oriented refinement abstains."
        )
    elif state == "COVERAGE_UNRESOLVED":
        action = "HOLD_UNRESOLVED"
        retrieval = True
        reasons.append("coverage_unresolved")
        interpretation = (
            "Coverage is unresolved. Hold the hypothesis for evidence "
            "resolution rather than optimize novelty."
        )
    elif state == "CONFLICTED":
        action = "CONFLICT_REVIEW"
        retrieval = True
        reasons.append("conflicting_prior_art_requires_review")
        interpretation = (
            "Conflicting prior art requires bounded conflict review before "
            "any novelty-oriented rewrite."
        )
    else:
        action = "HOLD_UNRESOLVED"
        retrieval = True
        reasons.append(
            "knownness_depth_mismatch_requires_resolution"
        )
        interpretation = (
            "Knownness and edge-depth signals disagree. Hold for additional "
            "coverage rather than selecting a refinement topology."
        )

    return ActionableRefinementTarget(
        hypothesis_id=fusion.hypothesis_id,
        title=fusion.title,
        fusion_state=fusion.fusion_state,
        novelty_depth_class=fusion.novelty_depth_class,
        conceptual_signal=fusion.conceptual_signal,
        primary_action=action,
        operator_candidates=operators,
        preferred_operator=preferred,
        operator_basis=basis,
        novelty_bearing_claim_ids=list(
            fusion.novelty_bearing_claim_ids
        ),
        known_backbone_claim_ids=list(
            fusion.known_backbone_claim_ids
        ),
        weak_bridge_claim_ids=list(
            fusion.weak_bridge_claim_ids
        ),
        safe_unused_evidence_statement_ids=list(
            safe_unused
        ),
        evidence_reaxis_capacity_present=bool(
            safe_unused
        ),
        requires_additional_retrieval=retrieval,
        generation_candidate=generation_candidate,
        reason_codes=sorted(set(reasons)),
        interpretation=interpretation,
    )


def build_actionable_refinement_plan(
    *,
    context: Any,
    portfolio: Any,
    gap_plan: Any,
    fusion_report: KnownnessDepthFusionReport,
    edge_graph_report: Any,
) -> ActionableRefinementPlan:
    del edge_graph_report  # lineage is retained by fusion_report.

    safe_unused = _safe_unused_statement_ids(
        context=context,
        portfolio=portfolio,
    )

    gap_hypothesis_ids = {
        str(row.hypothesis_id)
        for row in getattr(
            gap_plan,
            "gaps",
            [],
        )
    }

    targets = [
        _action_for_row(
            fusion=row,
            safe_unused=safe_unused,
        )
        for row in fusion_report.rows
        if row.hypothesis_id in gap_hypothesis_ids
    ]

    action_counts = Counter(
        row.primary_action
        for row in targets
    )
    operator_counts = Counter(
        operator
        for row in targets
        for operator in row.operator_candidates
    )
    preferred_counts = Counter(
        row.preferred_operator
        for row in targets
        if row.preferred_operator is not None
    )

    body = {
        "schema_version": "actionable-refinement-plan-v1",
        "source_fusion_report_id": fusion_report.report_id,
        "source_edge_graph_report_id":
            fusion_report.source_edge_graph_report_id,
        "source_gap_plan_id": str(gap_plan.plan_id),
        "source_context_id": str(context.context_id),
        "source_portfolio_id": str(portfolio.portfolio_id),
        "targets": [
            row.model_dump(mode="json")
            for row in targets
        ],
        "target_count": len(targets),
        "action_counts": dict(
            sorted(action_counts.items())
        ),
        "structural_sharpen_count": action_counts.get(
            "STRUCTURAL_SHARPEN",
            0,
        ),
        "evidence_reaxis_count": action_counts.get(
            "EVIDENCE_REAXIS",
            0,
        ),
        "retrieval_support_count": action_counts.get(
            "RETRIEVE_MECHANISM_SUPPORT",
            0,
        ),
        "keep_testable_gap_count": action_counts.get(
            "KEEP_TESTABLE_GAP",
            0,
        ),
        "abstain_or_hold_count": (
            action_counts.get("ABSTAIN", 0)
            + action_counts.get("HOLD_UNRESOLVED", 0)
            + action_counts.get("CONFLICT_REVIEW", 0)
        ),
        "operator_candidate_counts": dict(
            sorted(operator_counts.items())
        ),
        "preferred_operator_counts": dict(
            sorted(preferred_counts.items())
        ),
        "planner_only": True,
        "alpha6_runtime_behavior_changed": False,
        "novelty_authority_created": False,
        "production_selection_changed": False,
    }

    plan_id = _stable_id(
        "actionable_refinement_plan",
        body["source_fusion_report_id"],
        body["source_gap_plan_id"],
        *[
            (
                f"{row.hypothesis_id}:"
                f"{row.primary_action}:"
                f"{row.preferred_operator}"
            )
            for row in targets
        ],
    )

    return ActionableRefinementPlan(
        **body,
        plan_id=plan_id,
        plan_sha256=_sha256(
            {
                **body,
                "plan_id": plan_id,
            }
        ),
    )


__all__ = [
    "KnownnessDepthFusionRow",
    "KnownnessDepthFusionReport",
    "ActionableRefinementTarget",
    "ActionableRefinementPlan",
    "build_knownness_depth_fusion",
    "build_actionable_refinement_plan",
]
