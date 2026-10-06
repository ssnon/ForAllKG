from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.adaptive_discovery_controller import external_boundaries, safe_unused_premise_ids
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExploratoryAction = Literal[
    "KEEP_ELITE",
    "SAME_PREMISE_SHARPEN",
    "EVIDENCE_REAXIS",
    "AXIS_MUTATION",
    "REQUEST_GRAPH_RETRAVERSAL",
]
ExploratorySlot = Literal["EXPLOIT", "PROSPECTIVE", "UNCERTAINTY", "WILDCARD"]


class ExploratoryParentState(StrictModel):
    hypothesis_id: str
    prospective_identifiability: str
    current_epistemic_state: str
    external_status: str | None = None
    safe_unused_premise_statement_ids: list[str] = Field(default_factory=list)
    reproductive_eligible: bool
    reproductive_block_reason: str | None = None
    quality_prior: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    underexplored_bonus: float = Field(ge=0.0)
    verification_cost_prior: float = Field(ge=0.0, le=1.0)
    search_priority: float
    graph_retraversal_priority: float = Field(ge=0.0)
    branch_actions: list[ExploratoryAction] = Field(default_factory=list)


class ExploratoryPortfolioPlan(StrictModel):
    schema_version: Literal["exploratory-portfolio-search-plan-v1"] = (
        "exploratory-portfolio-search-plan-v1"
    )
    plan_id: str
    source_portfolio_id: str
    source_context_id: str
    parent_states: list[ExploratoryParentState] = Field(default_factory=list)
    max_local_branches_per_parent: int = Field(ge=1)
    max_graph_retraversal_allocations: int = Field(ge=0)
    exploration_temperature: float = Field(ge=0.0)

    parallel_branching_authority: Literal[True] = True
    search_parent_selection_authority: Literal[True] = True
    compute_allocation_authority: Literal[True] = True
    graph_retraversal_allocation_authority: Literal[True] = True
    not_operationalizable_reproductive_block_authority: Literal[True] = True

    scientific_truth_authority: Literal[False] = False
    literature_wide_novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class ExploratoryCandidateRecord(StrictModel):
    hypothesis_id: str
    source_hypothesis_id: str
    route: ExploratoryAction
    prospective_identifiability: str
    current_epistemic_state: str
    external_status: str | None = None
    premise_statement_ids: list[str] = Field(default_factory=list)
    hypothesis_type: str = ""
    title: str = ""
    hypothesis_statement: str = ""
    family_signature: str
    reproductive_eligible: bool = True
    quality_prior: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    novelty_of_search_path: float = Field(ge=0.0, le=1.0)
    verification_cost_prior: float = Field(ge=0.0, le=1.0)
    ucb_exploration_bonus: float = Field(ge=0.0)
    base_search_value: float
    selected_slot: ExploratorySlot | None = None
    selected_for_next_verification: bool = False


class ExploratoryPortfolioSelection(StrictModel):
    schema_version: Literal["exploratory-portfolio-search-selection-v1"] = (
        "exploratory-portfolio-search-selection-v1"
    )
    selection_id: str
    source_plan_id: str
    candidate_count: int = Field(ge=0)
    retained_count: int = Field(ge=0)
    retained_candidate_ids: list[str] = Field(default_factory=list)
    retained_count_by_slot: dict[str, int] = Field(default_factory=dict)
    records: list[ExploratoryCandidateRecord] = Field(default_factory=list)
    selection_policy: Literal["SEARCH_ONLY_QUOTA_DIVERSITY_UCB_V1"] = (
        "SEARCH_ONLY_QUOTA_DIVERSITY_UCB_V1"
    )
    search_parent_selection_authority: Literal[True] = True
    compute_allocation_authority: Literal[True] = True
    candidate_archival_preserved: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    literature_wide_novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


_STATE_Q = {
    "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW": 1.00,
    "UNRESOLVED_TOPOLOGY_GAP": 0.62,
    "UNRESOLVED_EVIDENCE_GAP": 0.58,
    "PRIOR_ART_BACKED_OR_NO_RESIDUAL": 0.35,
}
_EXT_Q = {
    "PLAUSIBLY_NOVEL": 1.00,
    "NEW_COMBINATION_OF_KNOWN_EFFECTS": 0.92,
    "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP": 0.86,
    "LITERATURE_SUPPORTED_EXTENSION": 0.58,
    "INSUFFICIENT_SEARCH_EVIDENCE": 0.52,
    "WELL_ESTABLISHED": 0.20,
    "CONFLICTING_PRIOR_ART": 0.18,
}
_PRO_Q = {
    "CURRENTLY_IDENTIFIED": 1.00,
    "PROSPECTIVELY_IDENTIFIABLE": 0.82,
    "NOT_OPERATIONALIZABLE": 0.00,
    "UNKNOWN": 0.50,
}
_ROUTE_NOVELTY = {
    "KEEP_ELITE": 0.00,
    "SAME_PREMISE_SHARPEN": 0.15,
    "EVIDENCE_REAXIS": 0.45,
    "AXIS_MUTATION": 0.80,
    "REQUEST_GRAPH_RETRAVERSAL": 1.00,
}
_ROUTE_COST = {
    "KEEP_ELITE": 0.05,
    "SAME_PREMISE_SHARPEN": 0.22,
    "EVIDENCE_REAXIS": 0.35,
    "AXIS_MUTATION": 0.52,
    "REQUEST_GRAPH_RETRAVERSAL": 0.90,
}


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _norm(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9α-ω가-힣]+", " ", str(value or "").casefold()).split())


def _tokens(value: object) -> set[str]:
    return set(_norm(value).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _quality(state: str, external: str | None, prospective: str) -> float:
    return max(0.0, min(1.0, 0.35*_STATE_Q.get(state, .5) + 0.30*_EXT_Q.get(str(external or ""), .5) + 0.35*_PRO_Q.get(prospective, .5)))


def _uncertainty(state: str, external: str | None, prospective: str) -> float:
    value = .15
    if state in {"UNRESOLVED_EVIDENCE_GAP", "UNRESOLVED_TOPOLOGY_GAP"}:
        value += .28
    if external == "INSUFFICIENT_SEARCH_EVIDENCE":
        value += .28
    if prospective == "PROSPECTIVELY_IDENTIFIABLE":
        value += .18
    if prospective == "NOT_OPERATIONALIZABLE":
        value += .35
    if external in {"WELL_ESTABLISHED", "CONFLICTING_PRIOR_ART"}:
        value -= .10
    return max(0.0, min(1.0, value))


def _underexplored(history_count: int, population_size: int) -> float:
    return math.sqrt(max(0.0, math.log(2.0 + max(1, population_size))) / (1.0 + max(0, history_count)))


def _branch_actions(
    prospective: str,
    state: str,
    has_unused: bool,
    has_target_claim_ids: bool,
    cap: int,
) -> list[ExploratoryAction]:
    # SAME_PREMISE_SHARPEN requires an actual external target claim.
    if prospective == "NOT_OPERATIONALIZABLE":
        actions: list[ExploratoryAction] = []
        if has_unused:
            actions.append("EVIDENCE_REAXIS")
        actions.append("AXIS_MUTATION")
        return actions[:cap]

    actions: list[ExploratoryAction] = []
    if has_target_claim_ids:
        actions.append("SAME_PREMISE_SHARPEN")
    if has_unused:
        actions.append("EVIDENCE_REAXIS")
    actions.append("AXIS_MUTATION")

    if state == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW":
        actions = ["AXIS_MUTATION"]
        if has_unused:
            actions.append("EVIDENCE_REAXIS")
        if has_target_claim_ids:
            actions.append("SAME_PREMISE_SHARPEN")

    return actions[:cap]


def build_exploratory_portfolio_plan(
    *,
    context: HypothesisContext,
    portfolio: HypothesisPortfolio,
    epistemic_state_report: Mapping[str, Any],
    external_report: Mapping[str, Any],
    prospective_by_hypothesis: Mapping[str, Mapping[str, Any]],
    history_count_by_hypothesis: Mapping[str, int] | None = None,
    max_local_branches_per_parent: int = 3,
    max_graph_retraversal_allocations: int = 2,
    exploration_temperature: float = 0.9,
) -> ExploratoryPortfolioPlan:
    history_count_by_hypothesis = dict(history_count_by_hypothesis or {})
    state_by_h = {str(x.get("hypothesis_id")): x for x in epistemic_state_report.get("hypotheses", [])}
    ext_by_h = {str(x.get("hypothesis_id")): x for x in external_report.get("cards", [])}
    parents = []
    n = max(1, len(portfolio.hypotheses))
    for card in portfolio.hypotheses:
        hid = str(card.hypothesis_id)
        state = str(state_by_h.get(hid, {}).get("final_epistemic_state") or "UNRESOLVED_EVIDENCE_GAP")
        ext = ext_by_h.get(hid, {})
        _, _, target_claim_ids = external_boundaries(ext)
        external = str(ext.get("status")) if ext.get("status") is not None else None
        prospective = str(prospective_by_hypothesis.get(hid, {}).get("prospective_identifiability") or "UNKNOWN")
        unused = safe_unused_premise_ids(context, card)
        q = _quality(state, external, prospective)
        u = _uncertainty(state, external, prospective)
        bonus = _underexplored(int(history_count_by_hypothesis.get(hid, 0)), n)
        cost = min(1.0, .10 + .06*len(card.premise_statement_ids))
        reproductive = prospective != "NOT_OPERATIONALIZABLE"
        parents.append(ExploratoryParentState(
            hypothesis_id=hid,
            prospective_identifiability=prospective,
            current_epistemic_state=state,
            external_status=external,
            safe_unused_premise_statement_ids=unused,
            reproductive_eligible=reproductive,
            reproductive_block_reason=None if reproductive else "NOT_OPERATIONALIZABLE_PARENT_CANNOT_REPRODUCE_DIRECTLY",
            quality_prior=q,
            uncertainty=u,
            underexplored_bonus=bonus,
            verification_cost_prior=cost,
            search_priority=.45*q + .30*u + .20*bonus - .10*cost,
            graph_retraversal_priority=.55*u + .25*bonus + (.35 if prospective == "NOT_OPERATIONALIZABLE" else 0.0) + (.15 if state == "UNRESOLVED_TOPOLOGY_GAP" else 0.0),
            branch_actions=_branch_actions(prospective, state, bool(unused), bool(target_claim_ids), max_local_branches_per_parent),
        ))
    body = {
        "source_portfolio_id": portfolio.portfolio_id,
        "source_context_id": context.context_id,
        "parent_states": [x.model_dump(mode="json") for x in parents],
        "max_local_branches_per_parent": max_local_branches_per_parent,
        "max_graph_retraversal_allocations": max_graph_retraversal_allocations,
        "exploration_temperature": exploration_temperature,
    }
    return ExploratoryPortfolioPlan(plan_id=stable_id("exploratory_portfolio_search_plan", body), **body)


def allocated_graph_parent_ids(plan: ExploratoryPortfolioPlan) -> list[str]:
    rows = sorted(plan.parent_states, key=lambda x: (-x.graph_retraversal_priority, -x.uncertainty, x.hypothesis_id))
    return [x.hypothesis_id for x in rows[:plan.max_graph_retraversal_allocations] if x.graph_retraversal_priority > 0]


def family_signature(card: Any) -> str:
    return stable_id("exploratory_family", str(getattr(card, "hypothesis_type", "")), sorted(map(str, getattr(card, "premise_statement_ids", []) or [])), sorted(_tokens(getattr(card, "inferential_bridge", "")))[:16])


def candidate_record(*, card: Any, source_hypothesis_id: str, route: ExploratoryAction, prospective_status: str, parent_state: ExploratoryParentState, lineage_visit_count: int, total_candidate_count: int) -> ExploratoryCandidateRecord:
    q = _quality(parent_state.current_epistemic_state, parent_state.external_status, prospective_status)
    u = _uncertainty(parent_state.current_epistemic_state, parent_state.external_status, prospective_status)
    novelty = _ROUTE_NOVELTY[route]
    cost = _ROUTE_COST[route]
    ucb = math.sqrt(max(0.0, math.log(2.0 + max(1, total_candidate_count))) / (1.0 + max(0, lineage_visit_count)))
    value = .40*q + .24*u + .18*novelty + .18*ucb - .12*cost
    return ExploratoryCandidateRecord(
        hypothesis_id=str(card.hypothesis_id),
        source_hypothesis_id=str(source_hypothesis_id),
        route=route,
        prospective_identifiability=prospective_status,
        current_epistemic_state=parent_state.current_epistemic_state,
        external_status=parent_state.external_status,
        premise_statement_ids=list(map(str, card.premise_statement_ids)),
        hypothesis_type=str(card.hypothesis_type),
        title=str(card.title),
        hypothesis_statement=str(card.hypothesis_statement),
        family_signature=family_signature(card),
        reproductive_eligible=prospective_status != "NOT_OPERATIONALIZABLE",
        quality_prior=q,
        uncertainty=u,
        novelty_of_search_path=novelty,
        verification_cost_prior=cost,
        ucb_exploration_bonus=ucb,
        base_search_value=value,
    )


def _diversity(row: ExploratoryCandidateRecord, selected: Sequence[ExploratoryCandidateRecord]) -> float:
    if not selected:
        return 1.0
    p = set(row.premise_statement_ids)
    t = _tokens(row.hypothesis_statement)
    worst = 0.0
    for other in selected:
        worst = max(worst, .55*_jaccard(p, set(other.premise_statement_ids)) + .45*_jaccard(t, _tokens(other.hypothesis_statement)))
    return max(0.0, 1.0 - worst)


def _slot_score(row: ExploratoryCandidateRecord, slot: ExploratorySlot, selected: Sequence[ExploratoryCandidateRecord]) -> float:
    d = _diversity(row, selected)
    if slot == "EXPLOIT":
        return .58*row.quality_prior + .17*d + .15*row.ucb_exploration_bonus - .10*row.verification_cost_prior
    if slot == "PROSPECTIVE":
        pb = 1.0 if row.prospective_identifiability == "PROSPECTIVELY_IDENTIFIABLE" else .35
        return .34*row.quality_prior + .28*pb + .23*d + .15*row.novelty_of_search_path
    if slot == "UNCERTAINTY":
        return .46*row.uncertainty + .24*row.ucb_exploration_bonus + .20*d + .10*row.quality_prior
    return .40*row.novelty_of_search_path + .28*d + .20*row.ucb_exploration_bonus + .12*row.uncertainty


def select_exploratory_portfolio(*, plan: ExploratoryPortfolioPlan, records: Sequence[ExploratoryCandidateRecord], max_retained_candidates: int = 8, slot_quotas: Mapping[str, int] | None = None) -> ExploratoryPortfolioSelection:
    quotas = dict(slot_quotas or {"EXPLOIT": 3, "PROSPECTIVE": 2, "UNCERTAINTY": 2, "WILDCARD": 1})
    order: tuple[ExploratorySlot, ...] = ("EXPLOIT", "PROSPECTIVE", "UNCERTAINTY", "WILDCARD")
    working = [x.model_copy(deep=True) for x in records]
    selected: list[ExploratoryCandidateRecord] = []
    ids: set[str] = set()
    families: set[str] = set()

    def eligible(row: ExploratoryCandidateRecord, slot: ExploratorySlot) -> bool:
        if not row.reproductive_eligible or row.hypothesis_id in ids or row.family_signature in families:
            return False
        if slot == "PROSPECTIVE":
            return row.prospective_identifiability == "PROSPECTIVELY_IDENTIFIABLE"
        if slot == "WILDCARD":
            return row.route in {"AXIS_MUTATION", "EVIDENCE_REAXIS"}
        return True

    def choose(slot: ExploratorySlot) -> bool:
        pool = [x for x in working if eligible(x, slot)]
        if not pool:
            return False
        pool.sort(key=lambda x: (-_slot_score(x, slot, selected), -x.base_search_value, x.hypothesis_id))
        picked = pool[0].model_copy(update={"selected_slot": slot, "selected_for_next_verification": True})
        for i, row in enumerate(working):
            if row.hypothesis_id == picked.hypothesis_id:
                working[i] = picked
                break
        selected.append(picked)
        ids.add(picked.hypothesis_id)
        families.add(picked.family_signature)
        return True

    for slot in order:
        for _ in range(max(0, int(quotas.get(slot, 0)))):
            if len(selected) >= max_retained_candidates or not choose(slot):
                break

    while len(selected) < max_retained_candidates:
        pool = [x for x in working if x.reproductive_eligible and x.hypothesis_id not in ids and x.family_signature not in families]
        if not pool:
            break
        pool.sort(key=lambda x: (-(x.base_search_value + .22*_diversity(x, selected)), x.hypothesis_id))
        picked = pool[0].model_copy(update={"selected_slot": "UNCERTAINTY", "selected_for_next_verification": True})
        for i, row in enumerate(working):
            if row.hypothesis_id == picked.hypothesis_id:
                working[i] = picked
                break
        selected.append(picked)
        ids.add(picked.hypothesis_id)
        families.add(picked.family_signature)

    counts = Counter(x.selected_slot for x in selected if x.selected_slot is not None)
    body = {
        "source_plan_id": plan.plan_id,
        "candidate_count": len(working),
        "retained_count": len(selected),
        "retained_candidate_ids": [x.hypothesis_id for x in selected],
        "retained_count_by_slot": dict(sorted(counts.items())),
        "records": [x.model_dump(mode="json") for x in working],
    }
    return ExploratoryPortfolioSelection(selection_id=stable_id("exploratory_portfolio_selection", body), **body)
