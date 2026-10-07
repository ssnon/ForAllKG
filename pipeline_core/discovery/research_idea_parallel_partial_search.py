from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_epistemic_archive import (
    EpistemicDecompositionReport,
    EpistemicRealizationRecord,
    MultiRealizationArchiveReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _norm(value: str) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9α-ω가-힣]+", " ", text)
    return " ".join(text.split())


SearchLane = Literal[
    "SAME_IDEA_REALIZATION",
    "CHILD_IDEA_EVOLUTION",
    "PRIOR_ART_PROBE",
    "EVIDENCE_ACQUISITION_ESCALATION",
    "HOLD",
]
DebtDisposition = Literal[
    "RECORDED_ONLY",
    "PROBE_ELIGIBLE",
    "ACQUISITION_ESCALATION_ELIGIBLE",
]
ParentingMode = Literal[
    "DIRECT_PARENT",
    "CONDITIONAL_PARENT",
    "NOT_PARENT_ELIGIBLE",
]


class EpistemicDebtTicket(StrictModel):
    schema_version: Literal[
        "research-idea-epistemic-debt-ticket-v1"
    ] = "research-idea-epistemic-debt-ticket-v1"
    debt_id: str
    idea_id: str
    generation_index: int = Field(ge=0)
    requirement_kind: str
    normalized_description: str
    source_requirement_ids: list[str] = Field(default_factory=list)
    source_epistemic_realization_ids: list[str] = Field(default_factory=list)
    source_realization_ids: list[str] = Field(default_factory=list)
    occurrence_count: int = Field(ge=1)
    recovery_routes_observed: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    disposition: DebtDisposition
    blocks_search_continuation: Literal[False] = False
    blocks_child_idea_generation: Literal[False] = False
    requires_positive_evidence_before_scientific_claim: Literal[True] = True
    automatic_acquisition_executed: Literal[False] = False
    automatic_graph_mutation_executed: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ParallelRealizationSearchMember(StrictModel):
    schema_version: Literal[
        "parallel-partial-realization-search-member-v1"
    ] = "parallel-partial-realization-search-member-v1"
    member_id: str
    idea_id: str
    generation_index: int = Field(ge=0)
    epistemic_realization_id: str
    realization_id: str | None = None
    hypothesis_id: str | None = None
    epistemic_maturity: str
    grounding_coverage: str
    materialization_status: str
    archive_retained: bool
    prediction_present: bool
    falsifier_present: bool
    missing_evidence_requirement_count: int = Field(ge=0)
    residual_epistemic_state: str | None = None
    prospective_identifiability: str | None = None

    search_lanes: list[SearchLane] = Field(default_factory=list)
    parenting_mode: ParentingMode
    eligible_for_same_idea_search: bool
    eligible_for_child_idea_parenting: bool
    eligible_for_program_selection: bool
    eligible_for_prior_art_probe: bool
    eligible_for_evidence_acquisition_escalation: bool

    grounding_required_before_claim: Literal[True] = True
    evidence_debt_is_not_search_blocker: Literal[True] = True
    research_idea_identity_is_not_realization_status: Literal[True] = True
    speculative_member_is_not_positive_premise: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_lanes(self) -> "ParallelRealizationSearchMember":
        self.search_lanes = list(dict.fromkeys(self.search_lanes))
        if self.eligible_for_same_idea_search != ("SAME_IDEA_REALIZATION" in self.search_lanes):
            raise ValueError("same-idea eligibility / lane mismatch")
        if self.eligible_for_child_idea_parenting != ("CHILD_IDEA_EVOLUTION" in self.search_lanes):
            raise ValueError("child-parent eligibility / lane mismatch")
        if self.eligible_for_prior_art_probe != ("PRIOR_ART_PROBE" in self.search_lanes):
            raise ValueError("prior-art eligibility / lane mismatch")
        if self.eligible_for_evidence_acquisition_escalation != (
            "EVIDENCE_ACQUISITION_ESCALATION" in self.search_lanes
        ):
            raise ValueError("acquisition escalation eligibility / lane mismatch")
        return self


class ParallelIdeaSearchState(StrictModel):
    schema_version: Literal[
        "parallel-partial-research-idea-search-state-v1"
    ] = "parallel-partial-research-idea-search-state-v1"
    idea_id: str
    generation_index: int = Field(ge=0)
    member_ids: list[str] = Field(default_factory=list)
    retained_member_ids: list[str] = Field(default_factory=list)
    active_member_ids: list[str] = Field(default_factory=list)
    epistemic_debt_ids: list[str] = Field(default_factory=list)
    maturity_counts: dict[str, int] = Field(default_factory=dict)
    search_lane_counts: dict[str, int] = Field(default_factory=dict)
    child_parent_member_ids: list[str] = Field(default_factory=list)
    same_idea_member_ids: list[str] = Field(default_factory=list)
    prior_art_probe_member_ids: list[str] = Field(default_factory=list)
    acquisition_escalation_debt_ids: list[str] = Field(default_factory=list)
    remains_in_search_population: bool
    has_parallel_realization_paths: bool
    evidence_acquisition_is_optional_escalation: Literal[True] = True
    evidence_debt_is_not_idea_failure: Literal[True] = True
    grounding_does_not_control_imagination: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ParallelIdeaEvolutionHandoff(StrictModel):
    schema_version: Literal[
        "parallel-partial-idea-evolution-handoff-v1"
    ] = "parallel-partial-idea-evolution-handoff-v1"
    handoff_id: str
    idea_id: str
    generation_index: int = Field(ge=0)
    source_member_ids: list[str] = Field(default_factory=list)
    source_epistemic_realization_ids: list[str] = Field(default_factory=list)
    recommended_channels: list[Literal["TRANSFORM", "EXPLORE", "WILDCARD"]] = Field(default_factory=list)
    nonbinding_operator_hints: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    grounding_is_not_precondition_for_child_idea_generation: Literal[True] = True
    child_idea_remains_inspiration_only_until_realized: Literal[True] = True
    executes_offspring_generation: Literal[False] = False
    production_generation_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False


class ParallelPartialSearchReport(StrictModel):
    schema_version: Literal[
        "sis-v2-8b-parallel-partial-realization-search-shadow-v1"
    ] = "sis-v2-8b-parallel-partial-realization-search-shadow-v1"
    report_id: str
    report_sha256: str
    members: list[ParallelRealizationSearchMember] = Field(default_factory=list)
    idea_states: list[ParallelIdeaSearchState] = Field(default_factory=list)
    epistemic_debts: list[EpistemicDebtTicket] = Field(default_factory=list)
    evolution_handoffs: list[ParallelIdeaEvolutionHandoff] = Field(default_factory=list)

    member_count: int = Field(ge=0)
    retained_member_count: int = Field(ge=0)
    active_member_count: int = Field(ge=0)
    idea_count: int = Field(ge=0)
    active_idea_count: int = Field(ge=0)
    partially_grounded_active_count: int = Field(ge=0)
    speculative_active_count: int = Field(ge=0)
    evidence_seeking_active_count: int = Field(ge=0)
    child_parent_eligible_member_count: int = Field(ge=0)
    same_idea_search_eligible_member_count: int = Field(ge=0)
    epistemic_debt_count: int = Field(ge=0)
    acquisition_escalation_eligible_debt_count: int = Field(ge=0)
    evolution_handoff_count: int = Field(ge=0)
    maturity_counts: dict[str, int] = Field(default_factory=dict)
    lane_counts: dict[str, int] = Field(default_factory=dict)

    evidence_recovery_is_blocking_inner_loop: Literal[False] = False
    partial_realizations_remain_search_reproductive: Literal[True] = True
    speculative_falsifiable_realizations_remain_search_reproductive: Literal[True] = True
    evidence_acquisition_is_optional_escalation: Literal[True] = True
    external_probe_is_not_positive_premise: Literal[True] = True
    strict_hypothesis_card_contract_preserved: Literal[True] = True
    new_retrieval_calls: Literal[False] = False
    new_llm_calls: Literal[False] = False
    offspring_generation_executed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "ParallelPartialSearchReport":
        if self.member_count != len(self.members):
            raise ValueError("member_count mismatch")
        if self.idea_count != len(self.idea_states):
            raise ValueError("idea_count mismatch")
        if self.epistemic_debt_count != len(self.epistemic_debts):
            raise ValueError("epistemic_debt_count mismatch")
        if self.evolution_handoff_count != len(self.evolution_handoffs):
            raise ValueError("evolution_handoff_count mismatch")
        return self


def _archive_retained_ids(archive: MultiRealizationArchiveReport) -> set[str]:
    return {
        entry.epistemic_realization_id
        for idea_archive in archive.archives
        for entry in idea_archive.entries
        if entry.archive_retained
    }


def _member_lanes(row: EpistemicRealizationRecord, *, archive_retained: bool) -> tuple[list[SearchLane], ParentingMode]:
    if not archive_retained:
        return ["HOLD"], "NOT_PARENT_ELIGIBLE"

    maturity = row.epistemic_maturity
    has_prediction = bool(row.prediction_texts)
    has_falsifier = bool(row.falsifier_texts)
    falsifiable = has_prediction and has_falsifier
    lanes: list[SearchLane] = []

    # Grounding controls claims, not search survival. Non-final realizations can
    # keep searching locally without first expanding the canonical KG.
    if maturity in {
        "IDEA_ONLY",
        "PARTIALLY_GROUNDED",
        "EVIDENCE_SEEKING",
        "SPECULATIVE_BUT_FALSIFIABLE",
        "STRICT_GROUNDED",
    }:
        lanes.append("SAME_IDEA_REALIZATION")

    # Conceptually coherent/falsifiable partial states may reproduce at the
    # ResearchIdea level. Strict grounding is deliberately not a prerequisite.
    if maturity in {
        "PARTIALLY_GROUNDED",
        "EVIDENCE_SEEKING",
        "SPECULATIVE_BUT_FALSIFIABLE",
        "STRICT_GROUNDED",
        "OPERATIONAL_GROUNDED",
    } and (falsifiable or row.hypothesis_id is not None):
        lanes.append("CHILD_IDEA_EVOLUTION")

    if row.missing_evidence_requirements or maturity in {
        "EVIDENCE_SEEKING",
        "SPECULATIVE_BUT_FALSIFIABLE",
        "PARTIALLY_GROUNDED",
    }:
        lanes.append("PRIOR_ART_PROBE")

    if not lanes:
        lanes.append("HOLD")

    if "CHILD_IDEA_EVOLUTION" in lanes:
        parenting: ParentingMode = (
            "DIRECT_PARENT" if falsifiable or row.hypothesis_id is not None else "CONDITIONAL_PARENT"
        )
    elif maturity == "IDEA_ONLY" and (has_prediction or has_falsifier):
        parenting = "CONDITIONAL_PARENT"
    else:
        parenting = "NOT_PARENT_ELIGIBLE"
    return lanes, parenting


def _build_debts(
    decomposition: EpistemicDecompositionReport,
    *,
    acquisition_persistence_threshold: int,
) -> list[EpistemicDebtTicket]:
    grouped: dict[tuple[str, str, str], list[tuple[EpistemicRealizationRecord, Any]]] = defaultdict(list)
    for row in decomposition.records:
        for req in row.missing_evidence_requirements:
            grouped[(row.idea_id, req.kind, _norm(req.description))].append((row, req))

    debts: list[EpistemicDebtTicket] = []
    for (idea_id, kind, description), occurrences in sorted(grouped.items()):
        source_records = [row for row, _ in occurrences]
        requirements = [req for _, req in occurrences]
        count = len(occurrences)
        literature_possible = any(
            "LITERATURE_ACQUISITION" in req.recovery_routes for req in requirements
        )
        if literature_possible and count >= acquisition_persistence_threshold:
            disposition: DebtDisposition = "ACQUISITION_ESCALATION_ELIGIBLE"
        elif literature_possible:
            disposition = "PROBE_ELIGIBLE"
        else:
            disposition = "RECORDED_ONLY"
        debts.append(
            EpistemicDebtTicket(
                debt_id=_stable_id("epistemic_debt", idea_id, kind, description),
                idea_id=idea_id,
                generation_index=max(row.generation_index for row in source_records),
                requirement_kind=kind,
                normalized_description=description,
                source_requirement_ids=_dedupe([req.requirement_id for req in requirements]),
                source_epistemic_realization_ids=_dedupe(
                    [row.epistemic_realization_id for row in source_records]
                ),
                source_realization_ids=_dedupe(
                    [row.realization_id or "" for row in source_records]
                ),
                occurrence_count=count,
                recovery_routes_observed=_dedupe(
                    [route for req in requirements for route in req.recovery_routes]
                ),
                reason_codes=_dedupe([code for req in requirements for code in req.reason_codes]),
                disposition=disposition,
            )
        )
    return debts


def _handoff_for_idea(
    idea_id: str,
    generation_index: int,
    members: Sequence[ParallelRealizationSearchMember],
    debts: Sequence[EpistemicDebtTicket],
) -> ParallelIdeaEvolutionHandoff | None:
    parents = [row for row in members if row.eligible_for_child_idea_parenting]
    if not parents:
        return None

    maturities = {row.epistemic_maturity for row in parents}
    reason_codes: list[str] = []
    channels: list[Literal["TRANSFORM", "EXPLORE", "WILDCARD"]] = []
    operator_hints: list[str] = []

    if "SPECULATIVE_BUT_FALSIFIABLE" in maturities:
        channels.extend(["TRANSFORM", "EXPLORE"])
        operator_hints.extend(["LATENT_VARIABLE", "REGIME_BOUNDARY", "PROXY_CHALLENGE"])
        reason_codes.append("FALSIFIABLE_SPECULATIVE_MEMBER_REMAINS_REPRODUCTIVE")
    if "PARTIALLY_GROUNDED" in maturities or "EVIDENCE_SEEKING" in maturities:
        channels.extend(["TRANSFORM", "EXPLORE"])
        operator_hints.extend(["AXIS_MUTATION", "CANDIDATE_INTERPRETATION"])
        reason_codes.append("PARTIAL_GROUNDING_DOES_NOT_BLOCK_IDEA_EVOLUTION")
    if any(row.epistemic_maturity in {"STRICT_GROUNDED", "OPERATIONAL_GROUNDED"} for row in parents):
        channels.append("TRANSFORM")
        reason_codes.append("GROUNDED_MEMBER_AVAILABLE_AS_PARALLEL_PARENT")
    if debts:
        channels.append("EXPLORE")
        operator_hints.extend(["REGIME_BOUNDARY", "PROXY_CHALLENGE"])
        reason_codes.append("EPISTEMIC_DEBT_IS_SEARCH_FEEDBACK_NOT_GENERATION_BLOCKER")
    if not channels:
        channels.append("TRANSFORM")
    return ParallelIdeaEvolutionHandoff(
        handoff_id=_stable_id(
            "parallel_idea_evolution_handoff",
            idea_id,
            [row.member_id for row in parents],
            [row.debt_id for row in debts],
        ),
        idea_id=idea_id,
        generation_index=generation_index,
        source_member_ids=[row.member_id for row in parents],
        source_epistemic_realization_ids=[row.epistemic_realization_id for row in parents],
        recommended_channels=list(dict.fromkeys(channels)),
        nonbinding_operator_hints=_dedupe(operator_hints),
        reason_codes=_dedupe(reason_codes),
    )


def build_parallel_partial_search_report(
    *,
    decomposition: EpistemicDecompositionReport,
    archive: MultiRealizationArchiveReport,
    acquisition_persistence_threshold: int = 2,
) -> ParallelPartialSearchReport:
    if acquisition_persistence_threshold < 1:
        raise ValueError("acquisition_persistence_threshold must be >= 1")

    retained_ids = _archive_retained_ids(archive)
    debts = _build_debts(
        decomposition,
        acquisition_persistence_threshold=acquisition_persistence_threshold,
    )
    debts_by_idea: dict[str, list[EpistemicDebtTicket]] = defaultdict(list)
    for debt in debts:
        debts_by_idea[debt.idea_id].append(debt)
    escalated_ideas = {
        debt.idea_id
        for debt in debts
        if debt.disposition == "ACQUISITION_ESCALATION_ELIGIBLE"
    }

    members: list[ParallelRealizationSearchMember] = []
    by_idea: dict[str, list[ParallelRealizationSearchMember]] = defaultdict(list)
    record_by_eid = {row.epistemic_realization_id: row for row in decomposition.records}
    for row in decomposition.records:
        retained = row.epistemic_realization_id in retained_ids
        lanes, parenting = _member_lanes(row, archive_retained=retained)
        if retained and row.idea_id in escalated_ideas and row.missing_evidence_requirements:
            lanes = list(dict.fromkeys([*lanes, "EVIDENCE_ACQUISITION_ESCALATION"]))
        prediction_present = bool(row.prediction_texts)
        falsifier_present = bool(row.falsifier_texts)
        member = ParallelRealizationSearchMember(
            member_id=_stable_id("parallel_partial_member", row.epistemic_realization_id),
            idea_id=row.idea_id,
            generation_index=row.generation_index,
            epistemic_realization_id=row.epistemic_realization_id,
            realization_id=row.realization_id,
            hypothesis_id=row.hypothesis_id,
            epistemic_maturity=row.epistemic_maturity,
            grounding_coverage=row.grounding_coverage,
            materialization_status=row.materialization_status,
            archive_retained=retained,
            prediction_present=prediction_present,
            falsifier_present=falsifier_present,
            missing_evidence_requirement_count=len(row.missing_evidence_requirements),
            residual_epistemic_state=row.residual_epistemic_state,
            prospective_identifiability=row.prospective_identifiability,
            search_lanes=lanes,
            parenting_mode=parenting,
            eligible_for_same_idea_search="SAME_IDEA_REALIZATION" in lanes,
            eligible_for_child_idea_parenting="CHILD_IDEA_EVOLUTION" in lanes,
            eligible_for_program_selection=retained and "HOLD" not in lanes,
            eligible_for_prior_art_probe="PRIOR_ART_PROBE" in lanes,
            eligible_for_evidence_acquisition_escalation=(
                "EVIDENCE_ACQUISITION_ESCALATION" in lanes
            ),
        )
        members.append(member)
        by_idea[row.idea_id].append(member)

    idea_states: list[ParallelIdeaSearchState] = []
    handoffs: list[ParallelIdeaEvolutionHandoff] = []
    for idea_id in sorted(by_idea):
        rows = by_idea[idea_id]
        idea_debts = debts_by_idea.get(idea_id, [])
        retained = [row for row in rows if row.archive_retained]
        active = [row for row in retained if "HOLD" not in row.search_lanes]
        lane_counts = Counter(lane for row in rows for lane in row.search_lanes)
        state = ParallelIdeaSearchState(
            idea_id=idea_id,
            generation_index=max(row.generation_index for row in rows),
            member_ids=[row.member_id for row in rows],
            retained_member_ids=[row.member_id for row in retained],
            active_member_ids=[row.member_id for row in active],
            epistemic_debt_ids=[row.debt_id for row in idea_debts],
            maturity_counts=dict(sorted(Counter(row.epistemic_maturity for row in rows).items())),
            search_lane_counts=dict(sorted(lane_counts.items())),
            child_parent_member_ids=[row.member_id for row in rows if row.eligible_for_child_idea_parenting],
            same_idea_member_ids=[row.member_id for row in rows if row.eligible_for_same_idea_search],
            prior_art_probe_member_ids=[row.member_id for row in rows if row.eligible_for_prior_art_probe],
            acquisition_escalation_debt_ids=[
                row.debt_id
                for row in idea_debts
                if row.disposition == "ACQUISITION_ESCALATION_ELIGIBLE"
            ],
            remains_in_search_population=bool(active),
            has_parallel_realization_paths=len(active) > 1,
        )
        idea_states.append(state)
        handoff = _handoff_for_idea(
            idea_id,
            state.generation_index,
            rows,
            idea_debts,
        )
        if handoff is not None:
            handoffs.append(handoff)

    maturity_counts = Counter(row.epistemic_maturity for row in members)
    lane_counts = Counter(lane for row in members for lane in row.search_lanes)
    provisional = ParallelPartialSearchReport(
        report_id="pending",
        report_sha256="pending",
        members=members,
        idea_states=idea_states,
        epistemic_debts=debts,
        evolution_handoffs=handoffs,
        member_count=len(members),
        retained_member_count=sum(row.archive_retained for row in members),
        active_member_count=sum(row.archive_retained and "HOLD" not in row.search_lanes for row in members),
        idea_count=len(idea_states),
        active_idea_count=sum(row.remains_in_search_population for row in idea_states),
        partially_grounded_active_count=sum(
            row.archive_retained and row.epistemic_maturity == "PARTIALLY_GROUNDED"
            for row in members
        ),
        speculative_active_count=sum(
            row.archive_retained and row.epistemic_maturity == "SPECULATIVE_BUT_FALSIFIABLE"
            for row in members
        ),
        evidence_seeking_active_count=sum(
            row.archive_retained and row.epistemic_maturity == "EVIDENCE_SEEKING"
            for row in members
        ),
        child_parent_eligible_member_count=sum(row.eligible_for_child_idea_parenting for row in members),
        same_idea_search_eligible_member_count=sum(row.eligible_for_same_idea_search for row in members),
        epistemic_debt_count=len(debts),
        acquisition_escalation_eligible_debt_count=sum(
            row.disposition == "ACQUISITION_ESCALATION_ELIGIBLE" for row in debts
        ),
        evolution_handoff_count=len(handoffs),
        maturity_counts=dict(sorted(maturity_counts.items())),
        lane_counts=dict(sorted(lane_counts.items())),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"parallel_partial_search:{digest[:20]}",
            "report_sha256": digest,
        }
    )


__all__ = [
    "DebtDisposition",
    "EpistemicDebtTicket",
    "ParallelIdeaEvolutionHandoff",
    "ParallelIdeaSearchState",
    "ParallelPartialSearchReport",
    "ParallelRealizationSearchMember",
    "ParentingMode",
    "SearchLane",
    "build_parallel_partial_search_report",
]
