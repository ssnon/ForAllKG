from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import ExternalNoveltyReport
from pipeline_core.discovery.research_idea_program_family_v3_3_1 import (
    ScientificProgramFamilyReport,
)
from pipeline_core.discovery.research_idea_search_assessment import (
    ResearchIdeaSearchAssessmentReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


PressureLevel = Literal["NONE", "LOW", "MODERATE", "HIGH", "INDETERMINATE"]
SearchPriority = Literal["DEFERRED", "LOW", "MODERATE", "HIGH"]
PreferredSearchAction = Literal["NONE", "REALIZE", "PROBE", "EVOLVE", "DIVERSIFY"]
NoveltyShape = Literal[
    "NOT_ASSESSED",
    "UNRESOLVED",
    "CONFLICTING",
    "SATURATED",
    "EXTENSION",
    "SHALLOW_RESIDUE",
    "CENTRAL_RESIDUE",
    "STRONG_RESIDUE",
]


class ResearchIdeaNoveltyDepthSignal(StrictModel):
    idea_id: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    external_status_counts: dict[str, int] = Field(default_factory=dict)
    novelty_shape: NoveltyShape
    search_coverage_sufficient: bool
    core_claim_count: int = Field(ge=0)
    relation_backed_core_claim_count: int = Field(ge=0)
    known_core_relation_fraction_max: float = Field(ge=0.0, le=1.0)
    novelty_bearing_claim_count: int = Field(ge=0)
    novelty_bearing_gap_like_claim_count: int = Field(ge=0)
    novelty_bearing_relation_backed_claim_count: int = Field(ge=0)
    novelty_bearing_conflicting_claim_count: int = Field(ge=0)
    novelty_bearing_unresolved_claim_count: int = Field(ge=0)
    gap_centrality_counts: dict[str, int] = Field(default_factory=dict)
    novelty_bearing_prior_art_state_counts: dict[str, int] = Field(default_factory=dict)
    relational_gap_kind_counts: dict[str, int] = Field(default_factory=dict)
    claim_prior_art_status_counts: dict[str, int] = Field(default_factory=dict)

    source_mode: Literal[
        "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
        "CONTEMPORANEOUS_SEARCH_PROBE",
        "NONE",
    ] = "NONE"
    prior_art_only_not_positive_premise: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_certification_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False


class ResearchIdeaSearchPressure(StrictModel):
    idea_id: str = Field(min_length=1)
    generation_index: int = Field(ge=2)
    remains_active: bool
    persistence: Literal["RETAIN", "ARCHIVE"]

    novelty_shape: NoveltyShape
    search_coverage_sufficient: bool
    same_program_key: str = Field(min_length=1)
    same_program_size: int = Field(ge=1)
    adjacent_program_neighbor_count: int = Field(ge=0)

    terminal_worthy: bool
    terminal_caveat_codes: list[str] = Field(default_factory=list)
    search_worthy: bool
    preferred_action: PreferredSearchAction
    search_priority: SearchPriority

    prior_art_escape_pressure: PressureLevel
    residual_refinement_pressure: PressureLevel
    discrimination_upside: PressureLevel
    evidence_pressure: PressureLevel
    family_diversification_pressure: PressureLevel
    realization_pressure: PressureLevel

    recommended_channels: list[Literal["TRANSFORM", "EXPLORE", "WILDCARD"]] = Field(
        default_factory=list
    )
    nonbinding_operator_hints: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    grounded_does_not_imply_search_complete: Literal[True] = True
    terminal_worthy_is_independent_from_search_worthy: Literal[True] = True
    single_scalar_fitness_used: Literal[False] = False
    shadow_only: Literal[True] = True
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _semantics(self) -> "ResearchIdeaSearchPressure":
        if self.persistence == "ARCHIVE" and self.preferred_action != "NONE":
            raise ValueError("archived idea must have preferred_action=NONE")
        if self.search_worthy != (self.preferred_action in {"EVOLVE", "DIVERSIFY"}):
            raise ValueError("search_worthy / preferred_action mismatch")
        return self


class ResearchIdeaSearchPressureReport(StrictModel):
    schema_version: Literal[
        "sis-v3-3-2-research-idea-search-pressure-shadow-v1"
    ] = "sis-v3-3-2-research-idea-search-pressure-shadow-v1"
    report_id: str
    report_sha256: str
    generation_index: int = Field(ge=2)
    source_search_assessment_report_id: str
    source_program_family_report_id: str
    pressures: list[ResearchIdeaSearchPressure] = Field(default_factory=list)
    pressure_count: int = Field(ge=0)
    preferred_action_counts: dict[str, int] = Field(default_factory=dict)
    search_priority_counts: dict[str, int] = Field(default_factory=dict)
    novelty_shape_counts: dict[str, int] = Field(default_factory=dict)
    terminal_worthy_count: int = Field(ge=0)
    search_worthy_count: int = Field(ge=0)

    v3_1_fertility_behavior_changed: Literal[False] = False
    v3_3_behavior_changed: Literal[False] = False
    child_generation_executed: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "ResearchIdeaSearchPressureReport":
        if self.pressure_count != len(self.pressures):
            raise ValueError("pressure_count mismatch")
        return self


class ScheduledResearchIdeaParent(StrictModel):
    idea_id: str = Field(min_length=1)
    scientific_program_key: str = Field(min_length=1)
    selected_rank: int = Field(ge=1)
    preferred_action: Literal["EVOLVE", "DIVERSIFY"]
    search_priority: Literal["LOW", "MODERATE", "HIGH"]
    terminal_worthy: bool
    recommended_channels: list[Literal["TRANSFORM", "EXPLORE", "WILDCARD"]]
    nonbinding_operator_hints: list[str] = Field(default_factory=list)
    selection_reason_codes: list[str] = Field(default_factory=list)


class BoundedResearchIdeaParentSchedule(StrictModel):
    schema_version: Literal[
        "sis-v3-3-2-bounded-research-idea-parent-schedule-shadow-v1"
    ] = "sis-v3-3-2-bounded-research-idea-parent-schedule-shadow-v1"
    schedule_id: str
    schedule_sha256: str
    generation_index: int = Field(ge=2)
    source_pressure_report_id: str
    source_program_family_report_id: str
    max_selected_parents: int = Field(ge=1)
    max_selected_per_same_program: int = Field(ge=1)
    candidate_parent_count: int = Field(ge=0)
    selected_parents: list[ScheduledResearchIdeaParent] = Field(default_factory=list)
    selected_parent_count: int = Field(ge=0)
    deferred_search_worthy_idea_ids: list[str] = Field(default_factory=list)
    selected_program_count: int = Field(ge=0)
    selected_action_counts: dict[str, int] = Field(default_factory=dict)
    selected_priority_counts: dict[str, int] = Field(default_factory=dict)
    selected_pair_relation_counts: dict[str, int] = Field(default_factory=dict)

    bounded_parent_selection: Literal[True] = True
    pairwise_program_relation_used_as_diversity_tiebreak: Literal[True] = True
    program_diversity_precedes_second_parent_from_same_program: Literal[True] = True
    single_scalar_fitness_used: Literal[False] = False
    child_generation_executed: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "BoundedResearchIdeaParentSchedule":
        if self.selected_parent_count != len(self.selected_parents):
            raise ValueError("selected_parent_count mismatch")
        if self.selected_parent_count > self.max_selected_parents:
            raise ValueError("selected parent count exceeds budget")
        return self


def _shape_from_cards(cards: Sequence[Any]) -> NoveltyShape:
    if not cards:
        return "NOT_ASSESSED"
    statuses = Counter(str(_get(card, "status", "")) for card in cards)
    depth_states = Counter()
    gap_centrality = Counter()
    coverage_ok = True
    for card in cards:
        coverage = _get(card, "coverage")
        coverage_ok = coverage_ok and bool(
            _get(coverage, "sufficient_for_absence_based_novelty", False)
        )
        depth = _get(card, "novelty_depth_profile")
        if depth is not None:
            depth_states[str(_get(depth, "novelty_bearing_prior_art_state", ""))] += 1
            gap_centrality[str(_get(depth, "gap_centrality", ""))] += 1

    if statuses.get("CONFLICTING_PRIOR_ART", 0) or depth_states.get("CONFLICTING", 0):
        return "CONFLICTING"
    if (
        statuses.get("INSUFFICIENT_SEARCH_EVIDENCE", 0)
        or depth_states.get("UNRESOLVED", 0)
        or not coverage_ok
    ):
        return "UNRESOLVED"
    if statuses.get("PLAUSIBLY_NOVEL", 0):
        return "STRONG_RESIDUE"
    if (
        gap_centrality.get("CENTRAL", 0)
        or gap_centrality.get("DISTRIBUTED", 0)
        or any(str(_get(card, "relational_gap_kind", "")) == "HIGHER_ORDER_RELATIONAL_GAP" for card in cards)
    ):
        return "CENTRAL_RESIDUE"
    if (
        statuses.get("NEW_COMBINATION_OF_KNOWN_EFFECTS", 0)
        or statuses.get("KNOWN_COMPONENTS_WITH_RELATIONAL_GAP", 0)
        or depth_states.get("ALL_GAP_LIKE", 0)
        or depth_states.get("MIXED", 0)
    ):
        return "SHALLOW_RESIDUE"
    if statuses.get("LITERATURE_SUPPORTED_EXTENSION", 0):
        return "EXTENSION"
    if statuses and set(statuses) <= {"WELL_ESTABLISHED"}:
        return "SATURATED"
    return "UNRESOLVED"


def build_novelty_depth_signals_by_idea(
    *,
    terminal_parallel_report: Any,
    external_novelty_report: ExternalNoveltyReport,
    source_mode: Literal[
        "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
        "CONTEMPORANEOUS_SEARCH_PROBE",
    ] = "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
) -> dict[str, ResearchIdeaNoveltyDepthSignal]:
    hypothesis_ids_by_idea: dict[str, list[str]] = defaultdict(list)
    for member in (_get(terminal_parallel_report, "members", []) or []):
        idea_id = str(_get(member, "idea_id", "") or "").strip()
        hypothesis_id = str(_get(member, "hypothesis_id", "") or "").strip()
        if idea_id and hypothesis_id:
            hypothesis_ids_by_idea[idea_id].append(hypothesis_id)

    cards_by_hid = {card.hypothesis_id: card for card in external_novelty_report.cards}
    result: dict[str, ResearchIdeaNoveltyDepthSignal] = {}
    for idea_id, hypothesis_ids in sorted(hypothesis_ids_by_idea.items()):
        cards = [cards_by_hid[hid] for hid in _dedupe(hypothesis_ids) if hid in cards_by_hid]
        statuses = Counter(card.status for card in cards)
        claim_statuses: Counter[str] = Counter()
        gap_centralities: Counter[str] = Counter()
        prior_states: Counter[str] = Counter()
        gap_kinds: Counter[str] = Counter()
        core_count = 0
        relation_backed_core_count = 0
        novelty_bearing_count = 0
        gap_like_count = 0
        relation_backed_nb_count = 0
        conflicting_count = 0
        unresolved_count = 0
        known_fraction_max = 0.0
        coverage_ok = bool(cards)
        for card in cards:
            coverage_ok = coverage_ok and card.coverage.sufficient_for_absence_based_novelty
            gap_kinds[card.relational_gap_kind] += 1
            for review in card.claim_reviews:
                claim_statuses[review.status] += 1
            depth = card.novelty_depth_profile
            if depth is None:
                continue
            core_count += depth.core_claim_count
            relation_backed_core_count += depth.relation_backed_core_claim_count
            known_fraction_max = max(known_fraction_max, depth.known_core_relation_fraction)
            novelty_bearing_count += depth.novelty_bearing_claim_count
            gap_like_count += len(depth.novelty_bearing_gap_like_claim_ids)
            relation_backed_nb_count += len(depth.novelty_bearing_relation_backed_claim_ids)
            conflicting_count += len(depth.novelty_bearing_conflicting_claim_ids)
            unresolved_count += len(depth.novelty_bearing_unresolved_claim_ids)
            gap_centralities[depth.gap_centrality] += 1
            prior_states[depth.novelty_bearing_prior_art_state] += 1

        result[idea_id] = ResearchIdeaNoveltyDepthSignal(
            idea_id=idea_id,
            hypothesis_ids=[card.hypothesis_id for card in cards],
            external_status_counts=dict(sorted(statuses.items())),
            novelty_shape=_shape_from_cards(cards),
            search_coverage_sufficient=coverage_ok,
            core_claim_count=core_count,
            relation_backed_core_claim_count=relation_backed_core_count,
            known_core_relation_fraction_max=known_fraction_max,
            novelty_bearing_claim_count=novelty_bearing_count,
            novelty_bearing_gap_like_claim_count=gap_like_count,
            novelty_bearing_relation_backed_claim_count=relation_backed_nb_count,
            novelty_bearing_conflicting_claim_count=conflicting_count,
            novelty_bearing_unresolved_claim_count=unresolved_count,
            gap_centrality_counts=dict(sorted(gap_centralities.items())),
            novelty_bearing_prior_art_state_counts=dict(sorted(prior_states.items())),
            relational_gap_kind_counts=dict(sorted(gap_kinds.items())),
            claim_prior_art_status_counts=dict(sorted(claim_statuses.items())),
            source_mode=source_mode,
        )
    return result


def _empty_novelty(idea_id: str) -> ResearchIdeaNoveltyDepthSignal:
    return ResearchIdeaNoveltyDepthSignal(
        idea_id=idea_id,
        novelty_shape="NOT_ASSESSED",
        search_coverage_sufficient=False,
        core_claim_count=0,
        relation_backed_core_claim_count=0,
        known_core_relation_fraction_max=0.0,
        novelty_bearing_claim_count=0,
        novelty_bearing_gap_like_claim_count=0,
        novelty_bearing_relation_backed_claim_count=0,
        novelty_bearing_conflicting_claim_count=0,
        novelty_bearing_unresolved_claim_count=0,
        source_mode="NONE",
    )


def _level_rank(value: PressureLevel) -> int:
    return {"NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "INDETERMINATE": -1}[value]


def _max_level(*values: PressureLevel) -> PressureLevel:
    return max(values, key=_level_rank)


def _pressure_for(
    *,
    assessment: Any,
    family: Any,
    novelty: ResearchIdeaNoveltyDepthSignal,
    generation_index: int,
) -> ResearchIdeaSearchPressure:
    remains_active = bool(assessment.remains_active)
    if not remains_active:
        return ResearchIdeaSearchPressure(
            idea_id=assessment.idea_id,
            generation_index=generation_index,
            remains_active=False,
            persistence="ARCHIVE",
            novelty_shape=novelty.novelty_shape,
            search_coverage_sufficient=novelty.search_coverage_sufficient,
            same_program_key=family.scientific_program_key,
            same_program_size=family.same_program_size,
            adjacent_program_neighbor_count=family.adjacent_program_neighbor_count,
            terminal_worthy=False,
            search_worthy=False,
            preferred_action="NONE",
            search_priority="DEFERRED",
            prior_art_escape_pressure="NONE",
            residual_refinement_pressure="NONE",
            discrimination_upside="NONE",
            evidence_pressure="NONE",
            family_diversification_pressure="NONE",
            realization_pressure="NONE",
            reason_codes=["IDEA_NOT_IN_ACTIVE_SEARCH_POPULATION"],
        )

    grounded = assessment.usable_grounded_realization_count > 0
    discrim_complete = bool(assessment.discrimination_sketch_complete)
    debt_count = len(assessment.epistemic_debt_ids)
    acquisition_debt = assessment.acquisition_escalation_eligible_debt_count > 0

    # v3.3.2: only actual SAME_PROGRAM multiplicity is saturation pressure.
    # ADJACENT_PROGRAM is a useful scientific neighborhood signal and is handled
    # later by the bounded scheduler as a diversity tie-break, never as evidence
    # that a program is overrepresented.
    if family.same_program_size >= 3:
        family_pressure: PressureLevel = "HIGH"
    elif family.same_program_size == 2:
        family_pressure = "MODERATE"
    else:
        family_pressure = "NONE"

    if not grounded and not assessment.local_search_exhausted:
        realization_pressure: PressureLevel = "HIGH"
    elif not grounded:
        realization_pressure = "MODERATE"
    else:
        realization_pressure = "NONE"

    if novelty.novelty_shape in {"NOT_ASSESSED", "UNRESOLVED", "CONFLICTING"}:
        evidence_pressure: PressureLevel = "HIGH"
    elif acquisition_debt:
        evidence_pressure = "HIGH"
    elif debt_count:
        evidence_pressure = "MODERATE"
    else:
        evidence_pressure = "NONE"

    if novelty.novelty_shape == "SATURATED":
        prior_art_escape: PressureLevel = "HIGH"
        residual_refine: PressureLevel = "NONE"
    elif novelty.novelty_shape == "EXTENSION":
        prior_art_escape = "HIGH"
        residual_refine = "LOW"
    elif novelty.novelty_shape == "SHALLOW_RESIDUE":
        prior_art_escape = "MODERATE"
        residual_refine = "MODERATE"
    elif novelty.novelty_shape == "CENTRAL_RESIDUE":
        # A central residue is evidence that the current idea still contains a
        # meaningful unresolved scientific relation.  Its existence alone must
        # not force high-priority mutation.  Stronger pressure can still come
        # from an incomplete discriminator, prior-art escape, or true same-
        # program saturation.
        prior_art_escape = "LOW"
        residual_refine = "MODERATE"
    elif novelty.novelty_shape == "STRONG_RESIDUE":
        prior_art_escape = "NONE"
        residual_refine = "LOW"
    elif novelty.novelty_shape in {"UNRESOLVED", "CONFLICTING", "NOT_ASSESSED"}:
        prior_art_escape = "INDETERMINATE"
        residual_refine = "INDETERMINATE"
    else:
        prior_art_escape = "INDETERMINATE"
        residual_refine = "INDETERMINATE"

    if discrim_complete:
        discrimination: PressureLevel = "NONE"
    elif assessment.differential_prediction_present or assessment.discriminating_observation_present:
        discrimination = "MODERATE"
    else:
        discrimination = "HIGH"

    terminal_caveats: list[str] = []
    if debt_count:
        terminal_caveats.append("PERSISTENT_EPISTEMIC_DEBT")
    if not novelty.search_coverage_sufficient:
        terminal_caveats.append("PRIOR_ART_COVERAGE_NOT_SUFFICIENT_FOR_ABSENCE_CLAIMS")
    if not discrim_complete:
        terminal_caveats.append("DISCRIMINATION_SKETCH_INCOMPLETE")

    terminal_worthy = (
        grounded
        and discrim_complete
        and novelty.novelty_shape
        in {"EXTENSION", "SHALLOW_RESIDUE", "CENTRAL_RESIDUE", "STRONG_RESIDUE"}
    )

    reasons: list[str] = []
    channels: list[Literal["TRANSFORM", "EXPLORE", "WILDCARD"]] = []
    hints: list[str] = []

    if assessment.realization_count == 0 or realization_pressure == "HIGH":
        action: PreferredSearchAction = "REALIZE"
        priority: SearchPriority = "HIGH"
        reasons.append("SAME_IDEA_REALIZATION_REMAINS_PRIMARY_NEXT_ACTION")
    elif novelty.novelty_shape in {"NOT_ASSESSED", "UNRESOLVED", "CONFLICTING"}:
        action = "PROBE"
        priority = "HIGH"
        reasons.append("PRIOR_ART_OR_CONFLICT_STATE_REQUIRES_PROBE_BEFORE_EVOLUTION")
    else:
        evolution_pressure = _max_level(
            prior_art_escape,
            residual_refine,
            discrimination,
            family_pressure,
        )
        if evolution_pressure == "HIGH":
            priority = "HIGH"
        elif evolution_pressure == "MODERATE":
            priority = "MODERATE"
        elif evolution_pressure == "LOW":
            priority = "LOW"
        else:
            priority = "DEFERRED"

        if novelty.novelty_shape == "STRONG_RESIDUE" and discrim_complete and family_pressure == "NONE":
            action = "NONE"
            priority = "DEFERRED"
            reasons.append("STRONG_RESIDUE_WITH_COMPLETE_DISCRIMINATION_CAN_BE_TERMINAL_WORTHY")
        elif priority in {"HIGH", "MODERATE"}:
            if family_pressure == "HIGH" or (
                family_pressure == "MODERATE" and prior_art_escape in {"HIGH", "MODERATE"}
            ):
                action = "DIVERSIFY"
                channels = ["EXPLORE", "WILDCARD"]
                hints = ["LATENT_VARIABLE", "REGIME_BOUNDARY", "PROXY_CHALLENGE", "CROSS_SOURCE_BRIDGE"]
                reasons.append("PROGRAM_SPACE_OVERREPRESENTED_RELATIVE_TO_SEARCH_UPSIDE")
            else:
                action = "EVOLVE"
                channels = ["TRANSFORM", "EXPLORE"]
                hints = ["BACKBONE_MUTATION", "LATENT_VARIABLE", "REGIME_BOUNDARY", "PROXY_CHALLENGE"]
                reasons.append("SCIENTIFIC_UPSIDE_SUPPORTS_BOUNDED_CHILD_EXPLORATION")
        else:
            action = "NONE"
            reasons.append("NO_HIGH_OR_MODERATE_CHILD_SEARCH_PRESSURE")

    if terminal_worthy:
        reasons.append("CURRENT_REALIZATION_IS_TERMINAL_WORTHY_EVEN_IF_SEARCH_CONTINUES")
    if grounded:
        reasons.append("GROUNDED_REALIZATION_PRESENT_BUT_NOT_SEARCH_COMPLETION_AUTHORITY")
    if novelty.novelty_shape in {"SHALLOW_RESIDUE", "CENTRAL_RESIDUE", "STRONG_RESIDUE"}:
        reasons.append("NOVELTY_RESIDUE_IS_SEARCH_FEEDBACK_NOT_AUTOMATIC_EVOLUTION_AUTHORITY")

    return ResearchIdeaSearchPressure(
        idea_id=assessment.idea_id,
        generation_index=generation_index,
        remains_active=True,
        persistence="RETAIN",
        novelty_shape=novelty.novelty_shape,
        search_coverage_sufficient=novelty.search_coverage_sufficient,
        same_program_key=family.scientific_program_key,
        same_program_size=family.same_program_size,
        adjacent_program_neighbor_count=family.adjacent_program_neighbor_count,
        terminal_worthy=terminal_worthy,
        terminal_caveat_codes=_dedupe(terminal_caveats),
        search_worthy=action in {"EVOLVE", "DIVERSIFY"},
        preferred_action=action,
        search_priority=priority,
        prior_art_escape_pressure=prior_art_escape,
        residual_refinement_pressure=residual_refine,
        discrimination_upside=discrimination,
        evidence_pressure=evidence_pressure,
        family_diversification_pressure=family_pressure,
        realization_pressure=realization_pressure,
        recommended_channels=_dedupe(channels),
        nonbinding_operator_hints=_dedupe(hints),
        reason_codes=_dedupe(reasons),
    )


def build_research_idea_search_pressure_report(
    *,
    search_assessment: ResearchIdeaSearchAssessmentReport,
    program_family: ScientificProgramFamilyReport,
    novelty_by_idea: Mapping[str, ResearchIdeaNoveltyDepthSignal] | None = None,
) -> ResearchIdeaSearchPressureReport:
    novelty_by_idea = dict(novelty_by_idea or {})
    family_by_id = {row.idea_id: row for row in program_family.assignments}
    if set(family_by_id) != {row.idea_id for row in search_assessment.assessments}:
        raise ValueError("program-family/search-assessment idea sets differ")

    pressures: list[ResearchIdeaSearchPressure] = []
    for assessment in search_assessment.assessments:
        pressures.append(
            _pressure_for(
                assessment=assessment,
                family=family_by_id[assessment.idea_id],
                novelty=novelty_by_idea.get(assessment.idea_id, _empty_novelty(assessment.idea_id)),
                generation_index=search_assessment.generation_index,
            )
        )

    provisional = ResearchIdeaSearchPressureReport(
        report_id="pending",
        report_sha256="pending",
        generation_index=search_assessment.generation_index,
        source_search_assessment_report_id=search_assessment.report_id,
        source_program_family_report_id=program_family.report_id,
        pressures=pressures,
        pressure_count=len(pressures),
        preferred_action_counts=dict(sorted(Counter(row.preferred_action for row in pressures).items())),
        search_priority_counts=dict(sorted(Counter(row.search_priority for row in pressures).items())),
        novelty_shape_counts=dict(sorted(Counter(row.novelty_shape for row in pressures).items())),
        terminal_worthy_count=sum(row.terminal_worthy for row in pressures),
        search_worthy_count=sum(row.search_worthy for row in pressures),
    )
    body = provisional.model_dump(mode="json")
    body.pop("report_id", None)
    body.pop("report_sha256", None)
    digest = _sha(body)
    return provisional.model_copy(
        update={
            "report_id": f"research_idea_search_pressure_v3_3_2:g{search_assessment.generation_index}:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def _priority_rank(value: SearchPriority) -> int:
    return {"DEFERRED": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}[value]


def _pressure_rank(value: PressureLevel) -> int:
    return {"INDETERMINATE": -1, "NONE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3}[value]


def _candidate_key(row: ResearchIdeaSearchPressure) -> tuple[Any, ...]:
    # Scientific upside precedes action labels.  DIVERSIFY is a routing choice,
    # not evidence that the candidate is intrinsically better.  Terminal-worthy
    # parents are not penalized: a good current hypothesis may still be the best
    # parent for further search.
    return (
        _priority_rank(row.search_priority),
        _pressure_rank(row.discrimination_upside),
        _pressure_rank(row.residual_refinement_pressure),
        _pressure_rank(row.prior_art_escape_pressure),
        _pressure_rank(row.family_diversification_pressure),
        int(row.terminal_worthy),
        row.idea_id,
    )


def _program_pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def _program_relation_rank(relation: str) -> int:
    return {
        "SAME_PROGRAM": 0,
        "ADJACENT_PROGRAM": 1,
        "DISTINCT_PROGRAM": 2,
    }.get(str(relation), 0)


def _relation_diversity_key(
    row: ResearchIdeaSearchPressure,
    *,
    selected: Sequence[ResearchIdeaSearchPressure],
    relation_by_pair: Mapping[tuple[str, str], str],
) -> tuple[int, int]:
    if not selected:
        # No prior parent exists, so program distance cannot discriminate yet.
        return (2, 0)
    ranks = [
        _program_relation_rank(
            relation_by_pair.get(
                _program_pair_key(row.idea_id, prior.idea_id),
                "DISTINCT_PROGRAM",
            )
        )
        for prior in selected
    ]
    # Prefer a candidate that is not close to ANY already-selected program;
    # total distance breaks ties among equally safe candidates.
    return (min(ranks), sum(ranks))


def build_bounded_parent_schedule(
    *,
    pressure_report: ResearchIdeaSearchPressureReport,
    program_family: ScientificProgramFamilyReport,
    max_selected_parents: int = 3,
    max_selected_per_same_program: int = 1,
) -> BoundedResearchIdeaParentSchedule:
    if max_selected_parents < 1:
        raise ValueError("max_selected_parents must be >= 1")
    if max_selected_per_same_program < 1:
        raise ValueError("max_selected_per_same_program must be >= 1")

    candidates = [
        row
        for row in pressure_report.pressures
        if row.search_worthy and row.search_priority in {"HIGH", "MODERATE", "LOW"}
    ]
    relation_by_pair = {
        _program_pair_key(row.idea_id_a, row.idea_id_b): row.relation
        for row in program_family.pair_assessments
    }
    family_by_idea = {row.idea_id: row for row in program_family.assignments}
    if set(family_by_idea) != {row.idea_id for row in pressure_report.pressures}:
        raise ValueError("program-family/pressure-report idea sets differ")

    selected: list[ResearchIdeaSearchPressure] = []
    family_counts: Counter[str] = Counter()
    remaining = list(candidates)

    # Greedy bounded scheduling. Scientific search pressure is primary; pairwise
    # program distance is the diversity tie-break. This uses the richer semantic
    # relation graph instead of treating every non-SAME family as equally distinct.
    while remaining and len(selected) < max_selected_parents:
        eligible = [
            row
            for row in remaining
            if family_counts[row.same_program_key] < max_selected_per_same_program
        ]
        if not eligible:
            break
        best = max(
            eligible,
            key=lambda row: (
                _priority_rank(row.search_priority),
                *_relation_diversity_key(
                    row,
                    selected=selected,
                    relation_by_pair=relation_by_pair,
                ),
                *_candidate_key(row)[1:],
            ),
        )
        selected.append(best)
        family_counts[best.same_program_key] += 1
        remaining.remove(best)

    ordered_candidates = sorted(candidates, key=_candidate_key, reverse=True)
    selected_ids = {row.idea_id for row in selected}
    scheduled = [
        ScheduledResearchIdeaParent(
            idea_id=row.idea_id,
            scientific_program_key=row.same_program_key,
            selected_rank=rank,
            preferred_action=row.preferred_action,  # type: ignore[arg-type]
            search_priority=row.search_priority,  # type: ignore[arg-type]
            terminal_worthy=row.terminal_worthy,
            recommended_channels=list(row.recommended_channels),
            nonbinding_operator_hints=list(row.nonbinding_operator_hints),
            selection_reason_codes=[
                "BOUNDED_PARENT_BUDGET",
                "PAIRWISE_PROGRAM_DISTANCE_TIEBREAK",
                "ORDINAL_LEXICOGRAPHIC_PRESSURE_SELECTION",
                *( ["CURRENT_PARENT_IS_ALSO_TERMINAL_WORTHY"] if row.terminal_worthy else [] ),
            ],
        )
        for rank, row in enumerate(selected, start=1)
    ]

    provisional = BoundedResearchIdeaParentSchedule(
        schedule_id="pending",
        schedule_sha256="pending",
        generation_index=pressure_report.generation_index,
        source_pressure_report_id=pressure_report.report_id,
        source_program_family_report_id=program_family.report_id,
        max_selected_parents=max_selected_parents,
        max_selected_per_same_program=max_selected_per_same_program,
        candidate_parent_count=len(candidates),
        selected_parents=scheduled,
        selected_parent_count=len(scheduled),
        deferred_search_worthy_idea_ids=[
            row.idea_id
            for row in ordered_candidates
            if row.idea_id not in selected_ids
        ],
        selected_program_count=len({row.same_program_key for row in selected}),
        selected_action_counts=dict(sorted(Counter(row.preferred_action for row in selected).items())),
        selected_priority_counts=dict(sorted(Counter(row.search_priority for row in selected).items())),
        selected_pair_relation_counts=dict(
            sorted(
                Counter(
                    relation_by_pair.get(
                        _program_pair_key(left.idea_id, right.idea_id),
                        "DISTINCT_PROGRAM",
                    )
                    for i, left in enumerate(selected)
                    for right in selected[i + 1 :]
                ).items()
            )
        ),
    )
    body = provisional.model_dump(mode="json")
    body.pop("schedule_id", None)
    body.pop("schedule_sha256", None)
    digest = _sha(body)
    return provisional.model_copy(
        update={
            "schedule_id": f"bounded_research_idea_parent_schedule_v3_3_2:g{pressure_report.generation_index}:{digest[:20]}",
            "schedule_sha256": digest,
        }
    )


__all__ = [
    "PressureLevel",
    "SearchPriority",
    "PreferredSearchAction",
    "NoveltyShape",
    "ResearchIdeaNoveltyDepthSignal",
    "ResearchIdeaSearchPressure",
    "ResearchIdeaSearchPressureReport",
    "ScheduledResearchIdeaParent",
    "BoundedResearchIdeaParentSchedule",
    "build_novelty_depth_signals_by_idea",
    "build_research_idea_search_pressure_report",
    "build_bounded_parent_schedule",
]
