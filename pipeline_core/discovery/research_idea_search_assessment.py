from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_family_calibration import (
    FamilyCalibrationReport,
    build_family_calibration,
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


NoveltySearchState = Literal[
    "NOT_ASSESSED",
    "UNRESOLVED",
    "CONFLICT",
    "SATURATED",
    "PARTIAL",
    "RESIDUAL",
    "STRONG_RESIDUE",
]

PersistenceDecision = Literal["RETAIN", "ARCHIVE"]
SearchAction = Literal["NONE", "REALIZE", "PROBE", "EVOLVE", "DIVERSIFY"]


class ResearchIdeaNoveltySignal(StrictModel):
    idea_id: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    external_status_counts: dict[str, int] = Field(default_factory=dict)
    claim_prior_art_status_counts: dict[str, int] = Field(default_factory=dict)
    novelty_state: NoveltySearchState
    direct_prior_art_claim_count: int = Field(ge=0)
    partial_prior_art_claim_count: int = Field(ge=0)
    residual_claim_count: int = Field(ge=0)
    unresolved_claim_count: int = Field(ge=0)
    source_mode: Literal[
        "NONE",
        "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
        "CONTEMPORANEOUS_SEARCH_PROBE",
    ] = "NONE"

    prior_art_only_not_positive_premise: Literal[True] = True
    search_feedback_only: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_certification_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ResearchIdeaSearchAssessmentItem(StrictModel):
    idea_id: str = Field(min_length=1)
    idea_birth_generation_index: int = Field(ge=0)
    cycle_generation_index: int = Field(ge=2)
    remains_active: bool

    realization_count: int = Field(ge=0)
    usable_grounded_realization_count: int = Field(ge=0)
    failed_or_abstained_count: int = Field(ge=0)
    not_operationalizable_count: int = Field(ge=0)
    local_search_budget: int = Field(ge=0)
    local_search_budget_used: int = Field(ge=0)
    local_search_exhausted: bool

    active_maturity_counts: dict[str, int] = Field(default_factory=dict)
    active_grounded_member_count: int = Field(ge=0)
    active_non_grounded_member_count: int = Field(ge=0)
    speculative_member_count: int = Field(ge=0)

    epistemic_debt_ids: list[str] = Field(default_factory=list)
    epistemic_debt_kinds: list[str] = Field(default_factory=list)
    acquisition_escalation_eligible_debt_count: int = Field(ge=0)

    program_family_key: str = Field(min_length=1)
    program_family_size: int = Field(ge=1)
    tight_neighborhood_key: str = Field(min_length=1)
    tight_neighborhood_size: int = Field(ge=1)
    family_pressure: Literal["ISOLATED", "REPRESENTED", "DENSE"]

    differential_prediction_present: bool
    falsification_condition_present: bool
    discriminating_observation_present: bool
    discrimination_sketch_complete: bool

    novelty: ResearchIdeaNoveltySignal

    grounding_is_claim_boundary_not_search_completion: Literal[True] = True
    family_is_soft_search_context_not_identity: Literal[True] = True
    prior_art_is_search_feedback_not_positive_premise: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ResearchIdeaSearchDecision(StrictModel):
    decision_id: str
    idea_id: str
    idea_birth_generation_index: int = Field(ge=0)
    cycle_generation_index: int = Field(ge=2)
    persistence: PersistenceDecision
    search_action: SearchAction
    search_complete_candidate: bool
    recommended_channels: list[Literal["TRANSFORM", "EXPLORE", "WILDCARD"]] = Field(
        default_factory=list
    )
    nonbinding_operator_hints: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    persistence_is_independent_from_search_action: Literal[True] = True
    grounded_does_not_imply_search_complete: Literal[True] = True
    retained_does_not_imply_nonfertile: Literal[True] = True
    shadow_only: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_certification_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_semantics(self) -> "ResearchIdeaSearchDecision":
        if self.persistence == "ARCHIVE" and self.search_action != "NONE":
            raise ValueError("ARCHIVE decision must use search_action=NONE")
        if self.search_complete_candidate != (
            self.persistence == "RETAIN" and self.search_action == "NONE"
        ):
            raise ValueError("search_complete_candidate / decision mismatch")
        return self


class ResearchIdeaSearchAssessmentReport(StrictModel):
    schema_version: Literal[
        "sis-v3-3-research-idea-search-assessment-shadow-v1"
    ] = "sis-v3-3-research-idea-search-assessment-shadow-v1"

    report_id: str
    report_sha256: str
    generation_index: int = Field(ge=2)
    source_parallel_report_id: str
    source_lifecycle_report_id: str
    source_family_calibration_report_id: str
    novelty_source_mode: Literal[
        "NONE",
        "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
        "CONTEMPORANEOUS_SEARCH_PROBE",
    ]

    assessments: list[ResearchIdeaSearchAssessmentItem] = Field(default_factory=list)
    decisions: list[ResearchIdeaSearchDecision] = Field(default_factory=list)
    assessment_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    persistence_counts: dict[str, int] = Field(default_factory=dict)
    search_action_counts: dict[str, int] = Field(default_factory=dict)
    novelty_state_counts: dict[str, int] = Field(default_factory=dict)
    family_pressure_counts: dict[str, int] = Field(default_factory=dict)

    old_fertility_behavior_changed: Literal[False] = False
    child_generation_executed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    external_prior_art_promoted_to_positive_premise: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_certification_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "ResearchIdeaSearchAssessmentReport":
        if self.assessment_count != len(self.assessments):
            raise ValueError("assessment_count mismatch")
        if self.decision_count != len(self.decisions):
            raise ValueError("decision_count mismatch")
        if {x.idea_id for x in self.assessments} != {x.idea_id for x in self.decisions}:
            raise ValueError("assessment/decision idea sets differ")
        return self


_NON_GROUNDED = {
    "IDEA_ONLY",
    "PARTIALLY_GROUNDED",
    "EVIDENCE_SEEKING",
    "SPECULATIVE_BUT_FALSIFIABLE",
}


def _novelty_state_from_counts(
    *,
    external_status_counts: Mapping[str, int],
    claim_status_counts: Mapping[str, int],
) -> NoveltySearchState:
    external = Counter(external_status_counts)
    claims = Counter(claim_status_counts)

    if not external and not claims:
        return "NOT_ASSESSED"
    if external.get("CONFLICTING_PRIOR_ART", 0) or claims.get("CONFLICTING_PRIOR_ART", 0):
        return "CONFLICT"
    if external.get("PLAUSIBLY_NOVEL", 0):
        return "STRONG_RESIDUE"
    if (
        external.get("NEW_COMBINATION_OF_KNOWN_EFFECTS", 0)
        or external.get("KNOWN_COMPONENTS_WITH_RELATIONAL_GAP", 0)
        or claims.get("COMPONENTS_ONLY", 0)
        or claims.get("NO_DIRECT_MATCH_FOUND", 0)
    ):
        return "RESIDUAL"
    if (
        external.get("LITERATURE_SUPPORTED_EXTENSION", 0)
        or claims.get("PARTIAL_PRIOR_ART", 0)
    ):
        return "PARTIAL"
    if (
        external.get("INSUFFICIENT_SEARCH_EVIDENCE", 0)
        or claims.get("INSUFFICIENT_METADATA", 0)
        or claims.get("TITLE_ONLY_NEIGHBORS", 0)
    ):
        return "UNRESOLVED"
    if external and set(external) <= {"WELL_ESTABLISHED"}:
        return "SATURATED"
    if claims and set(claims) <= {"DIRECT_PRIOR_ART"}:
        return "SATURATED"
    return "UNRESOLVED"


def build_terminal_novelty_signals_by_idea(
    *,
    terminal_parallel_report: Any,
    external_novelty_report: Any,
    source_mode: Literal[
        "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
        "CONTEMPORANEOUS_SEARCH_PROBE",
    ] = "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
) -> dict[str, ResearchIdeaNoveltySignal]:
    """Bind terminal hypothesis prior-art results back to ResearchIdea IDs.

    In v3.3 replay this is explicitly retrospective hindsight. It is diagnostic
    search feedback only and must not be interpreted as evidence that the original
    v3.1 fertility decision had access to these results.
    """

    hypothesis_ids_by_idea: dict[str, list[str]] = defaultdict(list)
    for member in (_get(terminal_parallel_report, "members", []) or []):
        idea_id = str(_get(member, "idea_id", "") or "").strip()
        hypothesis_id = str(_get(member, "hypothesis_id", "") or "").strip()
        if idea_id and hypothesis_id:
            hypothesis_ids_by_idea[idea_id].append(hypothesis_id)

    cards_by_hypothesis = {
        str(_get(card, "hypothesis_id")): card
        for card in (_get(external_novelty_report, "cards", []) or [])
    }

    result: dict[str, ResearchIdeaNoveltySignal] = {}
    for idea_id, hypothesis_ids in hypothesis_ids_by_idea.items():
        external_counts: Counter[str] = Counter()
        claim_counts: Counter[str] = Counter()
        matched_ids: list[str] = []
        for hypothesis_id in _dedupe(hypothesis_ids):
            card = cards_by_hypothesis.get(hypothesis_id)
            if card is None:
                continue
            matched_ids.append(hypothesis_id)
            external_counts[str(_get(card, "status", ""))] += 1
            for claim in (_get(card, "claim_reviews", []) or []):
                claim_counts[str(_get(claim, "status", ""))] += 1

        state = _novelty_state_from_counts(
            external_status_counts=external_counts,
            claim_status_counts=claim_counts,
        )
        result[idea_id] = ResearchIdeaNoveltySignal(
            idea_id=idea_id,
            hypothesis_ids=matched_ids,
            external_status_counts=dict(sorted(external_counts.items())),
            claim_prior_art_status_counts=dict(sorted(claim_counts.items())),
            novelty_state=state,
            direct_prior_art_claim_count=claim_counts.get("DIRECT_PRIOR_ART", 0),
            partial_prior_art_claim_count=claim_counts.get("PARTIAL_PRIOR_ART", 0),
            residual_claim_count=(
                claim_counts.get("COMPONENTS_ONLY", 0)
                + claim_counts.get("NO_DIRECT_MATCH_FOUND", 0)
            ),
            unresolved_claim_count=(
                claim_counts.get("INSUFFICIENT_METADATA", 0)
                + claim_counts.get("TITLE_ONLY_NEIGHBORS", 0)
            ),
            source_mode=source_mode,
        )
    return result


def _empty_novelty(idea_id: str) -> ResearchIdeaNoveltySignal:
    return ResearchIdeaNoveltySignal(
        idea_id=idea_id,
        novelty_state="NOT_ASSESSED",
        direct_prior_art_claim_count=0,
        partial_prior_art_claim_count=0,
        residual_claim_count=0,
        unresolved_claim_count=0,
        source_mode="NONE",
    )


def _decision_for(
    row: ResearchIdeaSearchAssessmentItem,
) -> ResearchIdeaSearchDecision:
    reasons: list[str] = []
    channels: list[Literal["TRANSFORM", "EXPLORE", "WILDCARD"]] = []
    hints: list[str] = []

    if not row.remains_active:
        persistence: PersistenceDecision = "ARCHIVE"
        action: SearchAction = "NONE"
        reasons.append("IDEA_NOT_IN_ACTIVE_SEARCH_POPULATION")
    else:
        persistence = "RETAIN"
        novelty = row.novelty.novelty_state

        if row.realization_count == 0:
            action = "REALIZE"
            reasons.append("NO_REALIZATION_ATTEMPT_RECORDED")
        elif row.usable_grounded_realization_count == 0 and not row.local_search_exhausted:
            action = "REALIZE"
            reasons.extend(
                [
                    "NO_USABLE_GROUNDED_REALIZATION_YET",
                    "LOCAL_REALIZATION_SPACE_REMAINS_OPEN",
                ]
            )
        elif novelty == "NOT_ASSESSED":
            action = "PROBE"
            reasons.extend(
                [
                    "PRIOR_ART_STATE_NOT_ASSESSED",
                    "GROUNDED_OR_REALIZABLE_DOES_NOT_IMPLY_SEARCH_COMPLETE",
                ]
            )
        elif novelty in {"UNRESOLVED", "CONFLICT"}:
            action = "PROBE"
            reasons.append(
                "PRIOR_ART_STATE_REQUIRES_RESOLUTION_BEFORE_SEARCH_COMPLETION"
            )
        elif novelty == "SATURATED":
            action = "DIVERSIFY" if row.program_family_size >= 2 else "EVOLVE"
            channels = ["EXPLORE", "WILDCARD"] if action == "DIVERSIFY" else ["TRANSFORM", "EXPLORE"]
            hints = ["LATENT_VARIABLE", "REGIME_BOUNDARY", "PROXY_CHALLENGE", "BACKBONE_MUTATION"]
            reasons.extend(
                [
                    "CURRENT_PROGRAM_PRIOR_ART_SATURATED",
                    "SEARCH_SHOULD_ESCAPE_KNOWN_PROGRAM_INSTEAD_OF_HOLDING_BECAUSE_GROUNDED",
                ]
            )
        elif novelty in {"PARTIAL", "RESIDUAL"}:
            if row.family_pressure == "DENSE":
                action = "DIVERSIFY"
                channels = ["EXPLORE", "WILDCARD"]
                hints = ["LATENT_VARIABLE", "REGIME_BOUNDARY", "PROXY_CHALLENGE", "CROSS_SOURCE_BRIDGE"]
                reasons.extend(
                    [
                        "NOVELTY_RESIDUE_REMAINS",
                        "CURRENT_PROGRAM_FAMILY_IS_DENSE",
                        "DIVERSIFY_BEYOND_LOCAL_VARIANTS",
                    ]
                )
            else:
                action = "EVOLVE"
                channels = ["TRANSFORM", "EXPLORE"]
                hints = ["BACKBONE_MUTATION", "LATENT_VARIABLE", "REGIME_BOUNDARY", "PROXY_CHALLENGE"]
                reasons.extend(
                    [
                        "NOVELTY_RESIDUE_REMAINS",
                        "EVOLVE_AROUND_UNRESOLVED_RELATION_NOT_GROUNDING_STATUS",
                    ]
                )
        elif novelty == "STRONG_RESIDUE":
            if (
                row.usable_grounded_realization_count > 0
                and row.discrimination_sketch_complete
                and not row.epistemic_debt_ids
                and row.family_pressure != "DENSE"
            ):
                action = "NONE"
                reasons.extend(
                    [
                        "SEARCH_BOUNDED_STRONG_RESIDUE_PRESENT",
                        "GROUNDED_REALIZATION_AVAILABLE",
                        "DISCRIMINATION_SKETCH_COMPLETE",
                        "NO_PERSISTENT_EPISTEMIC_DEBT",
                        "SEARCH_COMPLETE_CANDIDATE_ONLY_NOT_CERTIFIED",
                    ]
                )
            else:
                action = "DIVERSIFY" if row.family_pressure == "DENSE" else "EVOLVE"
                channels = ["EXPLORE", "WILDCARD"] if action == "DIVERSIFY" else ["TRANSFORM", "EXPLORE"]
                hints = ["REGIME_BOUNDARY", "LATENT_VARIABLE", "PROXY_CHALLENGE"]
                reasons.extend(
                    [
                        "STRONG_RESIDUE_EXISTS_BUT_SEARCH_COMPLETION_CONDITIONS_NOT_MET",
                        "PRESERVE_PARENT_WHILE_EXPLORING_ADDITIONAL_SCIENTIFIC_UPSIDE",
                    ]
                )
        else:  # pragma: no cover - Literal exhaustiveness fallback
            action = "PROBE"
            reasons.append("UNHANDLED_SEARCH_STATE_CONSERVATIVE_PROBE")

        if row.acquisition_escalation_eligible_debt_count:
            reasons.append("PERSISTENT_EPISTEMIC_DEBT_RECORDED_NONBLOCKING")
        if row.speculative_member_count:
            reasons.append("SPECULATIVE_FALSIFIABLE_REALIZATION_REMAINS_SEARCH_RELEVANT")

    decision_id = _stable_id(
        "research_idea_search_decision",
        row.idea_id,
        row.cycle_generation_index,
        persistence,
        action,
        row.novelty.novelty_state,
        row.program_family_key,
    )
    return ResearchIdeaSearchDecision(
        decision_id=decision_id,
        idea_id=row.idea_id,
        idea_birth_generation_index=row.idea_birth_generation_index,
        cycle_generation_index=row.cycle_generation_index,
        persistence=persistence,
        search_action=action,
        search_complete_candidate=(persistence == "RETAIN" and action == "NONE"),
        recommended_channels=_dedupe(channels),
        nonbinding_operator_hints=_dedupe(hints),
        reason_codes=_dedupe(reasons),
    )


def build_research_idea_search_assessment(
    *,
    parallel_report: Any,
    lifecycle: Any,
    population_nodes: Sequence[ResearchIdeaNode],
    novelty_signals_by_idea: Mapping[str, ResearchIdeaNoveltySignal] | None = None,
    family_calibration: FamilyCalibrationReport | None = None,
    program_similarity_floor: float = 0.58,
    dense_family_size: int = 3,
) -> ResearchIdeaSearchAssessmentReport:
    if dense_family_size < 2:
        raise ValueError("dense_family_size must be >= 2")

    novelty_signals_by_idea = dict(novelty_signals_by_idea or {})
    if family_calibration is None:
        family_calibration = build_family_calibration(
            population_nodes,
            program_similarity_floor=program_similarity_floor,
        )

    node_by_id = {row.idea_id: row for row in population_nodes}
    family_by_id = {row.idea_id: row for row in family_calibration.assignments}
    local_by_id = {
        str(_get(row, "idea_id")): row
        for row in (_get(lifecycle, "local_states", []) or [])
    }
    members_by_idea: dict[str, list[Any]] = defaultdict(list)
    for member in (_get(parallel_report, "members", []) or []):
        members_by_idea[str(_get(member, "idea_id"))].append(member)
    debts_by_idea: dict[str, list[Any]] = defaultdict(list)
    for debt in (_get(parallel_report, "epistemic_debts", []) or []):
        debts_by_idea[str(_get(debt, "idea_id"))].append(debt)

    generation_index = int(_get(lifecycle, "generation_index", 0) or 0)
    assessments: list[ResearchIdeaSearchAssessmentItem] = []
    decisions: list[ResearchIdeaSearchDecision] = []

    for state in (_get(parallel_report, "idea_states", []) or []):
        idea_id = str(_get(state, "idea_id", "") or "")
        node = node_by_id.get(idea_id)
        family = family_by_id.get(idea_id)
        if node is None:
            raise ValueError(f"parallel search references unknown ResearchIdea: {idea_id}")
        if family is None:
            raise ValueError(f"family calibration missing ResearchIdea: {idea_id}")

        members = members_by_idea.get(idea_id, [])
        active_ids = set(_get(state, "active_member_ids", []) or [])
        active_members = [
            row
            for row in members
            if str(_get(row, "member_id")) in active_ids
        ]
        maturity_counts = Counter(
            str(_get(row, "epistemic_maturity", "UNKNOWN"))
            for row in active_members
        )
        grounded_count = (
            maturity_counts.get("STRICT_GROUNDED", 0)
            + maturity_counts.get("OPERATIONAL_GROUNDED", 0)
        )
        non_grounded_count = sum(maturity_counts.get(value, 0) for value in _NON_GROUNDED)
        speculative_count = maturity_counts.get("SPECULATIVE_BUT_FALSIFIABLE", 0)

        local = local_by_id.get(idea_id)
        debts = debts_by_idea.get(idea_id, [])
        debt_kinds = _dedupe(
            [str(_get(row, "requirement_kind", "")) for row in debts]
        )
        acquisition_count = sum(
            str(_get(row, "disposition", "")) == "ACQUISITION_ESCALATION_ELIGIBLE"
            for row in debts
        )

        if family.program_family_size >= dense_family_size:
            pressure: Literal["ISOLATED", "REPRESENTED", "DENSE"] = "DENSE"
        elif family.program_family_size == 1:
            pressure = "ISOLATED"
        else:
            pressure = "REPRESENTED"

        novelty = novelty_signals_by_idea.get(idea_id, _empty_novelty(idea_id))
        diff = bool(str(node.differential_prediction or "").strip())
        falsifier = bool(str(node.falsification_condition or "").strip())
        discriminator = bool(str(node.discriminating_observation or "").strip())

        assessment = ResearchIdeaSearchAssessmentItem(
            idea_id=idea_id,
            idea_birth_generation_index=node.generation_index,
            cycle_generation_index=generation_index,
            remains_active=bool(_get(state, "remains_in_search_population", False)),
            realization_count=int(_get(local, "realization_count", 0) or 0),
            usable_grounded_realization_count=int(
                _get(local, "usable_grounded_realization_count", 0) or 0
            ),
            failed_or_abstained_count=int(
                _get(local, "failed_or_abstained_count", 0) or 0
            ),
            not_operationalizable_count=int(
                _get(local, "not_operationalizable_count", 0) or 0
            ),
            local_search_budget=int(_get(local, "local_search_budget", 0) or 0),
            local_search_budget_used=int(
                _get(local, "local_search_budget_used", 0) or 0
            ),
            local_search_exhausted=bool(
                _get(local, "local_search_exhausted", False)
            ),
            active_maturity_counts=dict(sorted(maturity_counts.items())),
            active_grounded_member_count=grounded_count,
            active_non_grounded_member_count=non_grounded_count,
            speculative_member_count=speculative_count,
            epistemic_debt_ids=[str(_get(row, "debt_id")) for row in debts],
            epistemic_debt_kinds=debt_kinds,
            acquisition_escalation_eligible_debt_count=acquisition_count,
            program_family_key=family.program_family_key,
            program_family_size=family.program_family_size,
            tight_neighborhood_key=family.tight_neighborhood_key,
            tight_neighborhood_size=family.tight_neighborhood_size,
            family_pressure=pressure,
            differential_prediction_present=diff,
            falsification_condition_present=falsifier,
            discriminating_observation_present=discriminator,
            discrimination_sketch_complete=diff and falsifier and discriminator,
            novelty=novelty,
        )
        assessments.append(assessment)
        decisions.append(_decision_for(assessment))

    novelty_modes = {
        row.novelty.source_mode for row in assessments if row.novelty.source_mode != "NONE"
    }
    if not novelty_modes:
        novelty_source_mode = "NONE"
    elif novelty_modes == {"RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY"}:
        novelty_source_mode = "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY"
    elif novelty_modes == {"CONTEMPORANEOUS_SEARCH_PROBE"}:
        novelty_source_mode = "CONTEMPORANEOUS_SEARCH_PROBE"
    else:
        raise ValueError("mixed novelty source modes are not supported in one report")

    provisional = ResearchIdeaSearchAssessmentReport(
        report_id="pending",
        report_sha256="pending",
        generation_index=generation_index,
        source_parallel_report_id=str(_get(parallel_report, "report_id", "unknown")),
        source_lifecycle_report_id=str(_get(lifecycle, "report_id", "unknown")),
        source_family_calibration_report_id=family_calibration.report_id,
        novelty_source_mode=novelty_source_mode,
        assessments=assessments,
        decisions=decisions,
        assessment_count=len(assessments),
        decision_count=len(decisions),
        persistence_counts=dict(
            sorted(Counter(row.persistence for row in decisions).items())
        ),
        search_action_counts=dict(
            sorted(Counter(row.search_action for row in decisions).items())
        ),
        novelty_state_counts=dict(
            sorted(Counter(row.novelty.novelty_state for row in assessments).items())
        ),
        family_pressure_counts=dict(
            sorted(Counter(row.family_pressure for row in assessments).items())
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"research_idea_search_assessment:g{generation_index}:{digest[:20]}",
            "report_sha256": digest,
        }
    )


__all__ = [
    "NoveltySearchState",
    "PersistenceDecision",
    "SearchAction",
    "ResearchIdeaNoveltySignal",
    "ResearchIdeaSearchAssessmentItem",
    "ResearchIdeaSearchDecision",
    "ResearchIdeaSearchAssessmentReport",
    "build_terminal_novelty_signals_by_idea",
    "build_research_idea_search_assessment",
]
