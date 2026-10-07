from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


_TOKEN_RE = re.compile(r"[A-Za-z0-9α-ωΑ-Ω가-힣]+")
_GENERIC_IDEA_TOKENS = {
    "effect",
    "effects",
    "influence",
    "influences",
    "relationship",
    "relation",
    "mechanism",
    "mechanisms",
    "study",
    "test",
    "whether",
    "under",
    "between",
    "across",
    "response",
    "system",
}


def _idea_tokens(node: Any) -> set[str]:
    kernel = getattr(node, "kernel", None)
    if kernel is None:
        return set()
    values = [
        getattr(kernel, "canonical_intent", ""),
        *list(getattr(kernel, "core_scientific_commitments", []) or []),
        *list(getattr(kernel, "scope_commitments", []) or []),
        *list(getattr(kernel, "contrastive_commitments", []) or []),
        getattr(kernel, "question_commitment", "") or "",
    ]
    tokens = {
        token.casefold()
        for token in _TOKEN_RE.findall(" ".join(str(value) for value in values))
    }
    return {token for token in tokens if token not in _GENERIC_IDEA_TOKENS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _conceptual_similarity(left: Any, right: Any) -> float:
    return _jaccard(_idea_tokens(left), _idea_tokens(right))


EvolutionChannel = Literal["TRANSFORM", "EXPLORE", "WILDCARD"]
SourceMode = Literal[
    "SPECULATIVE",
    "EVIDENCE_SEEKING",
    "PARTIALLY_GROUNDED",
    "GROUNDED_MIXED",
    "GROUNDED",
]
SemanticDisposition = Literal[
    "GENUINE_CHILD",
    "INDETERMINATE_PROBE",
    "SAME_IDEA_REFINEMENT",
    "EXACT_KERNEL_DUPLICATE_SUPPRESSED",
]

_ALLOWED_BY_CHANNEL: dict[EvolutionChannel, tuple[str, ...]] = {
    "TRANSFORM": (
        "AXIS_MUTATION",
        "BACKBONE_MUTATION",
        "CANDIDATE_INTERPRETATION",
        "LATENT_VARIABLE",
        "REGIME_BOUNDARY",
        "PROXY_CHALLENGE",
        "CROSS_SOURCE_BRIDGE",
    ),
    "EXPLORE": (
        "CROSS_SOURCE_BRIDGE",
        "AXIS_MUTATION",
        "REGIME_BOUNDARY",
        "LATENT_VARIABLE",
        "CANDIDATE_INTERPRETATION",
        "BACKBONE_MUTATION",
        "PROXY_CHALLENGE",
    ),
    "WILDCARD": (
        "AXIS_MUTATION",
        "CANDIDATE_INTERPRETATION",
        "REGIME_BOUNDARY",
        "LATENT_VARIABLE",
        "PROXY_CHALLENGE",
        "CROSS_SOURCE_BRIDGE",
        "BACKBONE_MUTATION",
    ),
}

_NON_GROUNDED_MATURITIES = {
    "IDEA_ONLY",
    "PARTIALLY_GROUNDED",
    "EVIDENCE_SEEKING",
    "SPECULATIVE_BUT_FALSIFIABLE",
}


class EpistemicEvolutionParentCandidate(StrictModel):
    schema_version: Literal[
        "sis-v2-9-epistemic-evolution-parent-candidate-v1"
    ] = "sis-v2-9-epistemic-evolution-parent-candidate-v1"
    idea_id: str
    handoff_id: str
    generation_index: int = Field(ge=0)
    source_member_ids: list[str] = Field(default_factory=list)
    source_epistemic_realization_ids: list[str] = Field(default_factory=list)
    source_maturities: list[str] = Field(default_factory=list)
    source_mode: SourceMode
    source_non_grounded_member_count: int = Field(ge=0)
    source_speculative_member_count: int = Field(ge=0)
    source_evidence_seeking_member_count: int = Field(ge=0)
    source_grounded_member_count: int = Field(ge=0)
    epistemic_debt_ids: list[str] = Field(default_factory=list)
    epistemic_debt_kinds: list[str] = Field(default_factory=list)
    recommended_channels: list[EvolutionChannel] = Field(default_factory=list)
    nonbinding_operator_hints: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    selected_for_g4: bool = False
    selected_channel: EvolutionChannel | None = None
    selection_reason_codes: list[str] = Field(default_factory=list)
    grounding_required_for_parenting: Literal[False] = False
    source_feedback_is_not_positive_premise: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class EpistemicG4GenerationTask(StrictModel):
    schema_version: Literal[
        "sis-v2-9-epistemic-g4-generation-task-v1"
    ] = "sis-v2-9-epistemic-g4-generation-task-v1"
    task_id: str
    handoff_id: str
    generation_index: int = Field(default=4, ge=2)
    primary_parent_idea_id: str
    eligible_secondary_parent_idea_ids: list[str] = Field(default_factory=list)
    channel: EvolutionChannel
    allowed_operator_ids: list[str] = Field(min_length=1)
    operator_hints: list[str] = Field(default_factory=list)
    max_output_count: int = Field(default=1, ge=1, le=2)
    source_member_ids: list[str] = Field(default_factory=list)
    source_epistemic_realization_ids: list[str] = Field(default_factory=list)
    source_maturities: list[str] = Field(default_factory=list)
    source_epistemic_debt_ids: list[str] = Field(default_factory=list)
    source_epistemic_debt_kinds: list[str] = Field(default_factory=list)
    source_reason_codes: list[str] = Field(default_factory=list)
    source_feedback_is_search_context_only: Literal[True] = True
    grounding_is_not_precondition_for_generation: Literal[True] = True
    generated_child_is_inspiration_only_until_realized: Literal[True] = True

    @model_validator(mode="after")
    def _validate_task(self) -> "EpistemicG4GenerationTask":
        self.eligible_secondary_parent_idea_ids = _dedupe(
            self.eligible_secondary_parent_idea_ids
        )
        self.allowed_operator_ids = _dedupe(self.allowed_operator_ids)
        self.operator_hints = _dedupe(self.operator_hints)
        self.source_member_ids = _dedupe(self.source_member_ids)
        self.source_epistemic_realization_ids = _dedupe(
            self.source_epistemic_realization_ids
        )
        self.source_epistemic_debt_ids = _dedupe(self.source_epistemic_debt_ids)
        self.source_epistemic_debt_kinds = _dedupe(self.source_epistemic_debt_kinds)
        self.source_reason_codes = _dedupe(self.source_reason_codes)
        if self.primary_parent_idea_id in set(self.eligible_secondary_parent_idea_ids):
            raise ValueError("primary parent cannot be its own secondary parent")
        allowed = set(_ALLOWED_BY_CHANNEL[self.channel])
        if not set(self.allowed_operator_ids).issubset(allowed):
            raise ValueError("task contains operator not allowed by channel")
        return self


class EpistemicG4GenerationPlan(StrictModel):
    schema_version: Literal[
        "sis-v2-9-epistemic-g4-generation-plan-v1"
    ] = "sis-v2-9-epistemic-g4-generation-plan-v1"
    plan_id: str
    source_parallel_report_id: str
    generation_index: int = Field(default=4, ge=2)
    available_handoff_count: int = Field(ge=0)
    candidate_parent_count: int = Field(ge=0)
    max_parent_count: int = Field(ge=1)
    selected_parent_idea_ids: list[str] = Field(default_factory=list)
    parent_candidates: list[EpistemicEvolutionParentCandidate] = Field(default_factory=list)
    tasks: list[EpistemicG4GenerationTask] = Field(default_factory=list)
    task_count: int = Field(ge=0)
    planned_max_offspring: int = Field(ge=0)
    selected_parent_source_mode_counts: dict[str, int] = Field(default_factory=dict)
    task_count_by_channel: dict[str, int] = Field(default_factory=dict)
    deterministic_bounded_parent_selection: Literal[True] = True
    single_scalar_fitness_used: Literal[False] = False
    grounding_required_for_parent_selection: Literal[False] = False
    evidence_acquisition_required_before_generation: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "EpistemicG4GenerationPlan":
        if self.candidate_parent_count != len(self.parent_candidates):
            raise ValueError("candidate_parent_count mismatch")
        if self.task_count != len(self.tasks):
            raise ValueError("task_count mismatch")
        if len(self.selected_parent_idea_ids) != self.task_count:
            raise ValueError("selected parent count / task count mismatch")
        if self.planned_max_offspring != sum(row.max_output_count for row in self.tasks):
            raise ValueError("planned_max_offspring mismatch")
        if self.task_count > self.max_parent_count:
            raise ValueError("task_count exceeds max_parent_count")
        if any(row.generation_index != self.generation_index for row in self.tasks):
            raise ValueError("task generation_index mismatch")
        return self


@dataclass(frozen=True)
class EpistemicG4Prompt:
    task_id: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


class EpistemicOffspringCandidateDraft(StrictModel):
    local_id: str
    chosen_operator_id: str
    secondary_parent_idea_id: str | None = None
    conceptual_change_summary: str
    kernel: dict[str, Any]
    differential_prediction: str = ""
    falsification_condition: str = ""
    discriminating_observation: str = ""
    task_relation_mode: Literal["DIRECT", "SUBORDINATE", "UNKNOWN"] = "UNKNOWN"


class EpistemicOffspringBatchDraft(StrictModel):
    task_id: str
    primary_parent_idea_id: str
    channel: str
    candidates: list[EpistemicOffspringCandidateDraft] = Field(default_factory=list)
    abstention_reason: str | None = None


@runtime_checkable
class EpistemicOffspringBackend(Protocol):
    def generate(self, prompt: EpistemicG4Prompt) -> Any: ...

    def repair(
        self,
        prompt: EpistemicG4Prompt,
        previous_draft: Any,
        feedback: str,
    ) -> Any: ...


class EpistemicG4GenerationRunRecord(StrictModel):
    task_id: str
    handoff_id: str
    generation_index: int = Field(default=4, ge=2)
    primary_parent_idea_id: str
    channel: EvolutionChannel
    decision: Literal[
        "GENERATED",
        "ABSTAINED",
        "GENERATION_FAILED",
        "REJECTED_INVALID_DRAFT",
    ]
    generated_idea_ids: list[str] = Field(default_factory=list)
    genuine_child_idea_ids: list[str] = Field(default_factory=list)
    indeterminate_probe_idea_ids: list[str] = Field(default_factory=list)
    same_idea_refinement_ids: list[str] = Field(default_factory=list)
    semantic_retry_count: int = Field(default=0, ge=0)
    compile_issue_codes: list[str] = Field(default_factory=list)
    generation_error: str | None = None
    llm_call_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class EpistemicG4SemanticRecord(StrictModel):
    idea_id: str
    task_id: str
    handoff_id: str
    generation_index: int = Field(default=4, ge=2)
    primary_parent_idea_id: str
    parent_idea_ids: list[str] = Field(default_factory=list)
    channel: EvolutionChannel
    chosen_operator_id: str
    identity_relation: Literal["SAME_IDEA", "DIFFERENT_IDEA", "INDETERMINATE"]
    disposition: SemanticDisposition
    retained_in_g4_population: bool
    conceptual_change_summary: str
    transition: dict[str, Any] = Field(default_factory=dict)
    diagnostic_codes: list[str] = Field(default_factory=list)
    source_member_ids: list[str] = Field(default_factory=list)
    source_maturities: list[str] = Field(default_factory=list)
    source_epistemic_debt_ids: list[str] = Field(default_factory=list)
    source_epistemic_debt_kinds: list[str] = Field(default_factory=list)
    source_contains_non_grounded_feedback: bool
    source_contains_speculative_feedback: bool
    source_contains_evidence_seeking_feedback: bool
    source_feedback_is_not_positive_premise: Literal[True] = True
    generated_idea_is_inspiration_only: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class EpistemicG4ExecutionReport(StrictModel):
    schema_version: Literal[
        "sis-v2-9-epistemic-to-generational-evolution-shadow-v1"
    ] = "sis-v2-9-epistemic-to-generational-evolution-shadow-v1"
    report_id: str
    report_sha256: str
    source_parallel_report_id: str
    source_plan_id: str
    generation_index: int = Field(default=4, ge=2)
    tasks: list[EpistemicG4GenerationTask] = Field(default_factory=list)
    run_records: list[EpistemicG4GenerationRunRecord] = Field(default_factory=list)
    offspring_nodes: list[Any] = Field(default_factory=list)
    semantic_records: list[EpistemicG4SemanticRecord] = Field(default_factory=list)
    g4_population_nodes: list[Any] = Field(default_factory=list)
    raw_offspring_count: int = Field(ge=0)
    g4_population_count: int = Field(ge=0)
    genuine_child_count: int = Field(ge=0)
    indeterminate_probe_count: int = Field(ge=0)
    same_idea_refinement_count: int = Field(ge=0)
    exact_kernel_duplicate_suppressed_count: int = Field(ge=0)
    identity_relation_counts: dict[str, int] = Field(default_factory=dict)
    disposition_counts: dict[str, int] = Field(default_factory=dict)
    generated_count_by_channel: dict[str, int] = Field(default_factory=dict)
    generated_count_by_operator: dict[str, int] = Field(default_factory=dict)
    genuine_child_from_non_grounded_feedback_count: int = Field(ge=0)
    genuine_child_from_speculative_feedback_count: int = Field(ge=0)
    genuine_child_from_evidence_seeking_feedback_count: int = Field(ge=0)
    llm_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    semantic_retry_count: int = Field(ge=0)
    offspring_generation_executed: bool = True
    carried_forward_idea_ids: list[str] = Field(default_factory=list)
    carried_forward_count: int = Field(default=0, ge=0)
    replaced_parent_idea_ids: list[str] = Field(default_factory=list)
    replaced_parent_count: int = Field(default=0, ge=0)
    population_composed_with_persistence: bool = False
    population_growth_budget: int = Field(default=0, ge=0)
    grounding_required_for_child_generation: Literal[False] = False
    grounding_required_before_scientific_claim: Literal[True] = True
    epistemic_feedback_used_as_search_context_only: Literal[True] = True
    evidence_acquisition_executed: Literal[False] = False
    external_probe_as_positive_premise: Literal[False] = False
    generated_children_are_inspiration_only: Literal[True] = True
    same_idea_is_routed_back_to_realization_lane: Literal[True] = True
    indeterminate_identity_is_retained_as_bounded_probe: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "EpistemicG4ExecutionReport":
        if self.raw_offspring_count != len(self.offspring_nodes):
            raise ValueError("raw_offspring_count mismatch")
        if self.raw_offspring_count != len(self.semantic_records):
            raise ValueError("semantic record count mismatch")
        if self.g4_population_count != len(self.g4_population_nodes):
            raise ValueError("g4_population_count mismatch")
        if self.genuine_child_count != sum(
            row.disposition == "GENUINE_CHILD" for row in self.semantic_records
        ):
            raise ValueError("genuine_child_count mismatch")
        if self.indeterminate_probe_count != sum(
            row.disposition == "INDETERMINATE_PROBE" for row in self.semantic_records
        ):
            raise ValueError("indeterminate_probe_count mismatch")
        if self.same_idea_refinement_count != sum(
            row.disposition == "SAME_IDEA_REFINEMENT" for row in self.semantic_records
        ):
            raise ValueError("same_idea_refinement_count mismatch")
        if self.carried_forward_count != len(self.carried_forward_idea_ids):
            raise ValueError("carried_forward_count mismatch")
        if self.replaced_parent_count != len(self.replaced_parent_idea_ids):
            raise ValueError("replaced_parent_count mismatch")
        if set(self.carried_forward_idea_ids) & set(self.replaced_parent_idea_ids):
            raise ValueError("carried-forward and replaced parent sets overlap")
        if any(row.generation_index != self.generation_index for row in self.tasks):
            raise ValueError("execution task generation_index mismatch")
        if any(row.generation_index != self.generation_index for row in self.run_records):
            raise ValueError("run record generation_index mismatch")
        if any(row.generation_index != self.generation_index for row in self.semantic_records):
            raise ValueError("semantic record generation_index mismatch")
        return self


def _source_mode(maturities: Sequence[str]) -> SourceMode:
    values = set(maturities)
    if "SPECULATIVE_BUT_FALSIFIABLE" in values:
        return "SPECULATIVE"
    if "EVIDENCE_SEEKING" in values:
        return "EVIDENCE_SEEKING"
    if "PARTIALLY_GROUNDED" in values:
        return "PARTIALLY_GROUNDED"
    grounded = values & {"STRICT_GROUNDED", "OPERATIONAL_GROUNDED"}
    if grounded and values - grounded:
        return "GROUNDED_MIXED"
    return "GROUNDED"


def _parent_candidates(
    *,
    parallel_report: Any,
    parent_by_id: Mapping[str, Any],
) -> list[EpistemicEvolutionParentCandidate]:
    member_by_id = {row.member_id: row for row in parallel_report.members}
    debts_by_idea: dict[str, list[Any]] = {}
    for debt in parallel_report.epistemic_debts:
        debts_by_idea.setdefault(debt.idea_id, []).append(debt)

    candidates: list[EpistemicEvolutionParentCandidate] = []
    for handoff in parallel_report.evolution_handoffs:
        if handoff.idea_id not in parent_by_id:
            continue
        members = [
            member_by_id[mid]
            for mid in handoff.source_member_ids
            if mid in member_by_id
        ]
        maturities = _dedupe([row.epistemic_maturity for row in members])
        debts = debts_by_idea.get(handoff.idea_id, [])
        non_grounded = sum(
            row.epistemic_maturity in _NON_GROUNDED_MATURITIES for row in members
        )
        speculative = sum(
            row.epistemic_maturity == "SPECULATIVE_BUT_FALSIFIABLE"
            for row in members
        )
        evidence_seeking = sum(
            row.epistemic_maturity == "EVIDENCE_SEEKING" for row in members
        )
        grounded = sum(
            row.epistemic_maturity in {"STRICT_GROUNDED", "OPERATIONAL_GROUNDED"}
            for row in members
        )
        channels = [
            channel
            for channel in handoff.recommended_channels
            if channel in _ALLOWED_BY_CHANNEL
        ]
        if not channels:
            channels = ["TRANSFORM"]
        candidates.append(
            EpistemicEvolutionParentCandidate(
                idea_id=handoff.idea_id,
                handoff_id=handoff.handoff_id,
                generation_index=handoff.generation_index,
                source_member_ids=list(handoff.source_member_ids),
                source_epistemic_realization_ids=list(
                    handoff.source_epistemic_realization_ids
                ),
                source_maturities=maturities,
                source_mode=_source_mode(maturities),
                source_non_grounded_member_count=non_grounded,
                source_speculative_member_count=speculative,
                source_evidence_seeking_member_count=evidence_seeking,
                source_grounded_member_count=grounded,
                epistemic_debt_ids=[row.debt_id for row in debts],
                epistemic_debt_kinds=_dedupe(
                    [row.requirement_kind for row in debts]
                ),
                recommended_channels=channels,
                nonbinding_operator_hints=list(handoff.nonbinding_operator_hints),
                reason_codes=list(handoff.reason_codes),
            )
        )
    return candidates


def _selection_score(
    candidate: EpistemicEvolutionParentCandidate,
    *,
    parent_by_id: Mapping[str, Any],
    selected: Sequence[EpistemicEvolutionParentCandidate],
    seen_modes: set[str],
    seen_debt_kinds: set[str],
) -> tuple[Any, ...]:
    parent = parent_by_id[candidate.idea_id]
    if selected:
        similarity = max(
            _conceptual_similarity(parent, parent_by_id[row.idea_id])
            for row in selected
        )
    else:
        similarity = 0.0
    conceptual_novelty = 1.0 - similarity
    new_debt_count = len(set(candidate.epistemic_debt_kinds) - seen_debt_kinds)
    mode_priority = {
        "SPECULATIVE": 4,
        "EVIDENCE_SEEKING": 3,
        "PARTIALLY_GROUNDED": 2,
        "GROUNDED_MIXED": 1,
        "GROUNDED": 0,
    }[candidate.source_mode]
    return (
        int(candidate.source_mode not in seen_modes),
        int(new_debt_count > 0),
        round(conceptual_novelty, 6),
        mode_priority,
        candidate.source_non_grounded_member_count,
        len(candidate.epistemic_debt_kinds),
        len(candidate.recommended_channels),
        candidate.generation_index,
        candidate.idea_id,
    )


def _choose_channel(
    candidate: EpistemicEvolutionParentCandidate,
    channel_counts: Counter[str],
) -> EvolutionChannel:
    preference = {"TRANSFORM": 0, "EXPLORE": 1, "WILDCARD": 2}
    channels = list(candidate.recommended_channels) or ["TRANSFORM"]
    return min(
        channels,
        key=lambda value: (channel_counts[value], preference[value], value),
    )


def build_epistemic_g4_generation_plan(
    *,
    parallel_report: Any,
    parent_by_id: Mapping[str, Any],
    max_parents: int = 4,
    max_outputs_per_parent: int = 1,
    generation_index: int = 4,
) -> EpistemicG4GenerationPlan:
    if generation_index < 2:
        raise ValueError("generation_index must be >= 2")
    if max_parents < 1:
        raise ValueError("max_parents must be >= 1")
    if max_outputs_per_parent < 1 or max_outputs_per_parent > 2:
        raise ValueError("max_outputs_per_parent must be 1 or 2")

    candidates = _parent_candidates(
        parallel_report=parallel_report,
        parent_by_id=parent_by_id,
    )
    selected: list[EpistemicEvolutionParentCandidate] = []
    seen_modes: set[str] = set()
    seen_debt_kinds: set[str] = set()
    remaining = list(candidates)
    while remaining and len(selected) < max_parents:
        best = max(
            remaining,
            key=lambda row: _selection_score(
                row,
                parent_by_id=parent_by_id,
                selected=selected,
                seen_modes=seen_modes,
                seen_debt_kinds=seen_debt_kinds,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        seen_modes.add(best.source_mode)
        seen_debt_kinds.update(best.epistemic_debt_kinds)

    selected_ids = [row.idea_id for row in selected]
    channel_counts: Counter[str] = Counter()
    tasks: list[EpistemicG4GenerationTask] = []
    selected_by_id = {row.idea_id: row for row in selected}
    marked_candidates: list[EpistemicEvolutionParentCandidate] = []
    for row in candidates:
        if row.idea_id not in selected_by_id:
            marked_candidates.append(row)
            continue
        channel = _choose_channel(row, channel_counts)
        channel_counts[channel] += 1
        allowed = list(_ALLOWED_BY_CHANNEL[channel])
        hints = [
            hint
            for hint in row.nonbinding_operator_hints
            if hint in set(allowed)
        ]
        if not hints:
            hints = allowed[:3]
        selection_reasons = [
            f"SOURCE_MODE_{row.source_mode}",
            "BOUNDED_EPISTEMIC_PARENT_SELECTION",
        ]
        if row.source_non_grounded_member_count:
            selection_reasons.append("NON_GROUNDED_FEEDBACK_REMAINS_REPRODUCTIVE")
        if row.epistemic_debt_ids:
            selection_reasons.append("EPISTEMIC_DEBT_USED_AS_SEARCH_FEEDBACK")
        marked_candidates.append(
            row.model_copy(
                update={
                    "selected_for_g4": True,
                    "selected_channel": channel,
                    "selection_reason_codes": selection_reasons,
                }
            )
        )
        parent = parent_by_id[row.idea_id]
        eligible_secondary = [
            other
            for other in selected_ids
            if other != row.idea_id
            and getattr(parent_by_id[other], "source_context_id", None)
            == getattr(parent, "source_context_id", None)
            and getattr(parent_by_id[other], "source_context_sha256", None)
            == getattr(parent, "source_context_sha256", None)
        ]
        tasks.append(
            EpistemicG4GenerationTask(
                task_id=_stable_id(
                    f"g{generation_index}_epistemic_offspring_task",
                    parallel_report.report_id,
                    row.handoff_id,
                    row.idea_id,
                    channel,
                ),
                handoff_id=row.handoff_id,
                generation_index=generation_index,
                primary_parent_idea_id=row.idea_id,
                eligible_secondary_parent_idea_ids=eligible_secondary,
                channel=channel,
                allowed_operator_ids=allowed,
                operator_hints=hints,
                max_output_count=max_outputs_per_parent,
                source_member_ids=row.source_member_ids,
                source_epistemic_realization_ids=row.source_epistemic_realization_ids,
                source_maturities=row.source_maturities,
                source_epistemic_debt_ids=row.epistemic_debt_ids,
                source_epistemic_debt_kinds=row.epistemic_debt_kinds,
                source_reason_codes=row.reason_codes,
            )
        )

    source_mode_counts = Counter(row.source_mode for row in selected)
    provisional = {
        "source_parallel_report_id": parallel_report.report_id,
        "generation_index": generation_index,
        "selected_parent_idea_ids": selected_ids,
        "tasks": [row.model_dump(mode="json") for row in tasks],
        "max_parents": max_parents,
    }
    return EpistemicG4GenerationPlan(
        plan_id=_stable_id(f"g{generation_index}_epistemic_generation_plan", provisional),
        source_parallel_report_id=parallel_report.report_id,
        generation_index=generation_index,
        available_handoff_count=len(parallel_report.evolution_handoffs),
        candidate_parent_count=len(marked_candidates),
        max_parent_count=max_parents,
        selected_parent_idea_ids=selected_ids,
        parent_candidates=marked_candidates,
        tasks=tasks,
        task_count=len(tasks),
        planned_max_offspring=sum(row.max_output_count for row in tasks),
        selected_parent_source_mode_counts=dict(sorted(source_mode_counts.items())),
        task_count_by_channel=dict(sorted(channel_counts.items())),
    )


def _kernel_payload(node: Any) -> dict[str, Any]:
    kernel = getattr(node, "kernel")
    return {
        "idea_id": getattr(node, "idea_id"),
        "generation_index": getattr(node, "generation_index"),
        "origin_kind": getattr(node, "origin_kind", None),
        "kernel": _dump(kernel),
        "task_relation_mode": getattr(node, "task_relation_mode", "UNKNOWN"),
        "differential_prediction": getattr(node, "differential_prediction", ""),
        "falsification_condition": getattr(node, "falsification_condition", ""),
        "discriminating_observation": getattr(node, "discriminating_observation", ""),
    }


def build_epistemic_g4_prompt(
    *,
    task: EpistemicG4GenerationTask,
    parent_by_id: Mapping[str, Any],
    parallel_report: Any,
    research_question: str,
) -> EpistemicG4Prompt:
    primary = parent_by_id[task.primary_parent_idea_id]
    member_by_id = {row.member_id: row for row in parallel_report.members}
    debt_by_id = {row.debt_id: row for row in parallel_report.epistemic_debts}
    member_context = []
    for member_id in task.source_member_ids:
        row = member_by_id.get(member_id)
        if row is None:
            continue
        member_context.append(
            {
                "member_id": row.member_id,
                "epistemic_maturity": row.epistemic_maturity,
                "grounding_coverage": row.grounding_coverage,
                "materialization_status": row.materialization_status,
                "prediction_present": row.prediction_present,
                "falsifier_present": row.falsifier_present,
                "missing_evidence_requirement_count": row.missing_evidence_requirement_count,
                "residual_epistemic_state": row.residual_epistemic_state,
                "prospective_identifiability": row.prospective_identifiability,
            }
        )
    debt_context = []
    for debt_id in task.source_epistemic_debt_ids:
        row = debt_by_id.get(debt_id)
        if row is None:
            continue
        debt_context.append(
            {
                "debt_id": row.debt_id,
                "requirement_kind": row.requirement_kind,
                "normalized_description": row.normalized_description,
                "occurrence_count": row.occurrence_count,
                "disposition": row.disposition,
            }
        )

    system = """You are the epistemic-to-generational ResearchIdea evolution engine in a scientific discovery system.

Generate a next-generation ResearchIdea child, not a grounded hypothesis and not a verified scientific claim.

Authority boundary:
- The parent ResearchIdea is INSPIRATION_ONLY.
- Partial/speculative realization summaries and epistemic-debt tickets are SEARCH FEEDBACK ONLY.
- They are not positive premises, empirical evidence, citations, novelty authority, or truth authority.
- Grounding is deliberately NOT a precondition for imagining or reproducing a ResearchIdea.
- Grounding WILL be required later before any generated child may become a scientific claim or grounded HypothesisCard.
- Do not invent citations, paper IDs, premise IDs, or claims of empirical support.
- Do not imply that an evidence gap was resolved merely because you generated a child.

Generation objective:
- Produce a scientifically coherent child that remains DIRECT or SUBORDINATE to the research task.
- TRANSFORM must make a substantive identity-bearing change to mechanism, mediator, regime, proxy assumption, causal structure, contrast, or scientific question.
- EXPLORE should move into an underexplored adjacent scientific region while preserving task relevance.
- WILDCARD may make a bounded higher-variance conceptual move while remaining falsifiable and scientifically coherent.
- Mere wording changes are failures.
- Operator hints are non-binding. Semantic identity is audited after generation.
- If the supplied epistemic feedback does not support a coherent conceptual move, abstain rather than fabricating evidence.

Return only the structured GenerationalOffspringBatchDraft requested by the caller.
"""
    secondary = [
        _kernel_payload(parent_by_id[idea_id])
        for idea_id in task.eligible_secondary_parent_idea_ids
        if idea_id in parent_by_id
    ]
    user_payload = {
        "task_id": task.task_id,
        "research_question": research_question,
        "channel": task.channel,
        "max_output_count": task.max_output_count,
        "primary_parent": _kernel_payload(primary),
        "eligible_secondary_parents": secondary,
        "epistemic_realization_feedback_search_context_only": member_context,
        "epistemic_debt_search_context_only": debt_context,
        "source_reason_codes_search_context_only": task.source_reason_codes,
        "operator_hints_nonbinding": task.operator_hints,
        "allowed_operator_ids": task.allowed_operator_ids,
        "output_rules": [
            "Return zero to max_output_count candidates.",
            "Use unique local_id values.",
            "Do not claim truth, novelty, literature support, or evidence closure.",
            "Do not turn epistemic-debt text into a positive premise.",
            "For non-CROSS_SOURCE_BRIDGE operators, secondary_parent_idea_id must be null.",
            "For CROSS_SOURCE_BRIDGE, use exactly one supplied eligible secondary parent.",
            "Each candidate should include a differential prediction and falsification condition when scientifically possible.",
        ],
    }
    user = json.dumps(user_payload, ensure_ascii=False, indent=2, sort_keys=True)
    return EpistemicG4Prompt(
        task_id=task.task_id,
        system_prompt=system,
        user_prompt=user,
        prompt_sha256=_sha({"system": system, "user": user}),
    )


def _normalize_generation(result: Any) -> tuple[EpistemicOffspringBatchDraft, int, int]:
    """Normalize either a native v2.9 draft or the reused v2.3 backend contract.

    SIS-v2.9 deliberately reuses ``InstructorOpenAICompatibleOffspringBackend``.
    That backend returns ``GenerationalOffspringBatchDraft``, whose serialized
    payload contains ``schema_version=generational-offspring-batch-draft-v1``.
    The v2.9 semantic adapter owns that compatibility boundary; the legacy
    transport schema must not leak into ``EpistemicOffspringBatchDraft``'s
    strict model validation.
    """
    draft = getattr(result, "draft", result)
    payload = _dump(draft)
    if not isinstance(payload, Mapping):
        raise TypeError("offspring generation payload must be a mapping")
    payload = dict(payload)
    source_schema = payload.pop("schema_version", None)
    if source_schema not in {None, "generational-offspring-batch-draft-v1"}:
        raise ValueError(
            "unsupported offspring batch schema at SIS-v2.9 adapter boundary: "
            f"{source_schema}"
        )
    batch = EpistemicOffspringBatchDraft.model_validate(payload)
    return (
        batch,
        int(getattr(result, "input_tokens", 0) or 0),
        int(getattr(result, "output_tokens", 0) or 0),
    )


def _default_node_builder(
    *,
    candidate: EpistemicOffspringCandidateDraft,
    task: EpistemicG4GenerationTask,
    parent_by_id: Mapping[str, Any],
    source_parallel_report_id: str,
) -> tuple[Any, list[Any]]:
    from pipeline_core.discovery.research_idea_contracts import (
        ResearchIdeaKernel,
        ResearchIdeaNode,
    )

    if candidate.chosen_operator_id not in set(task.allowed_operator_ids):
        raise ValueError("chosen operator is outside task.allowed_operator_ids")
    parent_ids = [task.primary_parent_idea_id]
    if candidate.chosen_operator_id == "CROSS_SOURCE_BRIDGE":
        secondary = str(candidate.secondary_parent_idea_id or "")
        if secondary not in set(task.eligible_secondary_parent_idea_ids):
            raise ValueError("CROSS_SOURCE_BRIDGE requires an eligible secondary parent")
        parent_ids.append(secondary)
    elif candidate.secondary_parent_idea_id is not None:
        raise ValueError("secondary_parent_idea_id is allowed only for CROSS_SOURCE_BRIDGE")

    parents = [parent_by_id[idea_id] for idea_id in parent_ids]
    primary = parents[0]
    if any(row.source_context_id != primary.source_context_id for row in parents):
        raise ValueError("offspring parents must share source context")
    if any(row.source_context_sha256 != primary.source_context_sha256 for row in parents):
        raise ValueError("offspring parents must share source context hash")

    kernel = ResearchIdeaKernel.model_validate(candidate.kernel)
    kernel_sha = _sha(kernel)
    source_object_id = _stable_id(
        f"g{task.generation_index}_epistemic_offspring_source",
        task.task_id,
        task.handoff_id,
        candidate.local_id,
        candidate.chosen_operator_id,
        parent_ids,
        kernel_sha,
    )
    idea_id = _stable_id(
        "research_idea",
        "GENERATIONAL_OFFSPRING",
        source_object_id,
    )
    node = ResearchIdeaNode(
        idea_id=idea_id,
        generation_index=task.generation_index,
        parent_idea_ids=parent_ids,
        source_context_id=primary.source_context_id,
        source_context_sha256=primary.source_context_sha256,
        origin_kind="GENERATIONAL_OFFSPRING",
        source_object_id=source_object_id,
        source_parent_object_ids=[row.source_object_id for row in parents],
        source_kind=f"SIS_V2_9_G{task.generation_index}_{task.channel}",
        operator_id=candidate.chosen_operator_id,
        source_artifact_refs=[source_parallel_report_id, task.handoff_id],
        kernel=kernel,
        kernel_sha256=kernel_sha,
        task_relation_mode=candidate.task_relation_mode,
        differential_prediction=candidate.differential_prediction,
        falsification_condition=candidate.falsification_condition,
        discriminating_observation=candidate.discriminating_observation,
        projection_diagnostic_codes=[
            f"GENERATED_G{task.generation_index}_FROM_EPISTEMIC_HANDOFF",
            f"CHANNEL_{task.channel}",
            "GROUNDING_NOT_REQUIRED_FOR_CONCEPTUAL_GENERATION",
        ],
    )
    return node, parents


def _default_transition_assessor(
    *,
    parents: Sequence[Any],
    proposed: Any,
    operator_id: str,
) -> Any:
    from pipeline_core.discovery.research_idea_semantics import assess_idea_transition

    return assess_idea_transition(
        parents=parents,
        proposed=proposed,
        operator_id=operator_id,
    )


def _transition_payload(transition: Any) -> dict[str, Any]:
    payload = _dump(transition)
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value": payload}


def _semantic_feedback(records: Sequence[EpistemicG4SemanticRecord]) -> str | None:
    if any(row.identity_relation == "DIFFERENT_IDEA" for row in records):
        return None
    return (
        "The previous candidates did not produce a deterministic DIFFERENT_IDEA child. "
        "Retry once with a substantive identity-bearing scientific change to the core "
        "relation, mediator, regime, proxy assumption, contrast, or scientific question. "
        "Do not add evidence claims, citations, or premise IDs. Epistemic feedback remains "
        "search context only."
    )


def execute_epistemic_g4_generation(
    *,
    plan: EpistemicG4GenerationPlan,
    parallel_report: Any,
    parent_by_id: Mapping[str, Any],
    research_question: str,
    backend: EpistemicOffspringBackend,
    semantic_retry_limit: int = 1,
    node_builder: Any | None = None,
    transition_assessor: Any | None = None,
) -> tuple[EpistemicG4ExecutionReport, tuple[EpistemicG4Prompt, ...]]:
    if plan.source_parallel_report_id != parallel_report.report_id:
        raise ValueError("plan / parallel report lineage mismatch")
    if semantic_retry_limit < 0:
        raise ValueError("semantic_retry_limit must be >= 0")
    node_builder = node_builder or _default_node_builder
    transition_assessor = transition_assessor or _default_transition_assessor

    prompts: list[EpistemicG4Prompt] = []
    runs: list[EpistemicG4GenerationRunRecord] = []
    nodes: list[Any] = []
    semantics: list[EpistemicG4SemanticRecord] = []
    population_nodes: list[Any] = []
    seen_kernel_keys: set[tuple[str, str]] = set()
    existing_kernel_owners: dict[str, list[str]] = {}
    for existing_idea_id, existing_node in parent_by_id.items():
        kernel_sha = str(getattr(existing_node, "kernel_sha256", "") or "")
        if kernel_sha:
            existing_kernel_owners.setdefault(kernel_sha, []).append(existing_idea_id)

    for task in plan.tasks:
        prompt = build_epistemic_g4_prompt(
            task=task,
            parent_by_id=parent_by_id,
            parallel_report=parallel_report,
            research_question=research_question,
        )
        prompts.append(prompt)
        calls = 0
        input_tokens = 0
        output_tokens = 0
        retries = 0
        compile_issues: list[str] = []
        task_nodes: list[Any] = []
        task_semantics: list[EpistemicG4SemanticRecord] = []
        previous_draft: Any | None = None

        try:
            generation = backend.generate(prompt)
            calls += 1
            batch, in_tok, out_tok = _normalize_generation(generation)
            input_tokens += in_tok
            output_tokens += out_tok
            previous_draft = getattr(generation, "draft", batch)
        except Exception as exc:
            runs.append(
                EpistemicG4GenerationRunRecord(
                    task_id=task.task_id,
                    handoff_id=task.handoff_id,
                    generation_index=plan.generation_index,
                    primary_parent_idea_id=task.primary_parent_idea_id,
                    channel=task.channel,
                    decision="GENERATION_FAILED",
                    generation_error=f"{type(exc).__name__}:{exc}",
                    llm_call_count=1,
                )
            )
            continue

        def consume(batch: EpistemicOffspringBatchDraft) -> list[EpistemicG4SemanticRecord]:
            if batch.task_id != task.task_id:
                raise ValueError("offspring backend returned wrong task_id")
            if batch.primary_parent_idea_id != task.primary_parent_idea_id:
                raise ValueError("offspring backend returned wrong primary parent")
            if batch.channel != task.channel:
                raise ValueError("offspring backend returned wrong channel")
            added: list[EpistemicG4SemanticRecord] = []
            for index, candidate in enumerate(batch.candidates, start=1):
                if index > task.max_output_count:
                    compile_issues.append(
                        f"EXCEEDS_MAX_OUTPUT_COUNT:{candidate.local_id}"
                    )
                    continue
                try:
                    node, parents = node_builder(
                        candidate=candidate,
                        task=task,
                        parent_by_id=parent_by_id,
                        source_parallel_report_id=plan.source_parallel_report_id,
                    )
                    transition = transition_assessor(
                        parents=parents,
                        proposed=node,
                        operator_id=candidate.chosen_operator_id,
                    )
                    identity = str(getattr(transition, "identity_relation"))
                    if identity not in {"SAME_IDEA", "DIFFERENT_IDEA", "INDETERMINATE"}:
                        raise ValueError(f"unexpected identity relation: {identity}")
                    kernel_sha = str(getattr(node, "kernel_sha256"))
                    duplicate_key = (task.primary_parent_idea_id, kernel_sha)
                    existing_owners = set(existing_kernel_owners.get(kernel_sha, []))
                    parent_ids = set(getattr(node, "parent_idea_ids", []) or [])
                    duplicate_existing_other = sorted(existing_owners - parent_ids)
                    if duplicate_key in seen_kernel_keys:
                        disposition: SemanticDisposition = "EXACT_KERNEL_DUPLICATE_SUPPRESSED"
                        retained = False
                        diagnostics = [f"DUPLICATE_G{plan.generation_index}_KERNEL_WITHIN_PRIMARY_LINEAGE"]
                    elif duplicate_existing_other:
                        seen_kernel_keys.add(duplicate_key)
                        disposition = "EXACT_KERNEL_DUPLICATE_SUPPRESSED"
                        retained = False
                        diagnostics = [
                            "DUPLICATE_EXISTING_RESEARCH_IDEA_KERNEL:"
                            + ",".join(duplicate_existing_other)
                        ]
                    elif identity == "DIFFERENT_IDEA":
                        seen_kernel_keys.add(duplicate_key)
                        disposition = "GENUINE_CHILD"
                        retained = True
                        diagnostics = []
                    elif identity == "INDETERMINATE":
                        seen_kernel_keys.add(duplicate_key)
                        disposition = "INDETERMINATE_PROBE"
                        retained = True
                        diagnostics = [f"INDETERMINATE_IDENTITY_RETAINED_AS_BOUNDED_G{plan.generation_index}_PROBE"]
                    else:
                        seen_kernel_keys.add(duplicate_key)
                        disposition = "SAME_IDEA_REFINEMENT"
                        retained = False
                        diagnostics = ["SAME_IDEA_RETURNED_TO_REALIZATION_LANE"]
                    transition_codes = list(
                        getattr(transition, "diagnostic_codes", []) or []
                    )
                    source_non_grounded = any(
                        maturity in _NON_GROUNDED_MATURITIES
                        for maturity in task.source_maturities
                    )
                    semantic = EpistemicG4SemanticRecord(
                        idea_id=str(getattr(node, "idea_id")),
                        task_id=task.task_id,
                        handoff_id=task.handoff_id,
                        generation_index=plan.generation_index,
                        primary_parent_idea_id=task.primary_parent_idea_id,
                        parent_idea_ids=list(getattr(node, "parent_idea_ids", [])),
                        channel=task.channel,
                        chosen_operator_id=candidate.chosen_operator_id,
                        identity_relation=identity,
                        disposition=disposition,
                        retained_in_g4_population=retained,
                        conceptual_change_summary=candidate.conceptual_change_summary,
                        transition=_transition_payload(transition),
                        diagnostic_codes=_dedupe([*transition_codes, *diagnostics]),
                        source_member_ids=task.source_member_ids,
                        source_maturities=task.source_maturities,
                        source_epistemic_debt_ids=task.source_epistemic_debt_ids,
                        source_epistemic_debt_kinds=task.source_epistemic_debt_kinds,
                        source_contains_non_grounded_feedback=source_non_grounded,
                        source_contains_speculative_feedback=(
                            "SPECULATIVE_BUT_FALSIFIABLE" in task.source_maturities
                        ),
                        source_contains_evidence_seeking_feedback=(
                            "EVIDENCE_SEEKING" in task.source_maturities
                        ),
                    )
                    task_nodes.append(node)
                    task_semantics.append(semantic)
                    added.append(semantic)
                except Exception as exc:
                    compile_issues.append(
                        f"COMPILE_REJECTED:{candidate.local_id}:{type(exc).__name__}:{exc}"
                    )
            return added

        try:
            latest = consume(batch)
        except Exception as exc:
            compile_issues.append(f"BATCH_REJECTED:{type(exc).__name__}:{exc}")
            latest = []

        feedback = _semantic_feedback(latest)
        while feedback is not None and retries < semantic_retry_limit:
            retries += 1
            try:
                generation = backend.repair(prompt, previous_draft, feedback)
                calls += 1
                retry_batch, in_tok, out_tok = _normalize_generation(generation)
                input_tokens += in_tok
                output_tokens += out_tok
                previous_draft = getattr(generation, "draft", retry_batch)
                latest = consume(retry_batch)
                feedback = _semantic_feedback(latest)
            except Exception as exc:
                compile_issues.append(
                    f"SEMANTIC_RETRY_FAILED:{type(exc).__name__}:{exc}"
                )
                break

        # Stable idea IDs make exact semantic retries easy to deduplicate.
        unique_nodes: list[Any] = []
        unique_semantics: list[EpistemicG4SemanticRecord] = []
        seen_ids: set[str] = set()
        semantic_by_id = {row.idea_id: row for row in task_semantics}
        node_by_id = {str(getattr(row, "idea_id")): row for row in task_nodes}
        for idea_id, node in node_by_id.items():
            if idea_id in seen_ids:
                continue
            seen_ids.add(idea_id)
            unique_nodes.append(node)
            unique_semantics.append(semantic_by_id[idea_id])

        nodes.extend(unique_nodes)
        semantics.extend(unique_semantics)
        for node, semantic in zip(unique_nodes, unique_semantics):
            if semantic.retained_in_g4_population:
                population_nodes.append(node)

        genuine_ids = [
            row.idea_id for row in unique_semantics if row.disposition == "GENUINE_CHILD"
        ]
        probe_ids = [
            row.idea_id
            for row in unique_semantics
            if row.disposition == "INDETERMINATE_PROBE"
        ]
        refinement_ids = [
            row.idea_id
            for row in unique_semantics
            if row.disposition == "SAME_IDEA_REFINEMENT"
        ]
        if unique_nodes:
            decision = "GENERATED" if (genuine_ids or probe_ids) else "REJECTED_INVALID_DRAFT"
        elif batch.candidates:
            decision = "REJECTED_INVALID_DRAFT"
        else:
            decision = "ABSTAINED"
        runs.append(
            EpistemicG4GenerationRunRecord(
                task_id=task.task_id,
                handoff_id=task.handoff_id,
                generation_index=plan.generation_index,
                primary_parent_idea_id=task.primary_parent_idea_id,
                channel=task.channel,
                decision=decision,
                generated_idea_ids=[str(getattr(row, "idea_id")) for row in unique_nodes],
                genuine_child_idea_ids=genuine_ids,
                indeterminate_probe_idea_ids=probe_ids,
                same_idea_refinement_ids=refinement_ids,
                semantic_retry_count=retries,
                compile_issue_codes=compile_issues,
                llm_call_count=calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    identity_counts = Counter(row.identity_relation for row in semantics)
    disposition_counts = Counter(row.disposition for row in semantics)
    channel_counts = Counter(row.channel for row in semantics)
    operator_counts = Counter(row.chosen_operator_id for row in semantics)
    genuine = [row for row in semantics if row.disposition == "GENUINE_CHILD"]
    provisional = EpistemicG4ExecutionReport(
        report_id="pending",
        report_sha256="pending",
        source_parallel_report_id=plan.source_parallel_report_id,
        source_plan_id=plan.plan_id,
        generation_index=plan.generation_index,
        tasks=plan.tasks,
        run_records=runs,
        offspring_nodes=nodes,
        semantic_records=semantics,
        g4_population_nodes=population_nodes,
        raw_offspring_count=len(nodes),
        g4_population_count=len(population_nodes),
        genuine_child_count=sum(row.disposition == "GENUINE_CHILD" for row in semantics),
        indeterminate_probe_count=sum(
            row.disposition == "INDETERMINATE_PROBE" for row in semantics
        ),
        same_idea_refinement_count=sum(
            row.disposition == "SAME_IDEA_REFINEMENT" for row in semantics
        ),
        exact_kernel_duplicate_suppressed_count=sum(
            row.disposition == "EXACT_KERNEL_DUPLICATE_SUPPRESSED" for row in semantics
        ),
        identity_relation_counts=dict(sorted(identity_counts.items())),
        disposition_counts=dict(sorted(disposition_counts.items())),
        generated_count_by_channel=dict(sorted(channel_counts.items())),
        generated_count_by_operator=dict(sorted(operator_counts.items())),
        genuine_child_from_non_grounded_feedback_count=sum(
            row.source_contains_non_grounded_feedback for row in genuine
        ),
        genuine_child_from_speculative_feedback_count=sum(
            row.source_contains_speculative_feedback for row in genuine
        ),
        genuine_child_from_evidence_seeking_feedback_count=sum(
            row.source_contains_evidence_seeking_feedback for row in genuine
        ),
        llm_call_count=sum(row.llm_call_count for row in runs),
        input_tokens=sum(row.input_tokens for row in runs),
        output_tokens=sum(row.output_tokens for row in runs),
        semantic_retry_count=sum(row.semantic_retry_count for row in runs),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return (
        provisional.model_copy(
            update={
                "report_id": f"g{plan.generation_index}_epistemic_evolution:{digest[:20]}",
                "report_sha256": digest,
            }
        ),
        tuple(prompts),
    )


def build_epistemic_generation_plan(
    *,
    parallel_report: Any,
    parent_by_id: Mapping[str, Any],
    generation_index: int,
    max_parents: int = 4,
    max_outputs_per_parent: int = 1,
) -> EpistemicG4GenerationPlan:
    """Generation-index generic wrapper over the v2.9 bounded parent selector."""
    return build_epistemic_g4_generation_plan(
        parallel_report=parallel_report,
        parent_by_id=parent_by_id,
        max_parents=max_parents,
        max_outputs_per_parent=max_outputs_per_parent,
        generation_index=generation_index,
    )


def execute_epistemic_generation(
    *,
    plan: EpistemicG4GenerationPlan,
    parallel_report: Any,
    parent_by_id: Mapping[str, Any],
    research_question: str,
    backend: EpistemicOffspringBackend,
    semantic_retry_limit: int = 1,
    node_builder: Any | None = None,
    transition_assessor: Any | None = None,
) -> tuple[EpistemicG4ExecutionReport, tuple[EpistemicG4Prompt, ...]]:
    """Execute one epistemic generation at ``plan.generation_index``.

    The legacy G4 names remain for compatibility; the contract is now reusable
    for G5+ closed-loop iterations.
    """
    return execute_epistemic_g4_generation(
        plan=plan,
        parallel_report=parallel_report,
        parent_by_id=parent_by_id,
        research_question=research_question,
        backend=backend,
        semantic_retry_limit=semantic_retry_limit,
        node_builder=node_builder,
        transition_assessor=transition_assessor,
    )


__all__ = [
    "EpistemicEvolutionParentCandidate",
    "EpistemicG4ExecutionReport",
    "EpistemicG4GenerationPlan",
    "EpistemicG4GenerationRunRecord",
    "EpistemicG4GenerationTask",
    "EpistemicG4Prompt",
    "EpistemicG4SemanticRecord",
    "EpistemicOffspringBackend",
    "EpistemicOffspringBatchDraft",
    "EpistemicOffspringCandidateDraft",
    "EvolutionChannel",
    "SemanticDisposition",
    "SourceMode",
    "build_epistemic_g4_generation_plan",
    "build_epistemic_generation_plan",
    "build_epistemic_g4_prompt",
    "execute_epistemic_g4_generation",
    "execute_epistemic_generation",
]
