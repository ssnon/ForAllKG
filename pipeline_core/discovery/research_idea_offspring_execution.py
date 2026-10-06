from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.hypothesis_compiler import (
    HypothesisCompileError,
    HypothesisCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
    HypothesisPortfolioDraft,
)
from pipeline_core.discovery.hypothesis_llm import (
    HypothesisDraftBackend,
    InstructorOpenAICompatibleHypothesisBackend,
)
from pipeline_core.discovery.hypothesis_prompt import (
    HypothesisPrompt,
    HypothesisPromptAssembler,
)
from pipeline_core.discovery.hypothesis_validation import HypothesisValidator
from pipeline_core.discovery.research_idea_contracts import (
    IdeaTransitionAssessment,
    ResearchIdeaKernel,
    ResearchIdeaNode,
)
from pipeline_core.discovery.research_idea_evolutionary_policy import (
    EvolutionAllocationRecord,
    EvolutionChannel,
    EvolutionOperatorHint,
    EvolutionaryIdeaSearchShadowReport,
)
from pipeline_core.discovery.research_idea_search_contracts import (
    GenerationalIdeaSearchShadowReport,
    IdeaOutcomeObservation,
)
from pipeline_core.discovery.research_idea_semantics import assess_idea_transition
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TaskIdentityGoal = Literal[
    "PRESERVE_IDEA",
    "CREATE_CHILD",
    "PREFER_CHILD",
    "OPEN_BOUNDED",
]

OffspringSemanticDisposition = Literal[
    "ACCEPTED_REFINEMENT",
    "ACCEPTED_CHILD",
    "ACCEPTED_CHANNEL_DRIFT_CHILD",
    "ACCEPTED_INDETERMINATE_PROBE",
    "SEMANTIC_NOOP",
    "EXACT_KERNEL_DUPLICATE_SUPPRESSED",
    "COMPILE_REJECTED",
]

RealizationStatus = Literal[
    "MATERIALIZED",
    "ABSTAINED",
    "GENERATION_FAILED",
    "COMPILE_REJECTED",
    "VALIDATION_REJECTED",
    "CARDINALITY_REJECTED",
]


_ALLOWED_BY_CHANNEL: dict[EvolutionChannel, tuple[EvolutionOperatorHint, ...]] = {
    "EXPLOIT": (
        "SAME_PREMISE_SHARPEN",
        "EVIDENCE_REAXIS",
    ),
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

_IDENTITY_GOAL: dict[EvolutionChannel, TaskIdentityGoal] = {
    "EXPLOIT": "PRESERVE_IDEA",
    "TRANSFORM": "CREATE_CHILD",
    "EXPLORE": "PREFER_CHILD",
    "WILDCARD": "OPEN_BOUNDED",
}

_DEFAULT_MAX_OUTPUTS: dict[EvolutionChannel, int] = {
    "EXPLOIT": 1,
    "TRANSFORM": 2,
    "EXPLORE": 1,
    "WILDCARD": 1,
}


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


class GenerationalOffspringCandidateDraft(StrictModel):
    local_id: str = Field(min_length=1)
    chosen_operator_id: EvolutionOperatorHint
    secondary_parent_idea_id: str | None = None
    conceptual_change_summary: str = Field(min_length=1)
    kernel: ResearchIdeaKernel
    differential_prediction: str = ""
    falsification_condition: str = ""
    discriminating_observation: str = ""
    task_relation_mode: Literal["DIRECT", "SUBORDINATE", "UNKNOWN"] = "UNKNOWN"


class GenerationalOffspringBatchDraft(StrictModel):
    schema_version: Literal[
        "generational-offspring-batch-draft-v1"
    ] = "generational-offspring-batch-draft-v1"
    task_id: str = Field(min_length=1)
    primary_parent_idea_id: str = Field(min_length=1)
    channel: EvolutionChannel
    candidates: list[GenerationalOffspringCandidateDraft] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> "GenerationalOffspringBatchDraft":
        ids = [row.local_id for row in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("offspring candidate local_id values must be unique")
        if self.candidates and self.abstention_reason is not None:
            raise ValueError("abstention_reason must be null when candidates exist")
        if not self.candidates and not str(self.abstention_reason or "").strip():
            raise ValueError("abstention_reason is required when no offspring are proposed")
        return self


class OffspringGenerationTask(StrictModel):
    schema_version: Literal[
        "generational-offspring-task-v1"
    ] = "generational-offspring-task-v1"
    task_id: str = Field(min_length=1)
    primary_parent_idea_id: str = Field(min_length=1)
    eligible_secondary_parent_idea_ids: list[str] = Field(default_factory=list)
    channel: EvolutionChannel
    identity_goal: TaskIdentityGoal
    operator_hints: list[EvolutionOperatorHint] = Field(default_factory=list)
    allowed_operator_ids: list[EvolutionOperatorHint] = Field(min_length=1)
    max_output_count: int = Field(ge=1, le=4)
    source_policy_reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_task(self) -> "OffspringGenerationTask":
        self.eligible_secondary_parent_idea_ids = _dedupe(
            self.eligible_secondary_parent_idea_ids
        )
        self.operator_hints = list(dict.fromkeys(self.operator_hints))
        self.allowed_operator_ids = list(dict.fromkeys(self.allowed_operator_ids))
        self.source_policy_reason_codes = _dedupe(self.source_policy_reason_codes)
        if self.primary_parent_idea_id in set(self.eligible_secondary_parent_idea_ids):
            raise ValueError("primary parent cannot be an eligible secondary parent")
        expected = set(_ALLOWED_BY_CHANNEL[self.channel])
        if not set(self.allowed_operator_ids).issubset(expected):
            raise ValueError("task contains operator not allowed for its channel")
        return self


class OffspringGenerationPlan(StrictModel):
    schema_version: Literal[
        "generational-offspring-plan-v1"
    ] = "generational-offspring-plan-v1"
    plan_id: str
    source_v2_2_report_id: str
    source_generational_report_id: str
    generation_index: int = Field(default=2, ge=2)
    tasks: list[OffspringGenerationTask] = Field(default_factory=list)
    task_count: int = Field(ge=0)
    max_raw_offspring: int = Field(ge=1)
    planned_max_offspring: int = Field(ge=0)
    task_count_by_channel: dict[str, int] = Field(default_factory=dict)
    operator_hints_are_nonbinding: Literal[True] = True
    conceptual_generation_is_inspiration_only: Literal[True] = True
    grounding_required_at_realization_boundary: Literal[True] = True
    external_prior_art_as_positive_premise: Literal[False] = False
    production_generation_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "OffspringGenerationPlan":
        if self.task_count != len(self.tasks):
            raise ValueError("task_count mismatch")
        if self.planned_max_offspring != sum(row.max_output_count for row in self.tasks):
            raise ValueError("planned_max_offspring mismatch")
        if self.planned_max_offspring > self.max_raw_offspring:
            raise ValueError("planned offspring exceeds raw-offspring bound")
        return self


@dataclass(frozen=True)
class OffspringPrompt:
    task_id: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class OffspringGeneration:
    draft: GenerationalOffspringBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class GenerationalOffspringBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(self, prompt: OffspringPrompt) -> OffspringGeneration: ...

    def repair(
        self,
        prompt: OffspringPrompt,
        previous_draft: GenerationalOffspringBatchDraft,
        feedback: str,
    ) -> OffspringGeneration: ...


class InstructorOpenAICompatibleOffspringBackend:
    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.2,
        parse_retries: int = 2,
        timeout: float | None = 180.0,
        extra_headers: Mapping[str, str] | None = None,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key_env = str(api_key_env)
        self.api_key = os.getenv(self.api_key_env)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.instructor_mode = str(instructor_mode).upper()
        self.temperature = float(temperature)
        self.parse_retries = int(parse_retries)
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self.telemetry_path = telemetry_path
        self.telemetry_context = dict(telemetry_context or {})
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(f"No API key available. Set {self.api_key_env}.")
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "SIS-v2.3 offspring generation requires openai and instructor."
            ) from exc
        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            raise ValueError(f"Unknown Instructor mode: {self.instructor_mode}")
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers
        self._client = instructor.from_openai(OpenAI(**kwargs), mode=mode)
        return self._client

    def _call(
        self,
        messages: list[dict[str, str]],
        *,
        task_id: str,
        stage: str,
    ) -> OffspringGeneration:
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=GenerationalOffspringBatchDraft,
            messages=messages,
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "sis_v2_3_generational_offspring",
                "stage": stage,
                "task_id": task_id,
                "call_kind": "research_idea_offspring_generation",
            },
            semantic_components={
                "authority": "INSPIRATION_ONLY",
                "research_idea_task_id": task_id,
            },
        )
        if not isinstance(draft, GenerationalOffspringBatchDraft):
            draft = GenerationalOffspringBatchDraft.model_validate(draft)
        return OffspringGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )

    def generate(self, prompt: OffspringPrompt) -> OffspringGeneration:
        return self._call(
            [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            task_id=prompt.task_id,
            stage="generation",
        )

    def repair(
        self,
        prompt: OffspringPrompt,
        previous_draft: GenerationalOffspringBatchDraft,
        feedback: str,
    ) -> OffspringGeneration:
        return self._call(
            [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
                {"role": "assistant", "content": previous_draft.model_dump_json(indent=2)},
                {"role": "user", "content": feedback},
            ],
            task_id=prompt.task_id,
            stage="semantic_retry",
        )


class OffspringGenerationRunRecord(StrictModel):
    task_id: str
    primary_parent_idea_id: str
    channel: EvolutionChannel
    decision: Literal[
        "GENERATED",
        "ABSTAINED",
        "GENERATION_FAILED",
        "REJECTED_INVALID_DRAFT",
    ]
    generated_idea_ids: list[str] = Field(default_factory=list)
    accepted_for_realization_idea_ids: list[str] = Field(default_factory=list)
    semantic_retry_count: int = Field(ge=0, default=0)
    compile_issue_codes: list[str] = Field(default_factory=list)
    generation_error: str | None = None
    llm_call_count: int = Field(ge=0, default=0)
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)


class OffspringSemanticRecord(StrictModel):
    idea_id: str
    task_id: str
    channel: EvolutionChannel
    chosen_operator_id: EvolutionOperatorHint
    parent_idea_ids: list[str] = Field(min_length=1)
    transition: IdeaTransitionAssessment
    disposition: OffspringSemanticDisposition
    accepted_for_realization: bool
    conceptual_change_summary: str
    diagnostic_codes: list[str] = Field(default_factory=list)


class OffspringExecutionReport(StrictModel):
    schema_version: Literal[
        "generational-offspring-execution-shadow-v1"
    ] = "generational-offspring-execution-shadow-v1"
    report_id: str
    report_sha256: str
    source_plan_id: str
    source_v2_2_report_id: str
    source_generational_report_id: str
    generation_index: int = Field(default=2, ge=2)
    generation_tasks: list[OffspringGenerationTask] = Field(default_factory=list)
    run_records: list[OffspringGenerationRunRecord] = Field(default_factory=list)
    offspring_nodes: list[ResearchIdeaNode] = Field(default_factory=list)
    semantic_records: list[OffspringSemanticRecord] = Field(default_factory=list)
    raw_offspring_count: int = Field(ge=0)
    accepted_for_realization_count: int = Field(ge=0)
    identity_relation_counts: dict[str, int] = Field(default_factory=dict)
    disposition_counts: dict[str, int] = Field(default_factory=dict)
    generated_count_by_channel: dict[str, int] = Field(default_factory=dict)
    generated_count_by_operator: dict[str, int] = Field(default_factory=dict)
    semantic_noop_count: int = Field(ge=0)
    mutation_attempt_count: int = Field(ge=0)
    distinct_child_count: int = Field(ge=0)
    mutation_semantic_yield_fraction: float = Field(ge=0.0, le=1.0)
    indeterminate_probe_count: int = Field(ge=0)
    channel_drift_child_count: int = Field(ge=0)
    exact_kernel_duplicate_suppressed_count: int = Field(ge=0)
    llm_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    semantic_retry_count: int = Field(ge=0)
    offspring_generation_executed: Literal[True] = True
    conceptual_generation_is_inspiration_only: Literal[True] = True
    semantic_identity_is_posthoc_diagnostic: Literal[True] = True
    indeterminate_identity_is_not_automatic_rejection: Literal[True] = True
    external_prior_art_as_positive_premise: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "OffspringExecutionReport":
        if self.raw_offspring_count != len(self.offspring_nodes):
            raise ValueError("raw_offspring_count mismatch")
        if self.raw_offspring_count != len(self.semantic_records):
            raise ValueError("semantic record count mismatch")
        if self.accepted_for_realization_count != sum(
            row.accepted_for_realization for row in self.semantic_records
        ):
            raise ValueError("accepted realization count mismatch")
        return self


class OffspringRealizationRecord(StrictModel):
    idea_id: str
    channel: EvolutionChannel
    semantic_disposition: OffspringSemanticDisposition
    status: RealizationStatus
    hypothesis_id: str | None = None
    issue_codes: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    generation_attempt_count: int = Field(ge=0, default=0)
    repair_attempt_count: int = Field(ge=0, default=0)
    input_tokens: int = Field(ge=0, default=0)
    output_tokens: int = Field(ge=0, default=0)


class OffspringRealizationReport(StrictModel):
    schema_version: Literal[
        "generational-offspring-realization-shadow-v1"
    ] = "generational-offspring-realization-shadow-v1"
    report_id: str
    report_sha256: str
    source_offspring_execution_report_id: str
    source_context_id: str
    source_context_sha256: str
    selected_idea_ids: list[str] = Field(default_factory=list)
    selected_idea_count: int = Field(ge=0)
    records: list[OffspringRealizationRecord] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    materialized_hypothesis_count: int = Field(ge=0)
    materialization_success_fraction: float = Field(ge=0.0, le=1.0)
    output_portfolio_id: str
    llm_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    idea_used_as_inspiration_only: Literal[True] = True
    grounded_premise_boundary_enforced: Literal[True] = True
    standard_hypothesis_compiler_used: Literal[True] = True
    standard_hypothesis_validator_used: Literal[True] = True
    realization_failure_is_not_idea_failure: Literal[True] = True
    production_selection_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "OffspringRealizationReport":
        if self.selected_idea_count != len(self.selected_idea_ids):
            raise ValueError("selected_idea_count mismatch")
        if len(self.records) != self.selected_idea_count:
            raise ValueError("realization record count mismatch")
        if self.materialized_hypothesis_count != sum(
            row.status == "MATERIALIZED" for row in self.records
        ):
            raise ValueError("materialized hypothesis count mismatch")
        return self


class G3FeedbackSeedReport(StrictModel):
    schema_version: Literal[
        "g3-idea-feedback-seed-v1"
    ] = "g3-idea-feedback-seed-v1"
    report_id: str
    report_sha256: str
    source_offspring_execution_report_id: str
    source_realization_report_id: str
    generation2_idea_ids: list[str] = Field(default_factory=list)
    suppressed_generation2_attempt_idea_ids: list[str] = Field(default_factory=list)
    realization_observations: list[IdeaOutcomeObservation] = Field(default_factory=list)
    observation_count: int = Field(ge=0)
    ready_for_next_credit_assignment: Literal[True] = True
    realization_observations_are_soft_feedback: Literal[True] = True
    verification_is_not_fertility_authority: Literal[True] = True
    production_selection_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "G3FeedbackSeedReport":
        if self.observation_count != len(self.realization_observations):
            raise ValueError("observation_count mismatch")
        return self


def build_offspring_generation_plan(
    *,
    evolutionary_report: EvolutionaryIdeaSearchShadowReport,
    generational_report: GenerationalIdeaSearchShadowReport,
    max_raw_offspring: int = 12,
    generation_index: int = 2,
) -> OffspringGenerationPlan:
    if max_raw_offspring < 1:
        raise ValueError("max_raw_offspring must be >= 1")
    if generation_index < 2:
        raise ValueError("generation_index must be >= 2")
    if evolutionary_report.source_generational_report_id != generational_report.report_id:
        raise ValueError("v2.2 / v2.1.1 generational lineage mismatch")

    node_by_id = {row.idea_id: row for row in generational_report.research_ideas}
    policy_by_id = {row.idea_id: row for row in evolutionary_report.policy_states}
    selected = list(evolutionary_report.selected_parent_idea_ids)
    for idea_id in selected:
        if idea_id not in node_by_id:
            raise ValueError(f"v2.2 parent missing from generational population: {idea_id}")

    tasks: list[OffspringGenerationTask] = []
    remaining = max_raw_offspring
    for allocation in evolutionary_report.allocations:
        if remaining <= 0:
            break
        idea_id = allocation.idea_id
        policy = policy_by_id.get(idea_id)
        allowed = list(_ALLOWED_BY_CHANNEL[allocation.channel])
        hints = [hint for hint in allocation.operator_hints if hint in set(allowed)]
        if not hints:
            hints = allowed[:2]
        max_outputs = min(_DEFAULT_MAX_OUTPUTS[allocation.channel], remaining)
        if max_outputs <= 0:
            continue
        secondary = [other for other in selected if other != idea_id]
        task = OffspringGenerationTask(
            task_id=_stable_id(
                f"g{generation_index}_offspring_task",
                evolutionary_report.report_id,
                idea_id,
                allocation.channel,
            ),
            primary_parent_idea_id=idea_id,
            eligible_secondary_parent_idea_ids=secondary,
            channel=allocation.channel,
            identity_goal=_IDENTITY_GOAL[allocation.channel],
            operator_hints=hints,
            allowed_operator_ids=allowed,
            max_output_count=max_outputs,
            source_policy_reason_codes=(
                list(policy.reason_codes) if policy is not None else []
            ),
        )
        tasks.append(task)
        remaining -= max_outputs

    provisional = {
        "source_v2_2_report_id": evolutionary_report.report_id,
        "source_generational_report_id": generational_report.report_id,
        "generation_index": generation_index,
        "tasks": [row.model_dump(mode="json") for row in tasks],
        "max_raw_offspring": max_raw_offspring,
    }
    plan_id = _stable_id(f"g{generation_index}_offspring_plan", provisional)
    counts = Counter(row.channel for row in tasks)
    return OffspringGenerationPlan(
        plan_id=plan_id,
        source_v2_2_report_id=evolutionary_report.report_id,
        source_generational_report_id=generational_report.report_id,
        generation_index=generation_index,
        tasks=tasks,
        task_count=len(tasks),
        max_raw_offspring=max_raw_offspring,
        planned_max_offspring=sum(row.max_output_count for row in tasks),
        task_count_by_channel=dict(sorted(counts.items())),
    )


def _kernel_payload(node: ResearchIdeaNode) -> dict[str, Any]:
    return {
        "idea_id": node.idea_id,
        "generation_index": node.generation_index,
        "origin_kind": node.origin_kind,
        "kernel": node.kernel.model_dump(mode="json"),
        "task_relation_mode": node.task_relation_mode,
        "differential_prediction": node.differential_prediction,
        "falsification_condition": node.falsification_condition,
        "discriminating_observation": node.discriminating_observation,
    }


def build_offspring_prompt(
    *,
    task: OffspringGenerationTask,
    parent_by_id: Mapping[str, ResearchIdeaNode],
    research_question: str,
) -> OffspringPrompt:
    primary = parent_by_id[task.primary_parent_idea_id]
    secondary_rows = [
        _kernel_payload(parent_by_id[idea_id])
        for idea_id in task.eligible_secondary_parent_idea_ids
        if idea_id in parent_by_id
    ]
    system = """You are the generational ResearchIdea evolution engine in a scientific discovery system.

Generate conceptual research-idea offspring, not verified scientific claims.
ResearchIdea kernels are INSPIRATION_ONLY. They may be imaginative and need not already be stated in the current grounded corpus.

Authority boundary:
- You MAY mutate mechanisms, mediators, causal direction, latent variables, regimes, proxy assumptions, contrasts, and scientific questions.
- You MUST remain DIRECT or SUBORDINATE to the supplied research task.
- You MUST NOT claim literature-wide novelty, truth, or empirical support.
- You MUST NOT invent citations, premise IDs, paper evidence, or provenance.
- External prior-art/verification outcomes, if reflected in policy hints, are search-boundary signals only; they are never positive premises.
- Grounded positive-premise admissibility is enforced later at realization time, not here.

Identity semantics:
- EXPLOIT: preserve the parent ResearchIdea kernel exactly; improve only realization-facing prediction/falsifier/discriminating observation. Use SAME_PREMISE_SHARPEN or EVIDENCE_REAXIS.
- TRANSFORM: make at least one substantive identity-bearing scientific commitment change. A lexical rewrite is a failure.
- EXPLORE: prefer a scientifically distinct child that enters an underexplored adjacent region while staying on-task.
- WILDCARD: allow a bounded high-variance conceptual move while remaining scientifically coherent and task-related.

Operator hints are non-binding search suggestions. chosen_operator_id must be one of allowed_operator_ids.
CROSS_SOURCE_BRIDGE requires one supplied eligible secondary parent and secondary_parent_idea_id. Other operators must not attach a secondary parent.
Return only the structured GenerationalOffspringBatchDraft requested by the caller.
"""
    user_payload = {
        "task_id": task.task_id,
        "research_question": research_question,
        "channel": task.channel,
        "identity_goal": task.identity_goal,
        "max_output_count": task.max_output_count,
        "primary_parent": _kernel_payload(primary),
        "eligible_secondary_parents": secondary_rows,
        "operator_hints_nonbinding": task.operator_hints,
        "allowed_operator_ids": task.allowed_operator_ids,
        "policy_reason_codes_boundary_only": task.source_policy_reason_codes,
        "output_rules": [
            "Return zero to max_output_count candidates.",
            "Use unique local_id values.",
            "Do not make novelty/truth/evidence claims.",
            "For EXPLOIT copy the parent kernel fields exactly.",
            "For non-EXPLOIT do not use mere wording changes as conceptual mutation.",
            "Every candidate must provide a falsifiable scientific direction, but it remains unverified inspiration.",
        ],
    }
    user = json.dumps(user_payload, ensure_ascii=False, indent=2, sort_keys=True)
    return OffspringPrompt(
        task_id=task.task_id,
        system_prompt=system,
        user_prompt=user,
        prompt_sha256=_sha({"system": system, "user": user}),
    )


def _compile_offspring_candidate(
    *,
    draft: GenerationalOffspringCandidateDraft,
    task: OffspringGenerationTask,
    parent_by_id: Mapping[str, ResearchIdeaNode],
    source_v2_2_report_id: str,
    generation_index: int,
) -> tuple[ResearchIdeaNode, list[ResearchIdeaNode]]:
    if draft.chosen_operator_id not in set(task.allowed_operator_ids):
        raise ValueError("chosen operator is outside task.allowed_operator_ids")

    parent_ids = [task.primary_parent_idea_id]
    if draft.chosen_operator_id == "CROSS_SOURCE_BRIDGE":
        secondary = str(draft.secondary_parent_idea_id or "")
        if secondary not in set(task.eligible_secondary_parent_idea_ids):
            raise ValueError("CROSS_SOURCE_BRIDGE requires an eligible secondary parent")
        parent_ids.append(secondary)
    elif draft.secondary_parent_idea_id is not None:
        raise ValueError("secondary_parent_idea_id is allowed only for CROSS_SOURCE_BRIDGE")

    parents = [parent_by_id[idea_id] for idea_id in parent_ids]
    primary = parents[0]
    if any(row.source_context_id != primary.source_context_id for row in parents):
        raise ValueError("offspring parents must share source context")
    if any(row.source_context_sha256 != primary.source_context_sha256 for row in parents):
        raise ValueError("offspring parents must share source context hash")

    kernel_sha = _sha(draft.kernel)
    source_object_id = _stable_id(
        f"g{generation_index}_offspring_source",
        task.task_id,
        draft.local_id,
        draft.chosen_operator_id,
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
        generation_index=generation_index,
        parent_idea_ids=parent_ids,
        source_context_id=primary.source_context_id,
        source_context_sha256=primary.source_context_sha256,
        origin_kind="GENERATIONAL_OFFSPRING",
        source_object_id=source_object_id,
        source_parent_object_ids=[row.source_object_id for row in parents],
        source_kind=f"SIS_GENERATIONAL_G{generation_index}_{task.channel}",
        operator_id=draft.chosen_operator_id,
        source_artifact_refs=[source_v2_2_report_id],
        kernel=draft.kernel,
        kernel_sha256=kernel_sha,
        task_relation_mode=draft.task_relation_mode,
        differential_prediction=draft.differential_prediction,
        falsification_condition=draft.falsification_condition,
        discriminating_observation=draft.discriminating_observation,
        projection_diagnostic_codes=[
            f"GENERATED_G{generation_index}_OFFSPRING",
            f"CHANNEL_{task.channel}",
        ],
    )
    return node, parents


def _semantic_disposition(
    *,
    channel: EvolutionChannel,
    transition: IdeaTransitionAssessment,
) -> tuple[OffspringSemanticDisposition, bool, list[str]]:
    identity = transition.identity_relation
    diagnostics: list[str] = []
    if channel == "EXPLOIT":
        if identity == "SAME_IDEA":
            return "ACCEPTED_REFINEMENT", True, diagnostics
        if identity == "DIFFERENT_IDEA":
            diagnostics.append("EXPLOIT_GENERATION_CROSSED_IDEA_BOUNDARY")
            return "ACCEPTED_CHANNEL_DRIFT_CHILD", True, diagnostics
        diagnostics.append("EXPLOIT_IDENTITY_INDETERMINATE_RETAINED_AS_PROBE")
        return "ACCEPTED_INDETERMINATE_PROBE", True, diagnostics

    if identity == "DIFFERENT_IDEA":
        return "ACCEPTED_CHILD", True, diagnostics
    if identity == "SAME_IDEA":
        diagnostics.append("NON_EXPLOIT_OFFSPRING_SEMANTIC_NOOP")
        return "SEMANTIC_NOOP", False, diagnostics
    diagnostics.append("INDETERMINATE_IDENTITY_RETAINED_AS_BOUNDED_PROBE")
    return "ACCEPTED_INDETERMINATE_PROBE", True, diagnostics


def _semantic_feedback(
    task: OffspringGenerationTask,
    records: Sequence[OffspringSemanticRecord],
) -> str | None:
    if task.channel == "EXPLOIT":
        if any(row.transition.identity_relation == "SAME_IDEA" for row in records):
            return None
        return (
            "The EXPLOIT candidates crossed or blurred the ResearchIdea identity boundary. "
            "Retry once by copying canonical_intent, core_scientific_commitments, "
            "scope_commitments, contrastive_commitments, and question_commitment exactly "
            "from the primary parent. Change only realization-facing prediction, falsifier, "
            "or discriminating observation."
        )
    if any(row.transition.identity_relation == "DIFFERENT_IDEA" for row in records):
        return None
    return (
        "The previous candidates did not establish a scientifically distinct child under "
        "the deterministic kernel comparator. Retry once with a substantive change to an "
        "identity-bearing scientific commitment (core relation, contrast, or scientific "
        "question), not a lexical rewrite. Preserve task alignment and do not invent evidence."
    )


def execute_offspring_plan(
    *,
    plan: OffspringGenerationPlan,
    evolutionary_report: EvolutionaryIdeaSearchShadowReport,
    generational_report: GenerationalIdeaSearchShadowReport,
    research_question: str,
    backend: GenerationalOffspringBackend,
    semantic_retry_limit: int = 1,
) -> tuple[OffspringExecutionReport, tuple[OffspringPrompt, ...]]:
    if plan.source_v2_2_report_id != evolutionary_report.report_id:
        raise ValueError("offspring plan / v2.2 lineage mismatch")
    if plan.source_generational_report_id != generational_report.report_id:
        raise ValueError("offspring plan / generational lineage mismatch")
    if semantic_retry_limit < 0:
        raise ValueError("semantic_retry_limit must be >= 0")

    parent_by_id = {row.idea_id: row for row in generational_report.research_ideas}
    prompts: list[OffspringPrompt] = []
    run_records: list[OffspringGenerationRunRecord] = []
    nodes: list[ResearchIdeaNode] = []
    semantic_rows: list[OffspringSemanticRecord] = []
    seen_new_kernel_keys: set[tuple[str, str]] = set()

    for task in plan.tasks:
        prompt = build_offspring_prompt(
            task=task,
            parent_by_id=parent_by_id,
            research_question=research_question,
        )
        prompts.append(prompt)
        calls = 0
        input_tokens = 0
        output_tokens = 0
        retries = 0
        compile_issues: list[str] = []
        task_nodes: list[ResearchIdeaNode] = []
        task_semantics: list[OffspringSemanticRecord] = []
        previous_draft: GenerationalOffspringBatchDraft | None = None

        try:
            generation = backend.generate(prompt)
            calls += 1
            input_tokens += int(generation.input_tokens or 0)
            output_tokens += int(generation.output_tokens or 0)
            previous_draft = generation.draft
        except Exception as exc:
            run_records.append(
                OffspringGenerationRunRecord(
                    task_id=task.task_id,
                    primary_parent_idea_id=task.primary_parent_idea_id,
                    channel=task.channel,
                    decision="GENERATION_FAILED",
                    generation_error=f"{type(exc).__name__}:{exc}",
                    llm_call_count=1,
                )
            )
            continue

        def consume(draft_batch: GenerationalOffspringBatchDraft) -> None:
            if draft_batch.task_id != task.task_id:
                raise ValueError("offspring backend returned wrong task_id")
            if draft_batch.primary_parent_idea_id != task.primary_parent_idea_id:
                raise ValueError("offspring backend returned wrong primary parent")
            if draft_batch.channel != task.channel:
                raise ValueError("offspring backend returned wrong channel")
            for index, candidate in enumerate(draft_batch.candidates, start=1):
                if index > task.max_output_count:
                    compile_issues.append(
                        f"EXCEEDS_MAX_OUTPUT_COUNT:{candidate.local_id}"
                    )
                    continue
                try:
                    node, parents = _compile_offspring_candidate(
                        draft=candidate,
                        task=task,
                        parent_by_id=parent_by_id,
                        source_v2_2_report_id=evolutionary_report.report_id,
                        generation_index=plan.generation_index,
                    )
                    transition = assess_idea_transition(
                        parents=parents,
                        proposed=node,
                        operator_id=candidate.chosen_operator_id,
                    )
                    disposition, accepted, diagnostics = _semantic_disposition(
                        channel=task.channel,
                        transition=transition,
                    )
                    duplicate_key = (task.primary_parent_idea_id, node.kernel_sha256)
                    if duplicate_key in seen_new_kernel_keys:
                        disposition = "EXACT_KERNEL_DUPLICATE_SUPPRESSED"
                        accepted = False
                        diagnostics.append(
                            f"DUPLICATE_G{plan.generation_index}_KERNEL_WITHIN_PRIMARY_LINEAGE"
                        )
                    else:
                        seen_new_kernel_keys.add(duplicate_key)
                    task_nodes.append(node)
                    task_semantics.append(
                        OffspringSemanticRecord(
                            idea_id=node.idea_id,
                            task_id=task.task_id,
                            channel=task.channel,
                            chosen_operator_id=candidate.chosen_operator_id,
                            parent_idea_ids=list(node.parent_idea_ids),
                            transition=transition,
                            disposition=disposition,
                            accepted_for_realization=accepted,
                            conceptual_change_summary=candidate.conceptual_change_summary,
                            diagnostic_codes=_dedupe(
                                [*transition.diagnostic_codes, *diagnostics]
                            ),
                        )
                    )
                except Exception as exc:
                    compile_issues.append(
                        f"COMPILE_REJECTED:{candidate.local_id}:{type(exc).__name__}:{exc}"
                    )

        try:
            consume(previous_draft)
        except Exception as exc:
            compile_issues.append(f"BATCH_REJECTED:{type(exc).__name__}:{exc}")

        feedback = _semantic_feedback(task, task_semantics)
        while feedback is not None and retries < semantic_retry_limit:
            retries += 1
            try:
                generation = backend.repair(prompt, previous_draft, feedback)
                calls += 1
                input_tokens += int(generation.input_tokens or 0)
                output_tokens += int(generation.output_tokens or 0)
                previous_draft = generation.draft
                before = len(task_semantics)
                consume(previous_draft)
                feedback = _semantic_feedback(task, task_semantics[before:])
            except Exception as exc:
                compile_issues.append(
                    f"SEMANTIC_RETRY_FAILED:{type(exc).__name__}:{exc}"
                )
                break

        # Deduplicate exact idea IDs if a semantic retry reproduces a prior draft.
        unique_nodes: list[ResearchIdeaNode] = []
        unique_semantics: list[OffspringSemanticRecord] = []
        seen_ids: set[str] = set()
        semantics_by_id = {row.idea_id: row for row in task_semantics}
        for node in task_nodes:
            if node.idea_id in seen_ids:
                continue
            seen_ids.add(node.idea_id)
            unique_nodes.append(node)
            unique_semantics.append(semantics_by_id[node.idea_id])

        nodes.extend(unique_nodes)
        semantic_rows.extend(unique_semantics)
        accepted_ids = [
            row.idea_id for row in unique_semantics if row.accepted_for_realization
        ]
        if unique_nodes:
            decision = "GENERATED" if accepted_ids else "REJECTED_INVALID_DRAFT"
        elif previous_draft is not None and not previous_draft.candidates:
            decision = "ABSTAINED"
        else:
            decision = "REJECTED_INVALID_DRAFT"
        run_records.append(
            OffspringGenerationRunRecord(
                task_id=task.task_id,
                primary_parent_idea_id=task.primary_parent_idea_id,
                channel=task.channel,
                decision=decision,
                generated_idea_ids=[row.idea_id for row in unique_nodes],
                accepted_for_realization_idea_ids=accepted_ids,
                semantic_retry_count=retries,
                compile_issue_codes=compile_issues,
                llm_call_count=calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    identity_counts = Counter(
        row.transition.identity_relation for row in semantic_rows
    )
    disposition_counts = Counter(row.disposition for row in semantic_rows)
    channel_counts = Counter(row.channel for row in semantic_rows)
    operator_counts = Counter(row.chosen_operator_id for row in semantic_rows)
    mutation_attempt_count = sum(
        row.channel != "EXPLOIT" for row in semantic_rows
    )
    distinct_child_count = sum(
        row.channel != "EXPLOIT"
        and row.transition.identity_relation == "DIFFERENT_IDEA"
        for row in semantic_rows
    )
    mutation_semantic_yield_fraction = (
        distinct_child_count / mutation_attempt_count
        if mutation_attempt_count
        else 0.0
    )
    provisional = OffspringExecutionReport(
        report_id="pending",
        report_sha256="pending",
        source_plan_id=plan.plan_id,
        source_v2_2_report_id=evolutionary_report.report_id,
        source_generational_report_id=generational_report.report_id,
        generation_index=plan.generation_index,
        generation_tasks=plan.tasks,
        run_records=run_records,
        offspring_nodes=nodes,
        semantic_records=semantic_rows,
        raw_offspring_count=len(nodes),
        accepted_for_realization_count=sum(
            row.accepted_for_realization for row in semantic_rows
        ),
        identity_relation_counts=dict(sorted(identity_counts.items())),
        disposition_counts=dict(sorted(disposition_counts.items())),
        generated_count_by_channel=dict(sorted(channel_counts.items())),
        generated_count_by_operator=dict(sorted(operator_counts.items())),
        semantic_noop_count=disposition_counts.get("SEMANTIC_NOOP", 0),
        mutation_attempt_count=mutation_attempt_count,
        distinct_child_count=distinct_child_count,
        mutation_semantic_yield_fraction=mutation_semantic_yield_fraction,
        indeterminate_probe_count=disposition_counts.get(
            "ACCEPTED_INDETERMINATE_PROBE", 0
        ),
        channel_drift_child_count=disposition_counts.get(
            "ACCEPTED_CHANNEL_DRIFT_CHILD", 0
        ),
        exact_kernel_duplicate_suppressed_count=disposition_counts.get(
            "EXACT_KERNEL_DUPLICATE_SUPPRESSED", 0
        ),
        llm_call_count=sum(row.llm_call_count for row in run_records),
        input_tokens=sum(row.input_tokens for row in run_records),
        output_tokens=sum(row.output_tokens for row in run_records),
        semantic_retry_count=sum(row.semantic_retry_count for row in run_records),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return (
        provisional.model_copy(
            update={
                "report_id": f"g{plan.generation_index}_offspring_execution:{digest[:20]}",
                "report_sha256": digest,
            }
        ),
        tuple(prompts),
    )


def select_offspring_for_realization(
    execution: OffspringExecutionReport,
    *,
    max_realizations: int = 6,
    max_per_parent: int = 2,
) -> list[str]:
    if max_realizations < 1:
        raise ValueError("max_realizations must be >= 1")
    if max_per_parent < 1:
        raise ValueError("max_per_parent must be >= 1")
    semantic_by_id = {row.idea_id: row for row in execution.semantic_records}
    node_by_id = {row.idea_id: row for row in execution.offspring_nodes}
    eligible = [
        row
        for row in execution.semantic_records
        if row.accepted_for_realization
    ]
    identity_rank = {"DIFFERENT_IDEA": 0, "INDETERMINATE": 1, "SAME_IDEA": 2}
    channel_cycle: tuple[EvolutionChannel, ...] = (
        "TRANSFORM",
        "EXPLORE",
        "EXPLOIT",
        "WILDCARD",
    )
    selected: list[str] = []
    per_parent: Counter[str] = Counter()
    selected_kernels: set[str] = set()

    grouped: dict[str, list[OffspringSemanticRecord]] = defaultdict(list)
    for row in eligible:
        grouped[row.channel].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                identity_rank[row.transition.identity_relation],
                row.idea_id,
            )
        )

    while len(selected) < max_realizations:
        progress = False
        for channel in channel_cycle:
            rows = grouped.get(channel, [])
            choice = None
            for row in rows:
                if row.idea_id in selected:
                    continue
                node = node_by_id[row.idea_id]
                primary_parent = node.parent_idea_ids[0]
                if per_parent[primary_parent] >= max_per_parent:
                    continue
                if node.kernel_sha256 in selected_kernels and row.channel != "EXPLOIT":
                    continue
                choice = row
                break
            if choice is None:
                continue
            node = node_by_id[choice.idea_id]
            selected.append(choice.idea_id)
            per_parent[node.parent_idea_ids[0]] += 1
            selected_kernels.add(node.kernel_sha256)
            progress = True
            if len(selected) >= max_realizations:
                break
        if not progress:
            break
    return selected


def build_realization_prompt(
    *,
    context: HypothesisContext,
    node: ResearchIdeaNode,
    semantic: OffspringSemanticRecord,
) -> HypothesisPrompt:
    base = HypothesisPromptAssembler(max_hypotheses=1).build(context)
    system = base.system_prompt + """

SIS-v2.3 GENERATED RESEARCH IDEA REALIZATION
===========================================
The supplied ResearchIdea is INSPIRATION_ONLY, not evidence.
Its kernel, parent lineage, operator, policy lane, and conceptual-change text MUST NOT be treated as a positive scientific premise.
Only exact evidence statement IDs marked eligible_as_premise in the HypothesisContext may appear in premise_statement_ids.
External prior-art/search feedback is not positive evidence.

Try to realize the supplied idea as exactly ONE grounded, falsifiable HypothesisProposalDraft.
The inferential bridge may make the conceptual leap explicit, but the positive premises must remain grounded.
Do not silently replace the supplied ResearchIdea with an easier different idea merely to fit the evidence.
If the supplied idea cannot be grounded into a scientifically coherent falsifiable realization under the current context, abstain.
Do not claim novelty, truth, or experimental confirmation.
"""
    idea_payload = {
        "idea_id": node.idea_id,
        "generation_index": node.generation_index,
        "parent_idea_ids": node.parent_idea_ids,
        "channel": semantic.channel,
        "operator_id": semantic.chosen_operator_id,
        "semantic_identity_relation": semantic.transition.identity_relation,
        "semantic_disposition": semantic.disposition,
        "kernel": node.kernel.model_dump(mode="json"),
        "differential_prediction": node.differential_prediction,
        "falsification_condition": node.falsification_condition,
        "discriminating_observation": node.discriminating_observation,
        "warning": "All fields above are inspiration-only and are not positive evidence.",
    }
    user = (
        base.user_prompt
        + "\n\nGENERATED RESEARCH IDEA — INSPIRATION ONLY\n"
        + "============================================\n"
        + json.dumps(idea_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\nReturn exactly one grounded hypothesis realization or abstain."
    )
    return HypothesisPrompt.create(system_prompt=system, user_prompt=user)


def _validation_issues(validation: Any) -> list[str]:
    return [
        f"{row.code}:{row.location}:{row.message}"
        for row in validation.issues
        if row.severity == "error"
    ]


def _compile_single_realization(
    *,
    context: HypothesisContext,
    draft: HypothesisPortfolioDraft,
) -> tuple[HypothesisPortfolio | None, RealizationStatus, list[str], list[str]]:
    if not draft.hypotheses:
        return None, "ABSTAINED", [], [str(draft.abstention_reason or "model abstained")]
    if len(draft.hypotheses) != 1:
        return None, "CARDINALITY_REJECTED", ["EXPECTED_EXACTLY_ONE_HYPOTHESIS"], []
    try:
        portfolio = HypothesisCompiler().compile(context, draft)
    except HypothesisCompileError as exc:
        return (
            None,
            "COMPILE_REJECTED",
            [row.code for row in exc.issues],
            [f"{row.code}:{row.location}:{row.message}" for row in exc.issues],
        )
    except Exception as exc:
        return None, "COMPILE_REJECTED", [type(exc).__name__], [str(exc)]
    validation = HypothesisValidator().validate(context, portfolio)
    if not validation.passes:
        issues = _validation_issues(validation)
        return (
            None,
            "VALIDATION_REJECTED",
            [row.split(":", 1)[0] for row in issues],
            issues,
        )
    return portfolio, "MATERIALIZED", [], []


def realize_offspring_bounded(
    *,
    execution: OffspringExecutionReport,
    context: HypothesisContext,
    backend: HypothesisDraftBackend,
    max_realizations: int = 6,
    max_per_parent: int = 2,
    max_repair_attempts: int = 1,
) -> tuple[OffspringRealizationReport, HypothesisPortfolio, tuple[HypothesisPrompt, ...]]:
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be >= 0")
    selected = select_offspring_for_realization(
        execution,
        max_realizations=max_realizations,
        max_per_parent=max_per_parent,
    )
    node_by_id = {row.idea_id: row for row in execution.offspring_nodes}
    semantic_by_id = {row.idea_id: row for row in execution.semantic_records}
    prompts: list[HypothesisPrompt] = []
    records: list[OffspringRealizationRecord] = []
    cards = []
    seen_hypothesis_ids: set[str] = set()

    for idea_id in selected:
        node = node_by_id[idea_id]
        semantic = semantic_by_id[idea_id]
        prompt = build_realization_prompt(
            context=context,
            node=node,
            semantic=semantic,
        )
        prompts.append(prompt)
        calls = 0
        repairs = 0
        input_tokens = 0
        output_tokens = 0
        try:
            generation = backend.generate(prompt)
            calls += 1
            input_tokens += int(generation.input_tokens or 0)
            output_tokens += int(generation.output_tokens or 0)
            draft = generation.draft
        except Exception as exc:
            records.append(
                OffspringRealizationRecord(
                    idea_id=idea_id,
                    channel=semantic.channel,
                    semantic_disposition=semantic.disposition,
                    status="GENERATION_FAILED",
                    issue_codes=[type(exc).__name__],
                    issues=[str(exc)],
                    generation_attempt_count=1,
                )
            )
            continue

        portfolio, status, issue_codes, issues = _compile_single_realization(
            context=context,
            draft=draft,
        )
        while (
            status in {"COMPILE_REJECTED", "VALIDATION_REJECTED", "CARDINALITY_REJECTED"}
            and repairs < max_repair_attempts
        ):
            repairs += 1
            feedback = "\n".join(
                [
                    "SIS-v2.3 grounded realization repair.",
                    "Keep the same ResearchIdea; do not replace it with a different easier idea.",
                    "Use only exact eligible positive-premise IDs from the supplied HypothesisContext.",
                    "Return exactly one corrected hypothesis or abstain.",
                    "Issues:",
                    *[f"- {row}" for row in issues],
                ]
            )
            try:
                repaired = backend.repair(prompt, draft, feedback)
                calls += 1
                input_tokens += int(repaired.input_tokens or 0)
                output_tokens += int(repaired.output_tokens or 0)
                draft = repaired.draft
                portfolio, status, issue_codes, issues = _compile_single_realization(
                    context=context,
                    draft=draft,
                )
            except Exception as exc:
                status = "GENERATION_FAILED"
                issue_codes = [type(exc).__name__]
                issues = [str(exc)]
                portfolio = None
                break

        hypothesis_id = None
        if portfolio is not None and portfolio.hypotheses:
            card = portfolio.hypotheses[0]
            hypothesis_id = str(card.hypothesis_id)
            if hypothesis_id in seen_hypothesis_ids:
                status = "COMPILE_REJECTED"
                issue_codes = [*issue_codes, "DUPLICATE_HYPOTHESIS_ID_ACROSS_OFFSPRING"]
                issues = [
                    *issues,
                    "A different G2 offspring materialized to an already-used hypothesis_id; "
                    "the duplicate realization was suppressed to preserve one-to-one lineage.",
                ]
                hypothesis_id = None
            else:
                seen_hypothesis_ids.add(hypothesis_id)
                cards.append(card)
        records.append(
            OffspringRealizationRecord(
                idea_id=idea_id,
                channel=semantic.channel,
                semantic_disposition=semantic.disposition,
                status=status,
                hypothesis_id=hypothesis_id,
                issue_codes=_dedupe(issue_codes),
                issues=_dedupe(issues),
                generation_attempt_count=calls,
                repair_attempt_count=repairs,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )

    card_ids = [str(card.hypothesis_id) for card in cards]
    portfolio_id = _stable_id(
        "g2_offspring_hypothesis_portfolio",
        execution.report_id,
        context.context_id,
        card_ids,
    )
    combined = HypothesisPortfolio(
        portfolio_id=portfolio_id,
        domain_profile_id=context.domain_profile_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        source_report_id=context.source_report_id,
        source_report_sha256=context.source_report_sha256,
        hypotheses=cards,
        abstention_reason=(
            None
            if cards
            else "No SIS-v2.3 G2 offspring could be grounded into a validated hypothesis realization."
        ),
    )
    status_counts = Counter(row.status for row in records)
    provisional = OffspringRealizationReport(
        report_id="pending",
        report_sha256="pending",
        source_offspring_execution_report_id=execution.report_id,
        source_context_id=context.context_id,
        source_context_sha256=context.context_sha256,
        selected_idea_ids=selected,
        selected_idea_count=len(selected),
        records=records,
        status_counts=dict(sorted(status_counts.items())),
        materialized_hypothesis_count=len(cards),
        materialization_success_fraction=(
            len(cards) / len(selected) if selected else 0.0
        ),
        output_portfolio_id=combined.portfolio_id,
        llm_call_count=sum(row.generation_attempt_count for row in records),
        input_tokens=sum(row.input_tokens for row in records),
        output_tokens=sum(row.output_tokens for row in records),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return (
        provisional.model_copy(
            update={
                "report_id": f"g2_offspring_realization:{digest[:20]}",
                "report_sha256": digest,
            }
        ),
        combined,
        tuple(prompts),
    )


def build_g3_feedback_seed(
    *,
    execution: OffspringExecutionReport,
    realization: OffspringRealizationReport,
) -> G3FeedbackSeedReport:
    if realization.source_offspring_execution_report_id != execution.report_id:
        raise ValueError("realization / offspring execution lineage mismatch")
    observations: list[IdeaOutcomeObservation] = []
    for row in realization.records:
        observations.append(
            IdeaOutcomeObservation(
                observation_id=_stable_id(
                    "g3_realization_observation",
                    execution.report_id,
                    row.idea_id,
                    row.hypothesis_id,
                    row.status,
                    row.issue_codes,
                ),
                idea_id=row.idea_id,
                target_scope="REALIZATION",
                source_systems=["SIS_V2_3_G2_REALIZATION"],
                source_versions=[realization.schema_version],
                candidate_id=f"g2_offspring:{row.idea_id}",
                hypothesis_id=row.hypothesis_id,
                materialization_status=row.status,
                materialization_issue_codes=list(row.issue_codes),
                source_report_ids=[realization.report_id],
            )
        )
    provisional = G3FeedbackSeedReport(
        report_id="pending",
        report_sha256="pending",
        source_offspring_execution_report_id=execution.report_id,
        source_realization_report_id=realization.report_id,
        generation2_idea_ids=[
            row.idea_id
            for row in execution.semantic_records
            if row.accepted_for_realization
        ],
        suppressed_generation2_attempt_idea_ids=[
            row.idea_id
            for row in execution.semantic_records
            if not row.accepted_for_realization
        ],
        realization_observations=observations,
        observation_count=len(observations),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"g3_feedback_seed:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def make_hypothesis_backend(
    *,
    model: str,
    api_key_env: str,
    base_url: str | None,
    instructor_mode: str,
    temperature: float,
    parse_retries: int,
    timeout: float | None,
    extra_headers: Mapping[str, str] | None,
    telemetry_path: str | Path | None,
    telemetry_context: Mapping[str, Any] | None,
) -> InstructorOpenAICompatibleHypothesisBackend:
    return InstructorOpenAICompatibleHypothesisBackend(
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        instructor_mode=instructor_mode,
        temperature=temperature,
        parse_retries=parse_retries,
        timeout=timeout,
        extra_headers=dict(extra_headers or {}),
        telemetry_path=telemetry_path,
        telemetry_context=telemetry_context,
    )


__all__ = [
    "G3FeedbackSeedReport",
    "GenerationalOffspringBackend",
    "GenerationalOffspringBatchDraft",
    "GenerationalOffspringCandidateDraft",
    "InstructorOpenAICompatibleOffspringBackend",
    "OffspringExecutionReport",
    "OffspringGenerationPlan",
    "OffspringGenerationTask",
    "OffspringRealizationReport",
    "OffspringSemanticRecord",
    "build_g3_feedback_seed",
    "build_offspring_generation_plan",
    "build_offspring_prompt",
    "build_realization_prompt",
    "execute_offspring_plan",
    "make_hypothesis_backend",
    "realize_offspring_bounded",
    "select_offspring_for_realization",
]
