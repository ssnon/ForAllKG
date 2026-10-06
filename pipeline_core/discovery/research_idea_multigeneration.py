from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_evolutionary_policy import (
    EvolutionChannel,
    EvolutionOperatorHint,
    EvolutionSlotBudget,
    IdeaEvolutionPolicyState,
    IdeaReproductiveValue,
    PolicyUncertainty,
    SignalLevel,
    derive_slot_budget,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    OffspringExecutionReport,
    OffspringRealizationReport,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ConceptualDeltaCategory = Literal[
    "SAME_CONCEPT",
    "REFINEMENT",
    "GENUINE_MUTATION",
    "COMPOSITION",
    "UNCERTAIN",
]
RealizationFertility = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]


_SIGNAL_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_FERTILITY_RANK = {"LOW": 0, "UNKNOWN": 1, "MEDIUM": 2, "HIGH": 3}
_UNCERTAINTY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_CONCEPTUAL_CATEGORY_RANK = {
    "SAME_CONCEPT": 0,
    "REFINEMENT": 1,
    "UNCERTAIN": 1,
    "GENUINE_MUTATION": 2,
    "COMPOSITION": 3,
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


class ConceptualDeltaAuditDraft(StrictModel):
    child_idea_id: str = Field(min_length=1)
    category: ConceptualDeltaCategory
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    identity_bearing_changes: list[str] = Field(default_factory=list)
    preserved_commitments: list[str] = Field(default_factory=list)


class ConceptualDeltaAuditBatchDraft(StrictModel):
    schema_version: Literal[
        "conceptual-delta-audit-batch-draft-v1"
    ] = "conceptual-delta-audit-batch-draft-v1"
    assessments: list[ConceptualDeltaAuditDraft] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> "ConceptualDeltaAuditBatchDraft":
        ids = [row.child_idea_id for row in self.assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("conceptual audit child_idea_id values must be unique")
        return self


@dataclass(frozen=True)
class ConceptualDeltaAuditGeneration:
    draft: ConceptualDeltaAuditBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class ConceptualDeltaAuditBackend(Protocol):
    backend_name: str
    model_name: str

    def audit(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        generation_label: str,
    ) -> ConceptualDeltaAuditGeneration: ...


class InstructorOpenAICompatibleConceptualDeltaBackend:
    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
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
        import instructor
        from openai import OpenAI

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

    def audit(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        generation_label: str,
    ) -> ConceptualDeltaAuditGeneration:
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=ConceptualDeltaAuditBatchDraft,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "sis_v2_4_conceptual_delta_audit",
                "stage": generation_label,
                "call_kind": "independent_conceptual_delta_audit",
            },
            semantic_components={
                "authority": "DIAGNOSTIC_ONLY",
                "deterministic_identity_result_visible_to_auditor": False,
            },
        )
        if not isinstance(draft, ConceptualDeltaAuditBatchDraft):
            draft = ConceptualDeltaAuditBatchDraft.model_validate(draft)
        return ConceptualDeltaAuditGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )


class ConceptualDeltaAuditRecord(StrictModel):
    child_idea_id: str = Field(min_length=1)
    parent_idea_ids: list[str] = Field(min_length=1)
    generation_index: int = Field(ge=2)
    deterministic_identity_relation: str = Field(min_length=1)
    deterministic_genealogy_relation: str = Field(min_length=1)
    audit_category: ConceptualDeltaCategory
    audit_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    identity_bearing_changes: list[str] = Field(default_factory=list)
    preserved_commitments: list[str] = Field(default_factory=list)
    comparator_audit_disagreement: bool
    strong_distinct_child: bool
    potential_semantic_inflation: bool
    diagnostic_only: Literal[True] = True
    candidate_deletion_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ConceptualDeltaAuditReport(StrictModel):
    schema_version: Literal[
        "conceptual-delta-calibration-shadow-v1"
    ] = "conceptual-delta-calibration-shadow-v1"
    report_id: str
    report_sha256: str
    source_offspring_execution_report_id: str
    generation_index: int = Field(ge=2)
    records: list[ConceptualDeltaAuditRecord] = Field(default_factory=list)
    audited_count: int = Field(ge=0)
    category_counts: dict[str, int] = Field(default_factory=dict)
    deterministic_identity_counts: dict[str, int] = Field(default_factory=dict)
    disagreement_count: int = Field(ge=0)
    strong_distinct_child_count: int = Field(ge=0)
    potential_semantic_inflation_count: int = Field(ge=0)
    mutation_attempt_count: int = Field(ge=0)
    calibrated_mutation_yield_fraction: float = Field(ge=0.0, le=1.0)
    llm_call_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    independent_auditor_not_shown_deterministic_result: Literal[True] = True
    calibration_is_not_hard_gate: Literal[True] = True
    candidate_deletion_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "ConceptualDeltaAuditReport":
        if self.audited_count != len(self.records):
            raise ValueError("audited_count mismatch")
        return self


_AUDIT_SYSTEM = """You are an independent conceptual-delta auditor for a scientific idea-evolution system.

Your task is to compare each CHILD ResearchIdea kernel only against its supplied PARENT kernel(s) and classify the scientific conceptual delta.

You are deliberately NOT shown the deterministic semantic comparator result. Do not infer or speculate about it.

Categories:
- SAME_CONCEPT: no identity-bearing scientific commitment changed; wording/prediction/test detail changed only.
- REFINEMENT: same core scientific idea, but scope, precision, operational framing, or a non-constitutive qualifier was sharpened.
- GENUINE_MUTATION: at least one substantive identity-bearing scientific commitment changed (mechanism, mediator, causal direction, latent construct, regime logic, proxy assumption, central contrast, or scientific question).
- COMPOSITION: the child substantively combines scientific commitments from multiple parents into a new joint idea; mere juxtaposition is not enough.
- UNCERTAIN: the supplied kernels do not support a confident classification.

Important boundaries:
- This audit is diagnostic only. It does not delete candidates, certify novelty, certify truth, or decide production selection.
- Do not reward lexical difference. Different words can still express the same concept.
- Do not punish imaginative scientific mutations merely because they lack current grounded evidence; grounding is evaluated at realization time.
- Do not treat prediction/falsifier changes alone as a new ResearchIdea.
- For a one-parent child, COMPOSITION is not appropriate.
- Return exactly one assessment for every supplied child_idea_id and no extras.
"""


def _kernel_view(node: ResearchIdeaNode) -> dict[str, Any]:
    return {
        "idea_id": node.idea_id,
        "generation_index": node.generation_index,
        "kernel": node.kernel.model_dump(mode="json"),
        "task_relation_mode": node.task_relation_mode,
    }


def build_conceptual_delta_audit_prompt(
    *,
    execution: OffspringExecutionReport,
    parent_nodes: Sequence[ResearchIdeaNode],
    research_question: str,
) -> tuple[str, str]:
    parent_by_id = {row.idea_id: row for row in parent_nodes}
    child_by_id = {row.idea_id: row for row in execution.offspring_nodes}
    rows = []
    for semantic in execution.semantic_records:
        child = child_by_id[semantic.idea_id]
        missing = [pid for pid in semantic.parent_idea_ids if pid not in parent_by_id]
        if missing:
            raise ValueError(f"conceptual audit missing parent nodes: {missing}")
        rows.append(
            {
                "child_idea_id": semantic.idea_id,
                "parents": [_kernel_view(parent_by_id[pid]) for pid in semantic.parent_idea_ids],
                "child": _kernel_view(child),
            }
        )
    payload = {
        "research_question": research_question,
        "generation_index": execution.generation_index,
        "comparisons": rows,
        "deterministic_comparator_result_included": False,
    }
    return _AUDIT_SYSTEM, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def run_conceptual_delta_audit(
    *,
    execution: OffspringExecutionReport,
    parent_nodes: Sequence[ResearchIdeaNode],
    research_question: str,
    backend: ConceptualDeltaAuditBackend,
) -> ConceptualDeltaAuditReport:
    system, user = build_conceptual_delta_audit_prompt(
        execution=execution,
        parent_nodes=parent_nodes,
        research_question=research_question,
    )
    generation = backend.audit(
        system_prompt=system,
        user_prompt=user,
        generation_label=f"G{execution.generation_index}",
    )
    expected_ids = [row.idea_id for row in execution.semantic_records]
    actual_ids = [row.child_idea_id for row in generation.draft.assessments]
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        raise ValueError(
            "conceptual audit cardinality/identity mismatch; "
            f"missing={missing}; extra={extra}"
        )
    draft_by_id = {row.child_idea_id: row for row in generation.draft.assessments}
    records: list[ConceptualDeltaAuditRecord] = []
    mutation_attempt_count = 0
    strong_mutation_count = 0
    for semantic in execution.semantic_records:
        draft = draft_by_id[semantic.idea_id]
        deterministic = semantic.transition.identity_relation
        expected_same = draft.category in {"SAME_CONCEPT", "REFINEMENT"}
        audit_distinct = draft.category in {"GENUINE_MUTATION", "COMPOSITION"}
        disagreement = (
            (deterministic == "DIFFERENT_IDEA" and expected_same)
            or (deterministic == "SAME_IDEA" and audit_distinct)
            or (deterministic == "INDETERMINATE" and draft.category != "UNCERTAIN")
        )
        composition_lineage_valid = (
            draft.category != "COMPOSITION" or len(semantic.parent_idea_ids) >= 2
        )
        strong = (
            deterministic == "DIFFERENT_IDEA"
            and audit_distinct
            and draft.confidence >= 0.60
            and composition_lineage_valid
        )
        inflation = (
            deterministic == "DIFFERENT_IDEA"
            and draft.category in {"SAME_CONCEPT", "REFINEMENT", "UNCERTAIN"}
        )
        if semantic.channel != "EXPLOIT":
            mutation_attempt_count += 1
            strong_mutation_count += int(strong)
        records.append(
            ConceptualDeltaAuditRecord(
                child_idea_id=semantic.idea_id,
                parent_idea_ids=list(semantic.parent_idea_ids),
                generation_index=execution.generation_index,
                deterministic_identity_relation=deterministic,
                deterministic_genealogy_relation=semantic.transition.genealogy_relation,
                audit_category=draft.category,
                audit_confidence=draft.confidence,
                rationale=draft.rationale,
                identity_bearing_changes=_dedupe(draft.identity_bearing_changes),
                preserved_commitments=_dedupe(draft.preserved_commitments),
                comparator_audit_disagreement=disagreement,
                strong_distinct_child=strong,
                potential_semantic_inflation=inflation,
            )
        )
    categories = Counter(row.audit_category for row in records)
    identities = Counter(row.deterministic_identity_relation for row in records)
    provisional = ConceptualDeltaAuditReport(
        report_id="pending",
        report_sha256="pending",
        source_offspring_execution_report_id=execution.report_id,
        generation_index=execution.generation_index,
        records=records,
        audited_count=len(records),
        category_counts=dict(sorted(categories.items())),
        deterministic_identity_counts=dict(sorted(identities.items())),
        disagreement_count=sum(row.comparator_audit_disagreement for row in records),
        strong_distinct_child_count=sum(row.strong_distinct_child for row in records),
        potential_semantic_inflation_count=sum(
            row.potential_semantic_inflation for row in records
        ),
        mutation_attempt_count=mutation_attempt_count,
        calibrated_mutation_yield_fraction=(
            strong_mutation_count / mutation_attempt_count
            if mutation_attempt_count
            else 0.0
        ),
        llm_call_count=1,
        input_tokens=int(generation.input_tokens or 0),
        output_tokens=int(generation.output_tokens or 0),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"g{execution.generation_index}_conceptual_delta:{digest[:20]}",
            "report_sha256": digest,
        }
    )


class CalibratedIdeaCreditState(StrictModel):
    idea_id: str = Field(min_length=1)
    inherited_channel: EvolutionChannel
    inherited_operator_id: str = Field(min_length=1)
    parent_idea_ids: list[str] = Field(min_length=1)
    deterministic_identity_relation: str = Field(min_length=1)
    audit_category: ConceptualDeltaCategory
    audit_confidence: float = Field(ge=0.0, le=1.0)
    comparator_audit_disagreement: bool
    realization_status: str = "NOT_SELECTED"
    conceptual_fertility: SignalLevel
    realization_fertility: RealizationFertility
    exploit_value: SignalLevel
    transformation_pressure: SignalLevel
    exploration_value: SignalLevel
    uncertainty: PolicyUncertainty
    idea_reproductive_value: IdeaReproductiveValue
    preferred_channels: list[EvolutionChannel] = Field(default_factory=list)
    operator_hints: list[EvolutionOperatorHint] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    verification_is_not_fertility_authority: Literal[True] = True
    calibration_is_not_fertility_authority: Literal[True] = True
    idea_termination_authority: Literal[False] = False


class CalibratedGenerationPopulation(StrictModel):
    schema_version: Literal[
        "calibrated-generation-population-v1"
    ] = "calibrated-generation-population-v1"
    report_id: str
    report_sha256: str
    generation_index: int = Field(ge=2)
    source_offspring_execution_report_id: str
    research_ideas: list[ResearchIdeaNode] = Field(default_factory=list)
    research_idea_count: int = Field(ge=0)
    shadow_only: Literal[True] = True
    production_selection_authority: Literal[False] = False


class CalibratedGenerationAllocation(StrictModel):
    idea_id: str = Field(min_length=1)
    channel: EvolutionChannel
    operator_hints: list[EvolutionOperatorHint] = Field(default_factory=list)
    conceptual_fertility: SignalLevel
    realization_fertility: RealizationFertility
    exploit_value: SignalLevel
    transformation_pressure: SignalLevel
    exploration_value: SignalLevel
    uncertainty: PolicyUncertainty
    audit_category: ConceptualDeltaCategory
    audit_confidence: float = Field(ge=0.0, le=1.0)
    inherited_channel: EvolutionChannel
    selection_reason_codes: list[str] = Field(default_factory=list)


class CalibratedGenerationSearchReport(StrictModel):
    schema_version: Literal[
        "calibrated-generation-search-shadow-v1"
    ] = "calibrated-generation-search-shadow-v1"
    report_id: str
    report_sha256: str
    source_generational_report_id: str
    source_generation_index: int = Field(ge=2)
    source_conceptual_audit_report_id: str
    source_realization_report_id: str
    policy_states: list[IdeaEvolutionPolicyState] = Field(default_factory=list)
    credit_states: list[CalibratedIdeaCreditState] = Field(default_factory=list)
    slot_budget: EvolutionSlotBudget
    allocations: list[CalibratedGenerationAllocation] = Field(default_factory=list)
    selected_parent_idea_ids: list[str] = Field(default_factory=list)
    selected_parent_count: int = Field(ge=0)
    selected_parent_channel_counts: dict[str, int] = Field(default_factory=dict)
    conceptual_fertility_counts: dict[str, int] = Field(default_factory=dict)
    realization_fertility_counts: dict[str, int] = Field(default_factory=dict)
    transformation_pressure_counts: dict[str, int] = Field(default_factory=dict)
    disagreement_parent_count: int = Field(ge=0)
    low_realization_transform_parent_count: int = Field(ge=0)
    single_scalar_fitness_used: Literal[False] = False
    calibration_is_not_hard_gate: Literal[True] = True
    verification_is_not_fertility_authority: Literal[True] = True
    conceptual_audit_is_not_fertility_authority: Literal[True] = True
    operator_hints_are_nonbinding: Literal[True] = True
    exact_kernel_duplicate_suppression_only: Literal[True] = True
    production_selection_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class ParentFertilityRecord(StrictModel):
    parent_idea_id: str
    offspring_count: int = Field(ge=0)
    deterministic_distinct_child_count: int = Field(ge=0)
    calibrated_strong_distinct_child_count: int = Field(ge=0)
    materialized_child_count: int = Field(ge=0)
    conceptual_fertility_and_realization_fertility_kept_separate: Literal[True] = True


class MultiGenerationCaseSummary(StrictModel):
    schema_version: Literal[
        "sis-v2-4-multigeneration-case-summary-v1"
    ] = "sis-v2-4-multigeneration-case-summary-v1"
    summary_id: str
    summary_sha256: str
    generation0_idea_count: int = Field(ge=0)
    generation1_idea_count: int = Field(ge=0)
    generation2_raw_offspring_count: int = Field(ge=0)
    generation2_deterministic_distinct_count: int = Field(ge=0)
    generation2_calibrated_strong_distinct_count: int = Field(ge=0)
    generation2_calibration_disagreement_count: int = Field(ge=0)
    generation2_materialized_hypothesis_count: int = Field(ge=0)
    generation2_calibrated_mutation_yield_fraction: float = Field(ge=0.0, le=1.0)
    generation3_parent_count: int = Field(ge=0)
    generation3_parent_channel_counts: dict[str, int] = Field(default_factory=dict)
    generation3_raw_offspring_count: int = Field(ge=0)
    generation3_deterministic_distinct_count: int = Field(ge=0)
    generation3_calibrated_strong_distinct_count: int = Field(ge=0)
    generation3_calibration_disagreement_count: int = Field(ge=0)
    generation3_calibrated_mutation_yield_fraction: float = Field(ge=0.0, le=1.0)
    generation2_potential_semantic_inflation_count: int = Field(ge=0)
    generation3_potential_semantic_inflation_count: int = Field(ge=0)
    cumulative_research_idea_node_count_through_g3: int = Field(ge=0)
    cumulative_exact_kernel_unique_count_through_g3: int = Field(ge=0)
    generation2_exact_kernel_collision_with_prior_count: int = Field(ge=0)
    generation3_exact_kernel_collision_with_prior_count: int = Field(ge=0)
    generation2_parent_fertility: list[ParentFertilityRecord] = Field(default_factory=list)
    generation3_parent_fertility: list[ParentFertilityRecord] = Field(default_factory=list)
    conceptual_audit_llm_calls: int = Field(ge=0)
    generation3_offspring_llm_calls: int = Field(ge=0)
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


_REALIZATION_LOW = {
    "ABSTAINED",
    "GENERATION_FAILED",
    "COMPILE_REJECTED",
    "VALIDATION_REJECTED",
    "CARDINALITY_REJECTED",
}


def _realization_fertility(status: str | None) -> RealizationFertility:
    if status == "MATERIALIZED":
        return "HIGH"
    if status in _REALIZATION_LOW:
        return "LOW"
    return "UNKNOWN"


def _conceptual_fertility(audit: ConceptualDeltaAuditRecord) -> SignalLevel:
    if audit.audit_category in {"GENUINE_MUTATION", "COMPOSITION"} and audit.audit_confidence >= 0.60:
        return "HIGH"
    if audit.audit_category == "SAME_CONCEPT" and audit.audit_confidence >= 0.70:
        return "LOW"
    return "MEDIUM"


def _credit_for_idea(
    *,
    semantic: Any,
    audit: ConceptualDeltaAuditRecord,
    realization_status: str | None,
) -> CalibratedIdeaCreditState:
    conceptual = _conceptual_fertility(audit)
    realization = _realization_fertility(realization_status)
    exploit: SignalLevel = "MEDIUM"
    transform: SignalLevel = "MEDIUM"
    explore: SignalLevel = "MEDIUM"
    uncertainty: PolicyUncertainty = "MEDIUM"
    reasons: list[str] = []
    channels: list[EvolutionChannel] = []
    hints: list[EvolutionOperatorHint] = []

    if realization == "HIGH":
        exploit = "HIGH"
        reasons.append("G2_REALIZATION_MATERIALIZED_SUPPORTS_EXPLOITATION")
        channels.append("EXPLOIT")
        hints.extend(["SAME_PREMISE_SHARPEN", "EVIDENCE_REAXIS"])
    elif realization == "LOW":
        exploit = "LOW"
        transform = "HIGH"
        explore = "HIGH"
        reasons.append("G2_REALIZATION_FAILURE_RAISES_TRANSFORMATION_NOT_TERMINATION")
        channels.extend(["TRANSFORM", "EXPLORE"])
        hints.extend(["AXIS_MUTATION", "REGIME_BOUNDARY", "PROXY_CHALLENGE"])
    else:
        explore = "HIGH"
        reasons.append("G2_REALIZATION_UNOBSERVED_PRESERVES_EXPLORATION")
        channels.extend(["EXPLORE", "WILDCARD"])

    if conceptual == "HIGH":
        explore = "HIGH"
        reasons.append("CALIBRATED_CONCEPTUAL_DELTA_SUPPORTS_EXPLORATION")
        channels.append("EXPLORE")
        hints.extend(["CROSS_SOURCE_BRIDGE", "LATENT_VARIABLE", "CANDIDATE_INTERPRETATION"])
    elif conceptual == "LOW":
        transform = "HIGH"
        reasons.append("LOW_CONCEPTUAL_DELTA_RAISES_TRANSFORMATION_PRESSURE")
        channels.append("TRANSFORM")
        hints.extend(["AXIS_MUTATION", "BACKBONE_MUTATION", "REGIME_BOUNDARY"])

    if audit.comparator_audit_disagreement:
        transform = "HIGH"
        uncertainty = "HIGH"
        reasons.append("COMPARATOR_AUDIT_DISAGREEMENT_IS_SEARCH_SIGNAL_NOT_REJECTION")
        channels.extend(["TRANSFORM", "WILDCARD"])
        hints.extend(["AXIS_MUTATION", "CANDIDATE_INTERPRETATION"])
    elif audit.audit_category == "UNCERTAIN" or audit.audit_confidence < 0.60:
        uncertainty = "HIGH"
        reasons.append("CONCEPTUAL_DELTA_UNCERTAINTY_PRESERVES_WILDCARD")
        channels.append("WILDCARD")
    elif realization == "HIGH":
        uncertainty = "LOW"

    if transform == "HIGH":
        channels.insert(0, "TRANSFORM")
    if exploit == "HIGH":
        channels.insert(0, "EXPLOIT")
    if not channels:
        channels = ["EXPLORE", "WILDCARD"]

    reproductive: IdeaReproductiveValue = (
        "ENCOURAGED"
        if conceptual == "HIGH" or transform == "HIGH" or explore == "HIGH"
        else "PRESERVED"
    )
    return CalibratedIdeaCreditState(
        idea_id=semantic.idea_id,
        inherited_channel=semantic.channel,
        inherited_operator_id=semantic.chosen_operator_id,
        parent_idea_ids=list(semantic.parent_idea_ids),
        deterministic_identity_relation=semantic.transition.identity_relation,
        audit_category=audit.audit_category,
        audit_confidence=audit.audit_confidence,
        comparator_audit_disagreement=audit.comparator_audit_disagreement,
        realization_status=str(realization_status or "NOT_SELECTED"),
        conceptual_fertility=conceptual,
        realization_fertility=realization,
        exploit_value=exploit,
        transformation_pressure=transform,
        exploration_value=explore,
        uncertainty=uncertainty,
        idea_reproductive_value=reproductive,
        preferred_channels=_dedupe(channels),
        operator_hints=list(dict.fromkeys(hints)),
        reason_codes=_dedupe(reasons),
    )


def _as_policy_state(credit: CalibratedIdeaCreditState, source_ids: Sequence[str]) -> IdeaEvolutionPolicyState:
    return IdeaEvolutionPolicyState(
        idea_id=credit.idea_id,
        observation_ids=_dedupe(source_ids),
        observation_scopes=["REALIZATION", "IDEA_DIAGNOSTIC"],
        realization_viability=credit.realization_fertility,
        idea_reproductive_value=credit.idea_reproductive_value,
        exploit_value=credit.exploit_value,
        transformation_pressure=credit.transformation_pressure,
        exploration_value=credit.exploration_value,
        uncertainty=credit.uncertainty,
        preferred_channels=credit.preferred_channels,
        operator_hints=credit.operator_hints,
        reason_codes=credit.reason_codes,
        hard_constraint_codes=[],
        soft_feedback_only=True,
    )


def _allocation_key(
    credit: CalibratedIdeaCreditState,
    *,
    channel: EvolutionChannel,
    lineage_already_selected: bool,
) -> tuple[Any, ...]:
    lineage_penalty = int(lineage_already_selected)
    if channel == "EXPLOIT":
        return (
            -_SIGNAL_RANK[credit.exploit_value],
            -_FERTILITY_RANK[credit.realization_fertility],
            -_SIGNAL_RANK[credit.conceptual_fertility],
            _UNCERTAINTY_RANK[credit.uncertainty],
            lineage_penalty,
            -credit.audit_confidence,
            credit.idea_id,
        )
    if channel == "TRANSFORM":
        return (
            -_SIGNAL_RANK[credit.transformation_pressure],
            _FERTILITY_RANK[credit.realization_fertility],
            -int(credit.comparator_audit_disagreement),
            -_SIGNAL_RANK[credit.exploration_value],
            lineage_penalty,
            credit.idea_id,
        )
    if channel == "EXPLORE":
        return (
            -_SIGNAL_RANK[credit.exploration_value],
            -_SIGNAL_RANK[credit.conceptual_fertility],
            -_UNCERTAINTY_RANK[credit.uncertainty],
            lineage_penalty,
            credit.idea_id,
        )
    return (
        -_UNCERTAINTY_RANK[credit.uncertainty],
        -int(credit.comparator_audit_disagreement),
        _FERTILITY_RANK[credit.realization_fertility],
        lineage_penalty,
        credit.idea_id,
    )


def build_calibrated_g3_search(
    *,
    execution: OffspringExecutionReport,
    realization: OffspringRealizationReport,
    conceptual_audit: ConceptualDeltaAuditReport,
    max_parent_budget: int = 8,
) -> tuple[CalibratedGenerationPopulation, CalibratedGenerationSearchReport]:
    if realization.source_offspring_execution_report_id != execution.report_id:
        raise ValueError("G2 realization / execution lineage mismatch")
    if conceptual_audit.source_offspring_execution_report_id != execution.report_id:
        raise ValueError("G2 conceptual audit / execution lineage mismatch")
    semantic_by_id = {row.idea_id: row for row in execution.semantic_records}
    audit_by_id = {row.child_idea_id: row for row in conceptual_audit.records}
    realization_by_id = {row.idea_id: row for row in realization.records}
    eligible_ids = [
        row.idea_id
        for row in execution.semantic_records
        if row.accepted_for_realization
    ]
    node_by_id = {row.idea_id: row for row in execution.offspring_nodes}
    missing = [idea_id for idea_id in eligible_ids if idea_id not in audit_by_id]
    if missing:
        raise ValueError(f"G2 credit missing conceptual audit rows: {missing}")
    nodes = [node_by_id[idea_id] for idea_id in eligible_ids]
    population_payload = {
        "generation_index": execution.generation_index,
        "source_execution": execution.report_id,
        "idea_ids": eligible_ids,
    }
    pop_digest = _sha(population_payload)
    population = CalibratedGenerationPopulation(
        report_id=f"g2_calibrated_population:{pop_digest[:20]}",
        report_sha256=pop_digest,
        generation_index=execution.generation_index,
        source_offspring_execution_report_id=execution.report_id,
        research_ideas=nodes,
        research_idea_count=len(nodes),
    )
    credits: list[CalibratedIdeaCreditState] = []
    for idea_id in eligible_ids:
        semantic = semantic_by_id[idea_id]
        audit = audit_by_id[idea_id]
        real = realization_by_id.get(idea_id)
        credits.append(
            _credit_for_idea(
                semantic=semantic,
                audit=audit,
                realization_status=(real.status if real is not None else None),
            )
        )
    credit_by_id = {row.idea_id: row for row in credits}
    policy_states = [
        _as_policy_state(
            row,
            [conceptual_audit.report_id, realization.report_id],
        )
        for row in credits
    ]

    budget_size = min(max_parent_budget, max(1, len(credits)))
    budget = derive_slot_budget(budget_size)
    selected_ids: list[str] = []
    selected_kernels: set[str] = set()
    selected_primary_lineages: set[str] = set()
    allocations: list[CalibratedGenerationAllocation] = []

    def eligible(credit: CalibratedIdeaCreditState, channel: EvolutionChannel) -> bool:
        if credit.idea_id in selected_ids:
            return False
        if node_by_id[credit.idea_id].kernel_sha256 in selected_kernels:
            return False
        if channel == "EXPLOIT":
            return credit.exploit_value != "LOW" and credit.realization_fertility == "HIGH"
        if channel == "TRANSFORM":
            return credit.transformation_pressure != "LOW"
        if channel == "EXPLORE":
            return credit.exploration_value != "LOW"
        return True

    def select_one(channel: EvolutionChannel) -> bool:
        pool = [row for row in credits if eligible(row, channel)]
        if not pool:
            return False
        pool.sort(
            key=lambda row: _allocation_key(
                row,
                channel=channel,
                lineage_already_selected=(row.parent_idea_ids[0] in selected_primary_lineages),
            )
        )
        row = pool[0]
        selected_ids.append(row.idea_id)
        selected_kernels.add(node_by_id[row.idea_id].kernel_sha256)
        selected_primary_lineages.add(row.parent_idea_ids[0])
        allocations.append(
            CalibratedGenerationAllocation(
                idea_id=row.idea_id,
                channel=channel,
                operator_hints=row.operator_hints,
                conceptual_fertility=row.conceptual_fertility,
                realization_fertility=row.realization_fertility,
                exploit_value=row.exploit_value,
                transformation_pressure=row.transformation_pressure,
                exploration_value=row.exploration_value,
                uncertainty=row.uncertainty,
                audit_category=row.audit_category,
                audit_confidence=row.audit_confidence,
                inherited_channel=row.inherited_channel,
                selection_reason_codes=[
                    f"G3_{channel}_SLOT",
                    "CALIBRATED_MULTI_AXIS_NONSCALAR_ORDERING",
                    "VERIFICATION_NOT_FERTILITY_AUTHORITY",
                    "CONCEPTUAL_AUDIT_NOT_HARD_GATE",
                    "EXACT_KERNEL_NOT_ALREADY_SELECTED",
                ],
            )
        )
        return True

    quotas = (
        ("EXPLOIT", budget.exploit_slots),
        ("TRANSFORM", budget.transform_slots),
        ("EXPLORE", budget.explore_slots),
        ("WILDCARD", budget.wildcard_slots),
    )
    for channel, quota in quotas:
        for _ in range(quota):
            if not select_one(channel):
                break
    refill_cycle: tuple[EvolutionChannel, ...] = (
        "TRANSFORM",
        "EXPLORE",
        "EXPLOIT",
        "WILDCARD",
    )
    while len(selected_ids) < budget.max_parent_budget:
        progress = False
        for channel in refill_cycle:
            if select_one(channel):
                allocations[-1].selection_reason_codes.append("UNUSED_SLOT_REFILL")
                progress = True
                if len(selected_ids) >= budget.max_parent_budget:
                    break
        if not progress:
            break

    counts = Counter(row.channel for row in allocations)
    provisional = CalibratedGenerationSearchReport(
        report_id="pending",
        report_sha256="pending",
        source_generational_report_id=population.report_id,
        source_generation_index=execution.generation_index,
        source_conceptual_audit_report_id=conceptual_audit.report_id,
        source_realization_report_id=realization.report_id,
        policy_states=policy_states,
        credit_states=credits,
        slot_budget=budget,
        allocations=allocations,
        selected_parent_idea_ids=selected_ids,
        selected_parent_count=len(selected_ids),
        selected_parent_channel_counts=dict(sorted(counts.items())),
        conceptual_fertility_counts=dict(
            sorted(Counter(row.conceptual_fertility for row in credits).items())
        ),
        realization_fertility_counts=dict(
            sorted(Counter(row.realization_fertility for row in credits).items())
        ),
        transformation_pressure_counts=dict(
            sorted(Counter(row.transformation_pressure for row in credits).items())
        ),
        disagreement_parent_count=sum(
            credit_by_id[row.idea_id].comparator_audit_disagreement
            for row in allocations
        ),
        low_realization_transform_parent_count=sum(
            row.channel == "TRANSFORM" and row.realization_fertility == "LOW"
            for row in allocations
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    report = provisional.model_copy(
        update={
            "report_id": f"g3_calibrated_search:{digest[:20]}",
            "report_sha256": digest,
        }
    )
    return population, report


def _parent_fertility(
    *,
    execution: OffspringExecutionReport,
    audit: ConceptualDeltaAuditReport,
    realization: OffspringRealizationReport | None,
) -> list[ParentFertilityRecord]:
    audit_by_id = {row.child_idea_id: row for row in audit.records}
    materialized = set()
    if realization is not None:
        materialized = {
            row.idea_id for row in realization.records if row.status == "MATERIALIZED"
        }
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in execution.semantic_records:
        for parent_id in row.parent_idea_ids:
            grouped[parent_id].append(row)
    result = []
    for parent_id, rows in sorted(grouped.items()):
        result.append(
            ParentFertilityRecord(
                parent_idea_id=parent_id,
                offspring_count=len(rows),
                deterministic_distinct_child_count=sum(
                    row.transition.identity_relation == "DIFFERENT_IDEA" for row in rows
                ),
                calibrated_strong_distinct_child_count=sum(
                    audit_by_id[row.idea_id].strong_distinct_child for row in rows
                ),
                materialized_child_count=sum(row.idea_id in materialized for row in rows),
            )
        )
    return result


def build_multigeneration_case_summary(
    *,
    source_generational_report: Any,
    g2_execution: OffspringExecutionReport,
    g2_audit: ConceptualDeltaAuditReport,
    g2_realization: OffspringRealizationReport,
    g3_search: CalibratedGenerationSearchReport,
    g3_execution: OffspringExecutionReport,
    g3_audit: ConceptualDeltaAuditReport,
) -> MultiGenerationCaseSummary:
    source_nodes = list(source_generational_report.research_ideas)
    g2_nodes = list(g2_execution.offspring_nodes)
    g3_nodes = list(g3_execution.offspring_nodes)
    prior_kernel_set = {row.kernel_sha256 for row in source_nodes}
    g2_kernel_set = {row.kernel_sha256 for row in g2_nodes}
    through_g2 = prior_kernel_set | g2_kernel_set
    g3_kernel_set = {row.kernel_sha256 for row in g3_nodes}

    provisional = MultiGenerationCaseSummary(
        summary_id="pending",
        summary_sha256="pending",
        generation0_idea_count=int(source_generational_report.generation0_idea_count),
        generation1_idea_count=int(source_generational_report.generation1_idea_count),
        generation2_raw_offspring_count=g2_execution.raw_offspring_count,
        generation2_deterministic_distinct_count=g2_execution.distinct_child_count,
        generation2_calibrated_strong_distinct_count=sum(
            row.strong_distinct_child and _semantic_channel(g2_execution, row.child_idea_id) != "EXPLOIT"
            for row in g2_audit.records
        ),
        generation2_calibration_disagreement_count=g2_audit.disagreement_count,
        generation2_materialized_hypothesis_count=g2_realization.materialized_hypothesis_count,
        generation2_calibrated_mutation_yield_fraction=g2_audit.calibrated_mutation_yield_fraction,
        generation3_parent_count=g3_search.selected_parent_count,
        generation3_parent_channel_counts=g3_search.selected_parent_channel_counts,
        generation3_raw_offspring_count=g3_execution.raw_offspring_count,
        generation3_deterministic_distinct_count=g3_execution.distinct_child_count,
        generation3_calibrated_strong_distinct_count=sum(
            row.strong_distinct_child and _semantic_channel(g3_execution, row.child_idea_id) != "EXPLOIT"
            for row in g3_audit.records
        ),
        generation3_calibration_disagreement_count=g3_audit.disagreement_count,
        generation3_calibrated_mutation_yield_fraction=g3_audit.calibrated_mutation_yield_fraction,
        generation2_potential_semantic_inflation_count=g2_audit.potential_semantic_inflation_count,
        generation3_potential_semantic_inflation_count=g3_audit.potential_semantic_inflation_count,
        cumulative_research_idea_node_count_through_g3=(
            len(source_nodes) + len(g2_nodes) + len(g3_nodes)
        ),
        cumulative_exact_kernel_unique_count_through_g3=len(
            prior_kernel_set | g2_kernel_set | g3_kernel_set
        ),
        generation2_exact_kernel_collision_with_prior_count=sum(
            row.kernel_sha256 in prior_kernel_set for row in g2_nodes
        ),
        generation3_exact_kernel_collision_with_prior_count=sum(
            row.kernel_sha256 in through_g2 for row in g3_nodes
        ),
        generation2_parent_fertility=_parent_fertility(
            execution=g2_execution,
            audit=g2_audit,
            realization=g2_realization,
        ),
        generation3_parent_fertility=_parent_fertility(
            execution=g3_execution,
            audit=g3_audit,
            realization=None,
        ),
        conceptual_audit_llm_calls=g2_audit.llm_call_count + g3_audit.llm_call_count,
        generation3_offspring_llm_calls=g3_execution.llm_call_count,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("summary_id", None)
    payload.pop("summary_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "summary_id": f"sis_v2_4_case_summary:{digest[:20]}",
            "summary_sha256": digest,
        }
    )


def _semantic_channel(execution: OffspringExecutionReport, idea_id: str) -> str:
    for row in execution.semantic_records:
        if row.idea_id == idea_id:
            return row.channel
    return "UNKNOWN"


__all__ = [
    "CalibratedGenerationPopulation",
    "CalibratedGenerationSearchReport",
    "CalibratedIdeaCreditState",
    "ConceptualDeltaAuditBackend",
    "ConceptualDeltaAuditBatchDraft",
    "ConceptualDeltaAuditDraft",
    "ConceptualDeltaAuditGeneration",
    "ConceptualDeltaAuditRecord",
    "ConceptualDeltaAuditReport",
    "InstructorOpenAICompatibleConceptualDeltaBackend",
    "MultiGenerationCaseSummary",
    "ParentFertilityRecord",
    "build_calibrated_g3_search",
    "build_conceptual_delta_audit_prompt",
    "build_multigeneration_case_summary",
    "run_conceptual_delta_audit",
]
