from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.ablation_evaluation import (
    AblationCondition,
    AblationDimensionId,
    BlindAblationComparison,
    ScientificReasoningAblationBlindKey,
    ScientificReasoningAblationPacket,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


BlindPreference = Literal["ARM_A", "ARM_B", "TIE", "UNCLEAR"]
UnblindedPreference = Literal[
    "RELATIONAL_ONLY",
    "REFRAMING_ONLY",
    "COMBINED",
    "TIE",
    "UNCLEAR",
]


class BlindCandidateReference(StrictModel):
    arm_alias: Literal["ARM_A", "ARM_B"]
    candidate_alias: str = Field(pattern=r"^C\d{2,}$")


class BlindDimensionJudgmentDraft(StrictModel):
    dimension_id: AblationDimensionId
    preference: BlindPreference
    rationale: str = Field(min_length=1)
    supporting_evidence_aliases: list[str] = Field(default_factory=list)
    supporting_candidate_refs: list[BlindCandidateReference] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    candidate_count_influenced_judgment: Literal[False] = False


class BlindComparisonJudgmentDraft(StrictModel):
    comparison_alias: str = Field(pattern=r"^COMPARISON_\d{2,}$")
    dimensions: list[BlindDimensionJudgmentDraft]
    candidate_count_treated_as_quality_signal: Literal[False] = False
    source_condition_inference_attempted: Literal[False] = False
    source_lane_or_operator_inference_attempted: Literal[False] = False

    @model_validator(mode="after")
    def validate_dimensions(self) -> "BlindComparisonJudgmentDraft":
        ids = [row.dimension_id for row in self.dimensions]
        if len(ids) != len(set(ids)):
            raise ValueError("blind evaluation contains duplicate dimensions")
        return self


class BlindDimensionJudgment(BlindDimensionJudgmentDraft):
    judgment_id: str = Field(min_length=1)


class BlindComparisonEvaluation(StrictModel):
    comparison_alias: str = Field(pattern=r"^COMPARISON_\d{2,}$")
    arm_a_candidate_count: int = Field(ge=0)
    arm_b_candidate_count: int = Field(ge=0)
    absolute_candidate_count_difference: int = Field(ge=0)
    dimensions: list[BlindDimensionJudgment]
    candidate_count_treated_as_quality_signal: Literal[False] = False
    source_condition_inference_attempted: Literal[False] = False
    source_lane_or_operator_inference_attempted: Literal[False] = False


class BlindJudgeCallRecord(StrictModel):
    comparison_alias: str = Field(pattern=r"^COMPARISON_\d{2,}$")
    response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    elapsed_seconds: float | None = Field(default=None, ge=0)


class BlindScientificReasoningEvaluationReport(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-ablation-blind-evaluation-v1"
    ] = "scientific-reasoning-ablation-blind-evaluation-v1"

    report_id: str = Field(min_length=1)
    source_packet_id: str = Field(min_length=1)
    source_packet_sha256: str = Field(min_length=64, max_length=64)
    backend_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    comparison_evaluations: list[BlindComparisonEvaluation]
    call_records: list[BlindJudgeCallRecord]
    llm_calls_performed: int = Field(ge=0)

    blind_key_loaded_by_evaluator: Literal[False] = False
    source_condition_labels_hidden_from_evaluator: Literal[True] = True
    source_candidate_ids_hidden_from_evaluator: Literal[True] = True
    source_lane_labels_hidden_from_evaluator: Literal[True] = True
    source_operator_labels_hidden_from_evaluator: Literal[True] = True
    candidate_count_is_not_quality_signal: Literal[True] = True
    per_dimension_pairwise_judgment_only: Literal[True] = True
    overall_score_computed: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    scientific_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "BlindScientificReasoningEvaluationReport":
        aliases = [row.comparison_alias for row in self.comparison_evaluations]
        if len(aliases) != len(set(aliases)):
            raise ValueError("blind evaluation report contains duplicate comparisons")
        call_aliases = [row.comparison_alias for row in self.call_records]
        if aliases != call_aliases:
            raise ValueError("call records must match comparison evaluations in stable order")
        if self.llm_calls_performed != len(self.call_records):
            raise ValueError("llm_calls_performed must equal call_records length")
        return self


class UnblindedCandidateReference(StrictModel):
    arm_alias: Literal["ARM_A", "ARM_B"]
    condition: AblationCondition
    candidate_alias: str
    source_candidate_id: str


class UnblindedDimensionJudgment(StrictModel):
    dimension_id: AblationDimensionId
    blind_preference: BlindPreference
    preferred_condition: UnblindedPreference
    rationale: str
    supporting_statement_ids: list[str] = Field(default_factory=list)
    supporting_candidate_refs: list[UnblindedCandidateReference] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    candidate_count_influenced_judgment: Literal[False] = False


class UnblindedComparisonEvaluation(StrictModel):
    comparison_alias: str
    comparison_kind: str
    arm_a_condition: AblationCondition
    arm_b_condition: AblationCondition
    dimensions: list[UnblindedDimensionJudgment]
    preference_counts: dict[str, int]
    overall_winner_selected: Literal[False] = False
    composite_score_computed: Literal[False] = False


class UnblindedScientificReasoningEvaluationReport(StrictModel):
    schema_version: Literal[
        "scientific-reasoning-ablation-unblinded-evaluation-v1"
    ] = "scientific-reasoning-ablation-unblinded-evaluation-v1"

    report_id: str
    source_blind_report_id: str
    source_blind_report_sha256: str = Field(min_length=64, max_length=64)
    source_packet_id: str
    source_packet_sha256: str = Field(min_length=64, max_length=64)
    source_task_id: str
    source_context_id: str
    source_context_sha256: str
    comparisons: list[UnblindedComparisonEvaluation]

    unblinding_performed_after_blind_evaluation: Literal[True] = True
    blind_key_used_only_for_unblinding: Literal[True] = True
    per_dimension_results_only: Literal[True] = True
    overall_score_computed: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    scientific_authority_created: Literal[False] = False


@dataclass(frozen=True)
class AblationBlindEvaluationPrompt:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class AblationBlindEvaluationGeneration:
    draft: BlindComparisonJudgmentDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class AblationBlindEvaluationBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(
        self, prompt: AblationBlindEvaluationPrompt
    ) -> AblationBlindEvaluationGeneration: ...


class InstructorOpenAICompatibleAblationJudgeBackend:
    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        instructor_mode: str = "JSON",
        temperature: float = 0.0,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
        extra_headers: dict[str, str] | None = None,
        telemetry_path: str | Path | None = None,
        telemetry_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = str(model)
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env)
        self.api_key_env = api_key_env
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
            raise RuntimeError(
                f"No API key available. Set {self.api_key_env} or pass api_key explicitly."
            )
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Blind scientific-reasoning evaluation requires installed openai and instructor packages."
            ) from exc
        mode = getattr(instructor.Mode, self.instructor_mode, None)
        if mode is None:
            available = sorted(name for name in dir(instructor.Mode) if name.isupper())
            raise ValueError(
                f"Unknown Instructor mode {self.instructor_mode!r}. Available modes include: {available}"
            )
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers
        self._client = instructor.from_openai(OpenAI(**kwargs), mode=mode)
        return self._client

    def generate(
        self, prompt: AblationBlindEvaluationPrompt
    ) -> AblationBlindEvaluationGeneration:
        from pipeline_core.llm.llm_telemetry import run_instructor_structured_call

        started = time.perf_counter()
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=BlindComparisonJudgmentDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "scientific_reasoning_ablation",
                "stage": "blind_pairwise_evaluation",
                "call_kind": "scientific_reasoning_blind_judge",
            },
            semantic_components={"evaluation_kind": "reasoning_ablation_pairwise"},
        )
        elapsed = event.elapsed_seconds
        if elapsed is None:
            elapsed = time.perf_counter() - started
        if not isinstance(draft, BlindComparisonJudgmentDraft):
            draft = BlindComparisonJudgmentDraft.model_validate(draft)
        return AblationBlindEvaluationGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=elapsed,
        )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical_json(parts).encode('utf-8')).hexdigest()[:20]}"


def build_ablation_blind_evaluation_prompt(
    *,
    packet: ScientificReasoningAblationPacket,
    comparison: BlindAblationComparison,
) -> AblationBlindEvaluationPrompt:
    system_prompt = """You are a blind scientific-reasoning evaluator.

You are comparing two anonymized hypothesis portfolios for the SAME scientific task and the SAME evidence packet. You do not know, and must not infer, which system or reasoning lane produced either arm.

For every supplied evaluation dimension, return exactly one of ARM_A, ARM_B, TIE, or UNCLEAR.

Rules:
- Judge scientific reasoning quality, not writing style or candidate count.
- A larger portfolio is NOT evidence of better quality. Do not reward an arm merely because it contains more candidates.
- Use only the supplied task, evidence aliases, and anonymized candidates. Do not add outside facts or literature knowledge.
- Preserve epistemic discipline: unresolved or hypothetical claims must not be treated as established facts.
- Prefer TIE when the arms are materially comparable on a dimension.
- Prefer UNCLEAR when the supplied material does not support a reliable comparison.
- For task_relevance_and_coverage, judge whether the portfolio addresses the task directly and substantively, not how many candidates it contains.
- For evidence_discipline, penalize unsupported strengthening and misuse of gaps/unresolved evidence.
- For explanatory_gain, distinguish genuine changes in explanatory representation from paraphrase or relabeling.
- For differential_prediction_quality, look for outcomes that separate plausible models/explanations.
- For falsifiability, look for credible observable outcomes that would count against the proposal.
- For discriminating_experiment_quality, judge whether a test distinguishes alternatives rather than merely collecting additional measurements.
- For portfolio_complementarity, judge whether the lines of reasoning are scientifically complementary. Do not equate portfolio size with complementarity.
- Cite only aliases that appear in the supplied packet.
- Set candidate_count_influenced_judgment=false for every dimension.
- Do not infer hidden source conditions, lane identities, operator identities, model families, or candidate provenance.
- There is no overall winner and no composite score in this task.
"""
    payload = {
        "task_alias": packet.task_alias,
        "question": packet.question,
        "evidence_statements": [
            row.model_dump(mode="json") for row in packet.evidence_statements
        ],
        "evaluation_dimensions": [
            row.model_dump(mode="json") for row in packet.evaluation_dimensions
        ],
        "comparison": comparison.model_dump(mode="json"),
        "output_contract": {
            "comparison_alias": comparison.comparison_alias,
            "exact_dimension_ids": [
                row.dimension_id for row in packet.evaluation_dimensions
            ],
            "allowed_preferences": ["ARM_A", "ARM_B", "TIE", "UNCLEAR"],
            "candidate_count_is_not_quality_signal": True,
            "overall_winner_requested": False,
            "composite_score_requested": False,
        },
    }
    user_prompt = (
        "Evaluate this single blind comparison. Return all dimensions exactly once. "
        "Do not attempt to identify the hidden source conditions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return AblationBlindEvaluationPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _compile_dimension(
    *,
    packet: ScientificReasoningAblationPacket,
    comparison: BlindAblationComparison,
    draft: BlindDimensionJudgmentDraft,
) -> BlindDimensionJudgment:
    evidence_aliases = {row.evidence_alias for row in packet.evidence_statements}
    unknown_evidence = sorted(set(draft.supporting_evidence_aliases) - evidence_aliases)
    if unknown_evidence:
        raise ValueError(
            "blind evaluator referenced unknown evidence aliases: "
            + ", ".join(unknown_evidence)
        )

    candidates_by_arm = {
        "ARM_A": {row.candidate_alias for row in comparison.arm_a.candidates},
        "ARM_B": {row.candidate_alias for row in comparison.arm_b.candidates},
    }
    seen_refs: set[tuple[str, str]] = set()
    for ref in draft.supporting_candidate_refs:
        if ref.candidate_alias not in candidates_by_arm[ref.arm_alias]:
            raise ValueError(
                "blind evaluator referenced candidate alias outside declared arm: "
                f"{ref.arm_alias}/{ref.candidate_alias}"
            )
        key = (ref.arm_alias, ref.candidate_alias)
        if key in seen_refs:
            raise ValueError("blind evaluator candidate references must be unique")
        seen_refs.add(key)

    if draft.preference == "ARM_A" and not any(
        ref.arm_alias == "ARM_A" for ref in draft.supporting_candidate_refs
    ):
        raise ValueError("ARM_A preference requires at least one ARM_A supporting candidate")
    if draft.preference == "ARM_B" and not any(
        ref.arm_alias == "ARM_B" for ref in draft.supporting_candidate_refs
    ):
        raise ValueError("ARM_B preference requires at least one ARM_B supporting candidate")

    judgment_id = _stable_id(
        "ablation_dimension_judgment",
        packet.packet_id,
        comparison.comparison_alias,
        draft.dimension_id,
        draft.preference,
        draft.rationale,
        *sorted(draft.supporting_evidence_aliases),
        *sorted(f"{row.arm_alias}/{row.candidate_alias}" for row in draft.supporting_candidate_refs),
    )
    return BlindDimensionJudgment(
        **draft.model_dump(mode="python"),
        judgment_id=judgment_id,
    )


def compile_blind_comparison_evaluation(
    *,
    packet: ScientificReasoningAblationPacket,
    comparison: BlindAblationComparison,
    draft: BlindComparisonJudgmentDraft,
) -> BlindComparisonEvaluation:
    if draft.comparison_alias != comparison.comparison_alias:
        raise ValueError(
            "blind evaluator comparison_alias mismatch: "
            f"expected={comparison.comparison_alias}, actual={draft.comparison_alias}"
        )
    expected_dimensions = [row.dimension_id for row in packet.evaluation_dimensions]
    actual_dimensions = [row.dimension_id for row in draft.dimensions]
    if set(actual_dimensions) != set(expected_dimensions) or len(actual_dimensions) != len(
        expected_dimensions
    ):
        raise ValueError(
            "blind evaluator must return every evaluation dimension exactly once; "
            f"expected={expected_dimensions}, actual={actual_dimensions}"
        )
    by_dimension = {row.dimension_id: row for row in draft.dimensions}
    compiled = [
        _compile_dimension(
            packet=packet,
            comparison=comparison,
            draft=by_dimension[dimension_id],
        )
        for dimension_id in expected_dimensions
    ]
    return BlindComparisonEvaluation(
        comparison_alias=comparison.comparison_alias,
        arm_a_candidate_count=comparison.arm_a.candidate_count,
        arm_b_candidate_count=comparison.arm_b.candidate_count,
        absolute_candidate_count_difference=comparison.absolute_candidate_count_difference,
        dimensions=compiled,
        candidate_count_treated_as_quality_signal=draft.candidate_count_treated_as_quality_signal,
        source_condition_inference_attempted=draft.source_condition_inference_attempted,
        source_lane_or_operator_inference_attempted=(
            draft.source_lane_or_operator_inference_attempted
        ),
    )


def run_blind_scientific_reasoning_evaluation(
    *,
    packet: ScientificReasoningAblationPacket,
    backend: AblationBlindEvaluationBackend,
) -> BlindScientificReasoningEvaluationReport:
    evaluations: list[BlindComparisonEvaluation] = []
    calls: list[BlindJudgeCallRecord] = []
    for comparison in packet.comparisons:
        prompt = build_ablation_blind_evaluation_prompt(
            packet=packet,
            comparison=comparison,
        )
        generated = backend.generate(prompt)
        evaluation = compile_blind_comparison_evaluation(
            packet=packet,
            comparison=comparison,
            draft=generated.draft,
        )
        evaluations.append(evaluation)
        calls.append(
            BlindJudgeCallRecord(
                comparison_alias=comparison.comparison_alias,
                response_id=generated.response_id,
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
                elapsed_seconds=generated.elapsed_seconds,
            )
        )

    report_id = _stable_id(
        "scientific_reasoning_ablation_blind_evaluation",
        packet.packet_id,
        backend.backend_name,
        backend.model_name,
        *[
            row.judgment_id
            for evaluation in evaluations
            for row in evaluation.dimensions
        ],
    )
    return BlindScientificReasoningEvaluationReport(
        report_id=report_id,
        source_packet_id=packet.packet_id,
        source_packet_sha256=_sha256(packet.model_dump(mode="json")),
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        comparison_evaluations=evaluations,
        call_records=calls,
        llm_calls_performed=len(calls),
    )


def _arm_condition(key_comparison, arm_alias: str) -> AblationCondition:
    matches = [row.condition for row in key_comparison.arms if row.arm_alias == arm_alias]
    if len(matches) != 1:
        raise ValueError(
            f"blind key must map {arm_alias} exactly once for {key_comparison.comparison_alias}"
        )
    return matches[0]


def _candidate_source_id(key_comparison, arm_alias: str, candidate_alias: str) -> str:
    matches = [row for row in key_comparison.arms if row.arm_alias == arm_alias]
    if len(matches) != 1:
        raise ValueError("blind key arm mapping is ambiguous")
    mapping = matches[0].candidate_alias_to_source_id
    if candidate_alias not in mapping:
        raise ValueError(
            "blind key lacks candidate alias from evaluation: "
            f"{arm_alias}/{candidate_alias}"
        )
    return mapping[candidate_alias]


def _preferred_condition(
    preference: BlindPreference,
    *,
    arm_a_condition: AblationCondition,
    arm_b_condition: AblationCondition,
) -> UnblindedPreference:
    if preference == "ARM_A":
        return arm_a_condition
    if preference == "ARM_B":
        return arm_b_condition
    return preference


def unblind_scientific_reasoning_evaluation(
    *,
    packet: ScientificReasoningAblationPacket,
    key: ScientificReasoningAblationBlindKey,
    blind_report: BlindScientificReasoningEvaluationReport,
    source_blind_report_sha256: str,
) -> UnblindedScientificReasoningEvaluationReport:
    packet_sha = _sha256(packet.model_dump(mode="json"))
    if blind_report.source_packet_id != packet.packet_id:
        raise ValueError("blind report source_packet_id mismatch")
    if blind_report.source_packet_sha256 != packet_sha:
        raise ValueError("blind report source_packet_sha256 mismatch")
    if key.packet_id != packet.packet_id:
        raise ValueError("blind key packet_id mismatch")

    key_by_alias = {row.comparison_alias: row for row in key.comparisons}
    report_aliases = [row.comparison_alias for row in blind_report.comparison_evaluations]
    packet_aliases = [row.comparison_alias for row in packet.comparisons]
    if report_aliases != packet_aliases:
        raise ValueError("blind report comparison ordering does not match packet")
    if set(key_by_alias) != set(packet_aliases):
        raise ValueError("blind key comparisons do not match packet")

    statement_map = key.evidence_alias_to_statement_id
    output: list[UnblindedComparisonEvaluation] = []
    for evaluation in blind_report.comparison_evaluations:
        key_row = key_by_alias[evaluation.comparison_alias]
        arm_a_condition = _arm_condition(key_row, "ARM_A")
        arm_b_condition = _arm_condition(key_row, "ARM_B")
        dimensions: list[UnblindedDimensionJudgment] = []
        for row in evaluation.dimensions:
            unknown_evidence = sorted(
                set(row.supporting_evidence_aliases) - set(statement_map)
            )
            if unknown_evidence:
                raise ValueError(
                    "blind report contains evidence aliases absent from key: "
                    + ", ".join(unknown_evidence)
                )
            candidate_refs = [
                UnblindedCandidateReference(
                    arm_alias=ref.arm_alias,
                    condition=_arm_condition(key_row, ref.arm_alias),
                    candidate_alias=ref.candidate_alias,
                    source_candidate_id=_candidate_source_id(
                        key_row,
                        ref.arm_alias,
                        ref.candidate_alias,
                    ),
                )
                for ref in row.supporting_candidate_refs
            ]
            dimensions.append(
                UnblindedDimensionJudgment(
                    dimension_id=row.dimension_id,
                    blind_preference=row.preference,
                    preferred_condition=_preferred_condition(
                        row.preference,
                        arm_a_condition=arm_a_condition,
                        arm_b_condition=arm_b_condition,
                    ),
                    rationale=row.rationale,
                    supporting_statement_ids=[
                        statement_map[alias]
                        for alias in row.supporting_evidence_aliases
                    ],
                    supporting_candidate_refs=candidate_refs,
                    uncertainty_notes=list(row.uncertainty_notes),
                    candidate_count_influenced_judgment=(
                        row.candidate_count_influenced_judgment
                    ),
                )
            )
        counts = Counter(row.preferred_condition for row in dimensions)
        output.append(
            UnblindedComparisonEvaluation(
                comparison_alias=evaluation.comparison_alias,
                comparison_kind=key_row.comparison_kind,
                arm_a_condition=arm_a_condition,
                arm_b_condition=arm_b_condition,
                dimensions=dimensions,
                preference_counts=dict(sorted(counts.items())),
            )
        )

    report_id = _stable_id(
        "scientific_reasoning_ablation_unblinded_evaluation",
        blind_report.report_id,
        key.source_task_id,
        *[
            f"{row.comparison_alias}:{dim.dimension_id}:{dim.preferred_condition}"
            for row in output
            for dim in row.dimensions
        ],
    )
    return UnblindedScientificReasoningEvaluationReport(
        report_id=report_id,
        source_blind_report_id=blind_report.report_id,
        source_blind_report_sha256=source_blind_report_sha256,
        source_packet_id=packet.packet_id,
        source_packet_sha256=packet_sha,
        source_task_id=key.source_task_id,
        source_context_id=key.source_context_id,
        source_context_sha256=key.source_context_sha256,
        comparisons=output,
    )
