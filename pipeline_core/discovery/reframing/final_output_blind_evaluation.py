from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.integrated_hypothesis_shadow import (
    IntegratedScientificHypothesisShadowPortfolio,
    IntegratedShadowHypothesis,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FinalOutputCondition = Literal["LEGACY_FINAL_VIEW", "INTEGRATED_FINAL_VIEW"]
ArmAlias = Literal["ARM_A", "ARM_B"]
DimensionPreference = Literal["ARM_A", "ARM_B", "TIE", "UNCLEAR"]


class BlindFinalPrediction(StrictModel):
    observable: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class BlindFinalFalsifier(StrictModel):
    observable: str | None = None
    falsifying_outcome: str = Field(min_length=1)


class BlindFinalDiscriminatingTest(StrictModel):
    test_design: str = Field(min_length=1)
    primary_observables: list[str] = Field(min_length=1)
    favoring_outcomes: list[str] = Field(min_length=1)


class BlindFinalHypothesis(StrictModel):
    hypothesis_alias: str = Field(min_length=1)
    title: str = Field(min_length=1)
    hypothesis_statement: str = Field(min_length=1)
    reasoning_rationale: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    predictions: list[BlindFinalPrediction] = Field(default_factory=list)
    falsifiers: list[BlindFinalFalsifier] = Field(default_factory=list)
    discriminating_test: BlindFinalDiscriminatingTest | None = None
    unresolved_questions: list[str] = Field(default_factory=list)


class BlindFinalPortfolioArm(StrictModel):
    arm_alias: ArmAlias
    hypotheses: list[BlindFinalHypothesis] = Field(min_length=1)
    hypothesis_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_count(self) -> "BlindFinalPortfolioArm":
        if self.hypothesis_count != len(self.hypotheses):
            raise ValueError("hypothesis_count mismatch")
        aliases = [row.hypothesis_alias for row in self.hypotheses]
        if len(aliases) != len(set(aliases)):
            raise ValueError("hypothesis aliases must be unique within an arm")
        return self


class FinalOutputEvaluationDimension(StrictModel):
    dimension_id: Literal[
        "task_relevance_and_coverage",
        "mechanistic_explanatory_gain",
        "prediction_specificity_and_differentiation",
        "falsifiability_and_discriminating_tests",
        "internal_coherence",
        "focus_and_redundancy_control",
        "practical_research_value",
    ]
    question: str = Field(min_length=1)


class BlindFinalOutputEvaluationPacket(StrictModel):
    schema_version: Literal[
        "blind-final-hypothesis-output-evaluation-packet-v1"
    ] = "blind-final-hypothesis-output-evaluation-packet-v1"

    packet_id: str = Field(min_length=1)
    source_shadow_portfolio_sha256: str = Field(min_length=64, max_length=64)
    task_alias: str = Field(min_length=1)
    question: str = Field(min_length=1)
    arm_a: BlindFinalPortfolioArm
    arm_b: BlindFinalPortfolioArm
    evaluation_dimensions: list[FinalOutputEvaluationDimension] = Field(min_length=7, max_length=7)
    interpretation_cautions: list[str] = Field(min_length=1)

    condition_labels_hidden: Literal[True] = True
    source_ids_hidden: Literal[True] = True
    hypothesis_count_may_differ: Literal[True] = True
    candidate_count_treated_as_quality_signal: Literal[False] = False
    overall_score_requested: Literal[False] = False
    overall_winner_requested: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_packet(self) -> "BlindFinalOutputEvaluationPacket":
        expected = {
            "task_relevance_and_coverage",
            "mechanistic_explanatory_gain",
            "prediction_specificity_and_differentiation",
            "falsifiability_and_discriminating_tests",
            "internal_coherence",
            "focus_and_redundancy_control",
            "practical_research_value",
        }
        observed = {row.dimension_id for row in self.evaluation_dimensions}
        if observed != expected:
            raise ValueError("final-output evaluation dimensions are incomplete or duplicated")
        payload = json.dumps(self.model_dump(mode="json"), ensure_ascii=False).lower()
        forbidden = (
            "legacy_final_view",
            "integrated_final_view",
            "legacy relational",
            "cross_lane_synthesis",
            "cross-lane synthesis",
            "scientific_reframing",
        )
        for token in forbidden:
            if token in payload:
                raise ValueError(f"blind final-output packet leaked source condition token: {token}")
        return self


class FinalOutputArmKey(StrictModel):
    arm_alias: ArmAlias
    condition: FinalOutputCondition
    hypothesis_alias_to_entry_id: dict[str, str] = Field(min_length=1)


class FinalOutputBlindKey(StrictModel):
    schema_version: Literal[
        "blind-final-hypothesis-output-key-v1"
    ] = "blind-final-hypothesis-output-key-v1"

    packet_id: str = Field(min_length=1)
    source_shadow_portfolio_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    arms: list[FinalOutputArmKey] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_key(self) -> "FinalOutputBlindKey":
        if {row.arm_alias for row in self.arms} != {"ARM_A", "ARM_B"}:
            raise ValueError("blind key must map ARM_A and ARM_B exactly once")
        if {row.condition for row in self.arms} != {
            "LEGACY_FINAL_VIEW",
            "INTEGRATED_FINAL_VIEW",
        }:
            raise ValueError("blind key must map both final-output conditions exactly once")
        return self


class FinalOutputDimensionJudgment(StrictModel):
    dimension_id: str = Field(min_length=1)
    preference: DimensionPreference
    rationale: str = Field(min_length=1)


class FinalOutputEvaluationDraft(StrictModel):
    judgments: list[FinalOutputDimensionJudgment] = Field(min_length=7, max_length=7)


class BlindFinalOutputEvaluationReport(StrictModel):
    schema_version: Literal[
        "blind-final-hypothesis-output-evaluation-report-v1"
    ] = "blind-final-hypothesis-output-evaluation-report-v1"

    report_id: str = Field(min_length=1)
    packet_id: str = Field(min_length=1)
    backend_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    judgments: list[FinalOutputDimensionJudgment] = Field(min_length=7, max_length=7)
    preference_counts: dict[str, int]
    llm_calls_performed: Literal[1] = 1
    input_tokens: int | None = None
    output_tokens: int | None = None

    blind_key_loaded_by_evaluator: Literal[False] = False
    candidate_count_treated_as_quality_signal: Literal[False] = False
    overall_score_computed: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    overall_scientific_quality_ranking_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "BlindFinalOutputEvaluationReport":
        expected = {
            "task_relevance_and_coverage",
            "mechanistic_explanatory_gain",
            "prediction_specificity_and_differentiation",
            "falsifiability_and_discriminating_tests",
            "internal_coherence",
            "focus_and_redundancy_control",
            "practical_research_value",
        }
        observed = [row.dimension_id for row in self.judgments]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError("evaluation report must judge each dimension exactly once")
        counts: dict[str, int] = {}
        for row in self.judgments:
            counts[row.preference] = counts.get(row.preference, 0) + 1
        if dict(sorted(counts.items())) != dict(sorted(self.preference_counts.items())):
            raise ValueError("preference_counts mismatch")
        return self


class UnblindedFinalOutputDimensionJudgment(StrictModel):
    dimension_id: str = Field(min_length=1)
    observed_preference: DimensionPreference
    preferred_condition: FinalOutputCondition | Literal["TIE", "UNCLEAR"]
    rationale: str = Field(min_length=1)


class UnblindedFinalOutputEvaluationReport(StrictModel):
    schema_version: Literal[
        "unblinded-final-hypothesis-output-evaluation-report-v1"
    ] = "unblinded-final-hypothesis-output-evaluation-report-v1"

    report_id: str = Field(min_length=1)
    packet_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    judgments: list[UnblindedFinalOutputDimensionJudgment] = Field(min_length=7, max_length=7)
    condition_preference_counts: dict[str, int]
    llm_calls_performed: Literal[0] = 0
    per_dimension_results_only: Literal[True] = True
    overall_score_computed: Literal[False] = False
    overall_winner_selected: Literal[False] = False
    overall_scientific_quality_ranking_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False


@dataclass(frozen=True)
class FinalOutputEvaluationPrompt:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class FinalOutputEvaluationGeneration:
    draft: FinalOutputEvaluationDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class FinalOutputEvaluationBackend(Protocol):
    backend_name: str
    model_name: str

    def evaluate(self, prompt: FinalOutputEvaluationPrompt) -> FinalOutputEvaluationGeneration: ...


class InstructorOpenAICompatibleFinalOutputEvaluator:
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
                "Final-output blind evaluation requires installed 'openai' and 'instructor'."
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

    def evaluate(self, prompt: FinalOutputEvaluationPrompt) -> FinalOutputEvaluationGeneration:
        client = self._get_client()
        draft, event = run_instructor_structured_call(
            client.chat.completions,
            model=self.model_name,
            response_model=FinalOutputEvaluationDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "final_hypothesis_output_blind_evaluation",
                "stage": "final_output_evaluation",
                "call_kind": "blind_portfolio_comparison",
            },
            semantic_components={"evaluation_contract": "final_output_v1"},
        )
        if not isinstance(draft, FinalOutputEvaluationDraft):
            draft = FinalOutputEvaluationDraft.model_validate(draft)
        return FinalOutputEvaluationGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )


def _canonical_json(value: object) -> str:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _dimensions() -> list[FinalOutputEvaluationDimension]:
    return [
        FinalOutputEvaluationDimension(
            dimension_id="task_relevance_and_coverage",
            question="Which portfolio better addresses the scientific question while covering the important parts of the problem?",
        ),
        FinalOutputEvaluationDimension(
            dimension_id="mechanistic_explanatory_gain",
            question="Which portfolio provides more useful mechanistic or explanatory structure rather than merely restating associations?",
        ),
        FinalOutputEvaluationDimension(
            dimension_id="prediction_specificity_and_differentiation",
            question="Which portfolio offers more specific predictions that distinguish plausible explanations or conditions?",
        ),
        FinalOutputEvaluationDimension(
            dimension_id="falsifiability_and_discriminating_tests",
            question="Which portfolio is more falsifiable and more clearly supports experiments that discriminate between explanations?",
        ),
        FinalOutputEvaluationDimension(
            dimension_id="internal_coherence",
            question="Which portfolio is more internally coherent, with hypotheses, assumptions, predictions, and falsifiers fitting together without avoidable contradiction?",
        ),
        FinalOutputEvaluationDimension(
            dimension_id="focus_and_redundancy_control",
            question="Which portfolio better balances useful breadth with focus, avoiding redundant or weak additions?",
        ),
        FinalOutputEvaluationDimension(
            dimension_id="practical_research_value",
            question="Which portfolio would be more useful for deciding what scientific study or measurement to pursue next?",
        ),
    ]


def _blind_hypothesis(row: IntegratedShadowHypothesis, alias: str) -> BlindFinalHypothesis:
    test = row.discriminating_test
    return BlindFinalHypothesis(
        hypothesis_alias=alias,
        title=row.title,
        hypothesis_statement=row.hypothesis_statement,
        reasoning_rationale=row.reasoning_rationale,
        assumptions=list(row.assumptions),
        predictions=[
            BlindFinalPrediction(
                observable=item.observable,
                expected_result=item.expected_result,
                rationale=item.rationale,
            )
            for item in row.predictions
        ],
        falsifiers=[
            BlindFinalFalsifier(
                observable=item.observable,
                falsifying_outcome=item.falsifying_outcome,
            )
            for item in row.falsifiers
        ],
        discriminating_test=(
            BlindFinalDiscriminatingTest(
                test_design=test.test_design,
                primary_observables=list(test.primary_observables),
                favoring_outcomes=list(test.favoring_outcomes),
            )
            if test is not None
            else None
        ),
        unresolved_questions=list(row.unresolved_questions),
    )


def _condition_goes_first(portfolio: IntegratedScientificHypothesisShadowPortfolio) -> bool:
    digest = hashlib.sha256(
        f"{portfolio.portfolio_id}|final-output-arm-order-v1".encode("utf-8")
    ).digest()
    return bool(digest[0] & 1)


def build_blind_final_output_evaluation_packet(
    portfolio: IntegratedScientificHypothesisShadowPortfolio,
) -> tuple[BlindFinalOutputEvaluationPacket, FinalOutputBlindKey]:
    if not portfolio.shadow_output_differs_from_legacy:
        raise ValueError("final-output comparison requires integrated shadow to differ from legacy")
    if not portfolio.legacy_hypotheses or not portfolio.integrated_hypotheses:
        raise ValueError("final-output comparison requires both portfolio views")

    condition_rows: dict[FinalOutputCondition, list[IntegratedShadowHypothesis]] = {
        "LEGACY_FINAL_VIEW": list(portfolio.legacy_hypotheses),
        "INTEGRATED_FINAL_VIEW": list(portfolio.integrated_hypotheses),
    }
    legacy_first = _condition_goes_first(portfolio)
    arm_conditions: list[tuple[ArmAlias, FinalOutputCondition]] = (
        [("ARM_A", "LEGACY_FINAL_VIEW"), ("ARM_B", "INTEGRATED_FINAL_VIEW")]
        if legacy_first
        else [("ARM_A", "INTEGRATED_FINAL_VIEW"), ("ARM_B", "LEGACY_FINAL_VIEW")]
    )

    arms: dict[ArmAlias, BlindFinalPortfolioArm] = {}
    key_rows: list[FinalOutputArmKey] = []
    for arm_alias, condition in arm_conditions:
        rows = condition_rows[condition]
        blind_rows: list[BlindFinalHypothesis] = []
        mapping: dict[str, str] = {}
        for index, row in enumerate(rows, start=1):
            alias = f"H{index:02d}"
            blind_rows.append(_blind_hypothesis(row, alias))
            mapping[alias] = row.entry_id
        arms[arm_alias] = BlindFinalPortfolioArm(
            arm_alias=arm_alias,
            hypotheses=blind_rows,
            hypothesis_count=len(blind_rows),
        )
        key_rows.append(
            FinalOutputArmKey(
                arm_alias=arm_alias,
                condition=condition,
                hypothesis_alias_to_entry_id=mapping,
            )
        )

    source_sha = _sha256_json(portfolio)
    packet_id = _stable_id(
        "blind_final_hypothesis_output_packet",
        portfolio.portfolio_id,
        source_sha,
    )
    packet = BlindFinalOutputEvaluationPacket(
        packet_id=packet_id,
        source_shadow_portfolio_sha256=source_sha,
        task_alias="TASK_01",
        question=portfolio.question,
        arm_a=arms["ARM_A"],
        arm_b=arms["ARM_B"],
        evaluation_dimensions=_dimensions(),
        interpretation_cautions=[
            "The two anonymous portfolios may contain different numbers of hypotheses. Do not reward or penalize an arm merely for having more hypotheses.",
            "Judge whether additional hypotheses provide substantive explanatory, predictive, falsifiable, or practical value versus redundancy or loss of focus.",
            "Some hypotheses can be shared across both anonymous portfolios. Evaluate each portfolio as a user-facing whole rather than counting unique items.",
            "Do not infer hidden source conditions, generation methods, or intended experimental treatment from portfolio size or writing style.",
            "Return per-dimension preferences only; do not compute an overall score or select an overall winner.",
        ],
    )
    key = FinalOutputBlindKey(
        packet_id=packet_id,
        source_shadow_portfolio_id=portfolio.portfolio_id,
        source_task_id=portfolio.source_task_id,
        source_context_id=portfolio.source_context_id,
        arms=key_rows,
    )
    return packet, key


def build_final_output_evaluation_prompt(
    packet: BlindFinalOutputEvaluationPacket,
) -> FinalOutputEvaluationPrompt:
    system = (
        "You are a blind scientific-output evaluator. Compare two anonymous hypothesis portfolios for the stated scientific question. "
        "Do not infer or discuss which system, reasoning method, source lane, or experimental condition produced either arm. "
        "Portfolio size is not a quality signal: more hypotheses are useful only if they add substantive value, and they can be harmful if redundant or distracting. "
        "For each required dimension choose exactly one of ARM_A, ARM_B, TIE, or UNCLEAR and give a concise evidence-based rationale. "
        "Do not aggregate dimensions, compute a score, or declare an overall winner."
    )
    payload = {
        "task_alias": packet.task_alias,
        "question": packet.question,
        "interpretation_cautions": packet.interpretation_cautions,
        "evaluation_dimensions": [row.model_dump(mode="json") for row in packet.evaluation_dimensions],
        "ARM_A": packet.arm_a.model_dump(mode="json"),
        "ARM_B": packet.arm_b.model_dump(mode="json"),
        "required_output": {
            "judgments": [
                {
                    "dimension_id": row.dimension_id,
                    "preference": "ARM_A | ARM_B | TIE | UNCLEAR",
                    "rationale": "brief reason grounded in the visible portfolios",
                }
                for row in packet.evaluation_dimensions
            ]
        },
    }
    return FinalOutputEvaluationPrompt(
        system_prompt=system,
        user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _validate_draft(
    packet: BlindFinalOutputEvaluationPacket,
    draft: FinalOutputEvaluationDraft,
) -> list[FinalOutputDimensionJudgment]:
    required = [row.dimension_id for row in packet.evaluation_dimensions]
    observed = [row.dimension_id for row in draft.judgments]
    if len(observed) != len(set(observed)) or set(observed) != set(required):
        raise ValueError("evaluator must return each required dimension exactly once")
    by_id = {row.dimension_id: row for row in draft.judgments}
    return [by_id[dimension_id] for dimension_id in required]


def run_blind_final_output_evaluation(
    *,
    packet: BlindFinalOutputEvaluationPacket,
    backend: FinalOutputEvaluationBackend,
) -> tuple[BlindFinalOutputEvaluationReport, FinalOutputEvaluationPrompt]:
    prompt = build_final_output_evaluation_prompt(packet)
    generation = backend.evaluate(prompt)
    judgments = _validate_draft(packet, generation.draft)
    counts: dict[str, int] = {}
    for row in judgments:
        counts[row.preference] = counts.get(row.preference, 0) + 1
    report = BlindFinalOutputEvaluationReport(
        report_id=_stable_id(
            "blind_final_hypothesis_output_evaluation",
            packet.packet_id,
            backend.backend_name,
            backend.model_name,
            _canonical_json([row.model_dump(mode="json") for row in judgments]),
        ),
        packet_id=packet.packet_id,
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        judgments=judgments,
        preference_counts=dict(sorted(counts.items())),
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
    )
    return report, prompt


def unblind_final_output_evaluation(
    *,
    packet: BlindFinalOutputEvaluationPacket,
    blind_report: BlindFinalOutputEvaluationReport,
    key: FinalOutputBlindKey,
) -> UnblindedFinalOutputEvaluationReport:
    if blind_report.packet_id != packet.packet_id or key.packet_id != packet.packet_id:
        raise ValueError("packet/report/key identity mismatch")
    condition_by_arm = {row.arm_alias: row.condition for row in key.arms}
    rows: list[UnblindedFinalOutputDimensionJudgment] = []
    counts: dict[str, int] = {}
    for judgment in blind_report.judgments:
        if judgment.preference in ("TIE", "UNCLEAR"):
            preferred: FinalOutputCondition | Literal["TIE", "UNCLEAR"] = judgment.preference
        else:
            preferred = condition_by_arm[judgment.preference]
        counts[preferred] = counts.get(preferred, 0) + 1
        rows.append(
            UnblindedFinalOutputDimensionJudgment(
                dimension_id=judgment.dimension_id,
                observed_preference=judgment.preference,
                preferred_condition=preferred,
                rationale=judgment.rationale,
            )
        )
    return UnblindedFinalOutputEvaluationReport(
        report_id=_stable_id(
            "unblinded_final_hypothesis_output_evaluation",
            blind_report.report_id,
            key.source_task_id,
        ),
        packet_id=packet.packet_id,
        source_task_id=key.source_task_id,
        judgments=rows,
        condition_preference_counts=dict(sorted(counts.items())),
    )


__all__ = [
    "BlindFinalOutputEvaluationPacket",
    "FinalOutputBlindKey",
    "BlindFinalOutputEvaluationReport",
    "UnblindedFinalOutputEvaluationReport",
    "FinalOutputEvaluationDraft",
    "FinalOutputDimensionJudgment",
    "FinalOutputEvaluationGeneration",
    "FinalOutputEvaluationPrompt",
    "FinalOutputEvaluationBackend",
    "InstructorOpenAICompatibleFinalOutputEvaluator",
    "build_blind_final_output_evaluation_packet",
    "build_final_output_evaluation_prompt",
    "run_blind_final_output_evaluation",
    "unblind_final_output_evaluation",
]
