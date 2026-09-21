from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionCandidateDiscriminatingTest,
    ProductionCandidateFalsifier,
    ProductionFacingScientificCandidatePortfolio,
    ProductionScientificCandidate,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SynthesisKind = Literal[
    "complementary_mechanism_integration",
    "conditional_relation_refinement",
    "competing_model_formulation",
    "measurement_model_integration",
    "cross_lane_explanatory_synthesis",
]


class SynthesisPredictionDraft(StrictModel):
    observable: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)


class SynthesisFalsifierDraft(StrictModel):
    observable: str | None = None
    falsifying_outcome: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)


class SynthesisDiscriminatingTestDraft(StrictModel):
    test_design: str = Field(min_length=1)
    primary_observables: list[str] = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)
    favoring_outcomes: list[str] = Field(min_length=2)


class CrossLaneSynthesizedHypothesisDraft(StrictModel):
    local_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=2)
    synthesis_kind: SynthesisKind
    hypothesis_statement: str = Field(min_length=1)
    synthesis_rationale: str = Field(min_length=1)
    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    predictions: list[SynthesisPredictionDraft] = Field(min_length=1)
    falsifiers: list[SynthesisFalsifierDraft] = Field(min_length=1)
    discriminating_test: SynthesisDiscriminatingTestDraft
    unresolved_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_refs(self) -> "CrossLaneSynthesizedHypothesisDraft":
        if len(self.source_candidate_ids) != len(set(self.source_candidate_ids)):
            raise ValueError("source_candidate_ids must be unique")
        return self


class CrossLaneSynthesisBatchDraft(StrictModel):
    schema_version: Literal[
        "cross-lane-scientific-synthesis-batch-draft-v1"
    ] = "cross-lane-scientific-synthesis-batch-draft-v1"
    hypotheses: list[CrossLaneSynthesizedHypothesisDraft] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_abstention(self) -> "CrossLaneSynthesisBatchDraft":
        if not self.hypotheses and not self.abstention_reason:
            raise ValueError("empty synthesis batch requires abstention_reason")
        if self.hypotheses and self.abstention_reason:
            raise ValueError("non-empty synthesis batch must not include abstention_reason")
        return self


class SynthesizedPrediction(StrictModel):
    observable: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)


class SynthesizedFalsifier(StrictModel):
    observable: str | None = None
    falsifying_outcome: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)


class SynthesizedDiscriminatingTest(StrictModel):
    test_design: str = Field(min_length=1)
    primary_observables: list[str] = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)
    favoring_outcomes: list[str] = Field(min_length=2)


class CrossLaneSynthesizedHypothesis(StrictModel):
    schema_version: Literal[
        "cross-lane-synthesized-scientific-hypothesis-v1"
    ] = "cross-lane-synthesized-scientific-hypothesis-v1"

    hypothesis_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_candidate_portfolio_id: str = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=2)
    source_object_ids: list[str] = Field(min_length=2)
    source_lanes: list[Literal["RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"]] = Field(
        min_length=2
    )
    synthesis_kind: SynthesisKind

    title: str = Field(min_length=1)
    hypothesis_statement: str = Field(min_length=1)
    synthesis_rationale: str = Field(min_length=1)
    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    predictions: list[SynthesizedPrediction] = Field(min_length=1)
    falsifiers: list[SynthesizedFalsifier] = Field(min_length=1)
    discriminating_test: SynthesizedDiscriminatingTest
    unresolved_questions: list[str] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    cross_lane_integration_verified: Literal[True] = True
    grounded_statement_ids_only: Literal[True] = True
    new_evidence_asserted: Literal[False] = False
    novelty_assessed: Literal[False] = False
    source_candidates_ranked: Literal[False] = False
    source_candidates_pruned: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_lanes(self) -> "CrossLaneSynthesizedHypothesis":
        if set(self.source_lanes) != {"RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"}:
            raise ValueError("synthesized hypothesis must integrate both source lanes")
        return self


class CrossLaneSynthesisShadowReport(StrictModel):
    schema_version: Literal[
        "cross-lane-scientific-synthesis-shadow-report-v1"
    ] = "cross-lane-scientific-synthesis-shadow-report-v1"

    report_id: str = Field(min_length=1)
    source_candidate_portfolio_id: str = Field(min_length=1)
    source_candidate_portfolio_sha256: str = Field(min_length=64, max_length=64)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    backend_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    source_candidate_ids: list[str]
    hypotheses: list[CrossLaneSynthesizedHypothesis]
    synthesized_hypothesis_ids: list[str]
    abstention_reason: str | None = None
    source_candidate_count: int = Field(ge=0)
    synthesized_hypothesis_count: int = Field(ge=0)
    llm_calls_performed: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    shadow_only: Literal[True] = True
    source_candidates_preserved: Literal[True] = True
    source_candidates_mutated: Literal[False] = False
    cross_lane_synthesis_performed: bool
    scientific_quality_ranking_performed: Literal[False] = False
    source_candidate_redundancy_pruning_performed: Literal[False] = False
    novelty_assessment_performed: Literal[False] = False
    final_hypothesis_selection_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "CrossLaneSynthesisShadowReport":
        if self.source_candidate_count != len(self.source_candidate_ids):
            raise ValueError("source_candidate_count mismatch")
        if self.synthesized_hypothesis_count != len(self.hypotheses):
            raise ValueError("synthesized_hypothesis_count mismatch")
        if self.synthesized_hypothesis_ids != [row.hypothesis_id for row in self.hypotheses]:
            raise ValueError("synthesized_hypothesis_ids mismatch")
        if self.cross_lane_synthesis_performed != bool(self.hypotheses):
            raise ValueError("cross_lane_synthesis_performed mismatch")
        if self.hypotheses and self.abstention_reason is not None:
            raise ValueError("successful synthesis must not carry abstention_reason")
        if not self.hypotheses and not self.abstention_reason:
            raise ValueError("empty synthesis report requires abstention_reason")
        return self


@dataclass(frozen=True)
class CrossLaneSynthesisPrompt:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class CrossLaneSynthesisGeneration:
    draft: CrossLaneSynthesisBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class CrossLaneSynthesisBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(self, prompt: CrossLaneSynthesisPrompt) -> CrossLaneSynthesisGeneration: ...


class InstructorOpenAICompatibleCrossLaneSynthesisBackend:
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
                "Cross-lane scientific synthesis requires installed 'openai' and 'instructor'."
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

    def generate(self, prompt: CrossLaneSynthesisPrompt) -> CrossLaneSynthesisGeneration:
        client = self._get_client()
        draft, event = run_instructor_structured_call(
            client.chat.completions,
            model=self.model_name,
            response_model=CrossLaneSynthesisBatchDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "cross_lane_scientific_synthesis_shadow",
                "stage": "cross_lane_synthesis",
                "call_kind": "scientific_hypothesis_synthesis",
            },
            semantic_components={"synthesis_contract": "cross_lane_v1"},
        )
        if not isinstance(draft, CrossLaneSynthesisBatchDraft):
            draft = CrossLaneSynthesisBatchDraft.model_validate(draft)
        return CrossLaneSynthesisGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=event.elapsed_seconds,
        )


def _sha256_json(value: object) -> str:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:20]}"


def _candidate_payload(
    row: ProductionScientificCandidate,
    *,
    candidate_ref: str,
) -> dict[str, object]:
    return {
        "candidate_ref": candidate_ref,
        "source_lane": row.source_lane,
        "source_object_kind": row.source_object_kind,
        "reasoning_label": row.reasoning_label,
        "title": row.title,
        "scientific_proposal": row.scientific_proposal,
        "reasoning_rationale": row.reasoning_rationale,
        "premise_statement_ids": row.premise_statement_ids,
        "gap_statement_ids": row.gap_statement_ids,
        "assumptions": row.assumptions,
        "predictions": [p.model_dump(mode="json") for p in row.predictions],
        "falsifiers": [f.model_dump(mode="json") for f in row.falsifiers],
        "discriminating_test": (
            row.discriminating_test.model_dump(mode="json")
            if row.discriminating_test is not None
            else None
        ),
        "unresolved_questions": row.unresolved_questions,
    }


def build_cross_lane_synthesis_prompt(
    portfolio: ProductionFacingScientificCandidatePortfolio,
    *,
    max_syntheses: int,
) -> CrossLaneSynthesisPrompt:
    if max_syntheses < 1:
        raise ValueError("max_syntheses must be at least 1")
    system_prompt = """You are performing source-bounded scientific hypothesis synthesis across two reasoning lanes.

Your task is NOT to rank, score, select, or discard the supplied source candidates. Every source candidate remains preserved independently.

Create a synthesized hypothesis only when combining at least one RELATIONAL_DISCOVERY candidate and at least one SCIENTIFIC_REFRAMING candidate produces a testably different scientific proposition that is not a superficial concatenation of either source alone.

Hard constraints:
- Use only the supplied candidates and their statement IDs. Do not invent evidence, papers, measurements, or statement IDs.
- Every synthesized hypothesis must cite at least two source_candidate_ids and must include candidates from BOTH source lanes.
- In every source_candidate_ids field, copy ONLY the supplied candidate_ref aliases such as CANDIDATE_01. Never reconstruct, shorten, extend, or invent canonical production_candidate IDs.
- premise_statement_ids and gap_statement_ids must come from the cited source candidates only.
- Each prediction/falsifier/test must cite only source_candidate_ids used by that synthesized hypothesis.
- Do not claim literature novelty, truth, superiority, or empirical confirmation.
- Preserve uncertainty and assumptions. If the candidates do not justify genuine cross-lane synthesis, abstain.
- Prefer a mechanistic, conditional, measurement-aware, or competing-model integration that creates a discriminating experiment.
- Do not merely rewrite one source candidate with more words.
"""
    user_payload = {
        "question": portfolio.question,
        "max_syntheses": max_syntheses,
        "source_candidates": [
            _candidate_payload(row, candidate_ref=f"CANDIDATE_{index:02d}")
            for index, row in enumerate(portfolio.candidates, start=1)
        ],
    }
    user_prompt = (
        "Construct up to max_syntheses grounded cross-lane synthesized hypotheses from this payload. "
        "Return an abstention_reason instead if no genuine synthesis is justified.\n\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )
    return CrossLaneSynthesisPrompt(system_prompt=system_prompt, user_prompt=user_prompt)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _candidate_map(
    portfolio: ProductionFacingScientificCandidatePortfolio,
) -> dict[str, ProductionScientificCandidate]:
    return {row.candidate_id: row for row in portfolio.candidates}


def _candidate_ref_map(
    portfolio: ProductionFacingScientificCandidatePortfolio,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for index, row in enumerate(portfolio.candidates, start=1):
        refs[f"CANDIDATE_{index:02d}"] = row.candidate_id
        # Backward compatibility for deterministic tests and historical
        # programmatic callers that already provide canonical IDs.
        refs[row.candidate_id] = row.candidate_id
    return refs


def _resolve_candidate_refs(
    *,
    source_ids: list[str],
    ref_map: dict[str, str],
    label: str,
) -> list[str]:
    unknown = sorted(set(source_ids) - set(ref_map))
    if unknown:
        raise ValueError(f"{label} contains unknown source candidate IDs: {unknown}")
    return [ref_map[source_id] for source_id in source_ids]


def _validate_refs(
    *,
    source_ids: list[str],
    allowed_ids: set[str],
    label: str,
) -> None:
    unknown = sorted(set(source_ids) - allowed_ids)
    if unknown:
        raise ValueError(f"{label} contains unknown source candidate IDs: {unknown}")


def _compile_hypothesis(
    *,
    draft: CrossLaneSynthesizedHypothesisDraft,
    portfolio: ProductionFacingScientificCandidatePortfolio,
    candidate_map: dict[str, ProductionScientificCandidate],
) -> CrossLaneSynthesizedHypothesis:
    ref_map = _candidate_ref_map(portfolio)
    source_candidate_ids = _resolve_candidate_refs(
        source_ids=draft.source_candidate_ids,
        ref_map=ref_map,
        label="synthesis",
    )
    selected = [candidate_map[candidate_id] for candidate_id in source_candidate_ids]
    lanes = sorted({row.source_lane for row in selected})
    if set(lanes) != {"RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"}:
        raise ValueError("synthesis must cite at least one candidate from each source lane")

    available_premises = {sid for row in selected for sid in row.premise_statement_ids}
    available_gaps = {sid for row in selected for sid in row.gap_statement_ids}
    unknown_premises = sorted(set(draft.premise_statement_ids) - available_premises)
    unknown_gaps = sorted(set(draft.gap_statement_ids) - available_gaps)
    if unknown_premises:
        raise ValueError(f"synthesis cites unsupported premise statement IDs: {unknown_premises}")
    if unknown_gaps:
        raise ValueError(f"synthesis cites unsupported gap statement IDs: {unknown_gaps}")

    selected_ids = set(source_candidate_ids)
    prediction_source_ids: list[list[str]] = []
    for index, prediction in enumerate(draft.predictions):
        resolved = _resolve_candidate_refs(
            source_ids=prediction.source_candidate_ids,
            ref_map=ref_map,
            label=f"prediction[{index}]",
        )
        _validate_refs(
            source_ids=resolved,
            allowed_ids=selected_ids,
            label=f"prediction[{index}]",
        )
        prediction_source_ids.append(resolved)
    falsifier_source_ids: list[list[str]] = []
    for index, falsifier in enumerate(draft.falsifiers):
        resolved = _resolve_candidate_refs(
            source_ids=falsifier.source_candidate_ids,
            ref_map=ref_map,
            label=f"falsifier[{index}]",
        )
        _validate_refs(
            source_ids=resolved,
            allowed_ids=selected_ids,
            label=f"falsifier[{index}]",
        )
        falsifier_source_ids.append(resolved)
    discriminating_test_source_ids = _resolve_candidate_refs(
        source_ids=draft.discriminating_test.source_candidate_ids,
        ref_map=ref_map,
        label="discriminating_test",
    )
    _validate_refs(
        source_ids=discriminating_test_source_ids,
        allowed_ids=selected_ids,
        label="discriminating_test",
    )

    proposal_norm = _normalize(draft.hypothesis_statement)
    if proposal_norm in {_normalize(row.scientific_proposal) for row in selected}:
        raise ValueError("synthesized hypothesis must differ from each source proposal")

    source_object_ids = sorted({row.source_object_id for row in selected})
    hypothesis_id = _stable_id(
        "cross_lane_synthesized_hypothesis",
        portfolio.portfolio_id,
        *sorted(source_candidate_ids),
        draft.synthesis_kind,
        draft.title,
        draft.hypothesis_statement,
    )
    return CrossLaneSynthesizedHypothesis(
        hypothesis_id=hypothesis_id,
        source_task_id=portfolio.source_task_id,
        source_context_id=portfolio.source_context_id,
        source_candidate_portfolio_id=portfolio.portfolio_id,
        source_candidate_ids=list(source_candidate_ids),
        source_object_ids=source_object_ids,
        source_lanes=lanes,  # type: ignore[arg-type]
        synthesis_kind=draft.synthesis_kind,
        title=draft.title,
        hypothesis_statement=draft.hypothesis_statement,
        synthesis_rationale=draft.synthesis_rationale,
        premise_statement_ids=list(dict.fromkeys(draft.premise_statement_ids)),
        gap_statement_ids=list(dict.fromkeys(draft.gap_statement_ids)),
        assumptions=list(dict.fromkeys(draft.assumptions)),
        predictions=[
            SynthesizedPrediction(
                observable=row.observable,
                expected_result=row.expected_result,
                rationale=row.rationale,
                source_candidate_ids=list(prediction_source_ids[index]),
            )
            for index, row in enumerate(draft.predictions)
        ],
        falsifiers=[
            SynthesizedFalsifier(
                observable=row.observable,
                falsifying_outcome=row.falsifying_outcome,
                source_candidate_ids=list(falsifier_source_ids[index]),
            )
            for index, row in enumerate(draft.falsifiers)
        ],
        discriminating_test=SynthesizedDiscriminatingTest(
            test_design=draft.discriminating_test.test_design,
            primary_observables=list(draft.discriminating_test.primary_observables),
            source_candidate_ids=list(discriminating_test_source_ids),
            favoring_outcomes=list(draft.discriminating_test.favoring_outcomes),
        ),
        unresolved_questions=list(dict.fromkeys(draft.unresolved_questions)),
    )


def compile_cross_lane_synthesis(
    *,
    portfolio: ProductionFacingScientificCandidatePortfolio,
    draft: CrossLaneSynthesisBatchDraft,
    backend_name: str,
    model_name: str,
    max_syntheses: int,
    llm_calls_performed: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> CrossLaneSynthesisShadowReport:
    if len(draft.hypotheses) > max_syntheses:
        raise ValueError(
            f"synthesis returned {len(draft.hypotheses)} hypotheses, max_syntheses={max_syntheses}"
        )
    candidate_map = _candidate_map(portfolio)
    compiled = [
        _compile_hypothesis(draft=row, portfolio=portfolio, candidate_map=candidate_map)
        for row in draft.hypotheses
    ]
    source_sets = [tuple(sorted(row.source_candidate_ids)) for row in compiled]
    if len(source_sets) != len(set(source_sets)):
        raise ValueError("multiple synthesized hypotheses may not reuse the exact same source set")

    report_id = _stable_id(
        "cross_lane_scientific_synthesis_shadow",
        portfolio.portfolio_id,
        backend_name,
        model_name,
        *(row.hypothesis_id for row in compiled),
        draft.abstention_reason or "",
    )
    return CrossLaneSynthesisShadowReport(
        report_id=report_id,
        source_candidate_portfolio_id=portfolio.portfolio_id,
        source_candidate_portfolio_sha256=_sha256_json(portfolio),
        source_task_id=portfolio.source_task_id,
        source_context_id=portfolio.source_context_id,
        question=portfolio.question,
        backend_name=backend_name,
        model_name=model_name,
        source_candidate_ids=[row.candidate_id for row in portfolio.candidates],
        hypotheses=compiled,
        synthesized_hypothesis_ids=[row.hypothesis_id for row in compiled],
        abstention_reason=draft.abstention_reason,
        source_candidate_count=portfolio.candidate_count,
        synthesized_hypothesis_count=len(compiled),
        llm_calls_performed=llm_calls_performed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cross_lane_synthesis_performed=bool(compiled),
    )


class CrossLaneSynthesisRuntime:
    def __init__(self, backend: CrossLaneSynthesisBackend):
        self.backend = backend

    def run(
        self,
        *,
        portfolio: ProductionFacingScientificCandidatePortfolio,
        max_syntheses: int = 3,
    ) -> tuple[CrossLaneSynthesisShadowReport, CrossLaneSynthesisPrompt | None]:
        lanes = {row.source_lane for row in portfolio.candidates}
        if lanes != {"RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"}:
            draft = CrossLaneSynthesisBatchDraft(
                hypotheses=[],
                abstention_reason=(
                    "Cross-lane synthesis requires at least one candidate from each source lane."
                ),
            )
            report = compile_cross_lane_synthesis(
                portfolio=portfolio,
                draft=draft,
                backend_name=self.backend.backend_name,
                model_name=self.backend.model_name,
                max_syntheses=max_syntheses,
                llm_calls_performed=0,
            )
            return report, None

        prompt = build_cross_lane_synthesis_prompt(portfolio, max_syntheses=max_syntheses)
        generation = self.backend.generate(prompt)
        report = compile_cross_lane_synthesis(
            portfolio=portfolio,
            draft=generation.draft,
            backend_name=self.backend.backend_name,
            model_name=self.backend.model_name,
            max_syntheses=max_syntheses,
            llm_calls_performed=1,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
        )
        return report, prompt


__all__ = [
    "CrossLaneSynthesizedHypothesisDraft",
    "CrossLaneSynthesisBatchDraft",
    "CrossLaneSynthesizedHypothesis",
    "CrossLaneSynthesisShadowReport",
    "CrossLaneSynthesisPrompt",
    "CrossLaneSynthesisGeneration",
    "CrossLaneSynthesisBackend",
    "InstructorOpenAICompatibleCrossLaneSynthesisBackend",
    "CrossLaneSynthesisRuntime",
    "build_cross_lane_synthesis_prompt",
    "compile_cross_lane_synthesis",
]
