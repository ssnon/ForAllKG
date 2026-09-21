from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.evidence_tension import (
    EvidenceLevelTensionWitness,
    ScientificEvidenceTensionReport,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    DifferentialPredictionDraft,
    DiscriminatingTestDraft,
    ReframeFalsifierDraft,
    ScientificModelDraft,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ResolutionKind = Literal[
    "context_partition",
    "scope_separation",
    "competing_mechanism_balance",
    "measurement_conditioning",
    "multi_stage_process",
    "hidden_moderator",
    "other",
]


class ContradictionResolutionSeed(StrictModel):
    witness_id: str = Field(min_length=1)
    source_tension_id: str = Field(min_length=1)
    source_tension_type: str
    tension_types: list[str] = Field(default_factory=list)
    side_a_statement_ids: list[str] = Field(min_length=1)
    side_b_statement_ids: list[str] = Field(min_length=1)
    side_a_texts: list[str] = Field(min_length=1)
    side_b_texts: list[str] = Field(min_length=1)
    focal_statement_id: str | None = None
    focal_statement_text: str | None = None
    paper_ids: list[str] = Field(default_factory=list)
    independence_basis: str
    independent_family_signal: bool
    paired_response_signal: bool
    relevant_condition_names: list[str] = Field(default_factory=list)
    semantic_incompatibility_signals: list[str] = Field(default_factory=list)
    eligibility_bases: list[str] = Field(default_factory=list)
    attempt_eligible: bool = False

    diagnostic_only: Literal[True] = True
    scientific_conflict_authority: Literal[False] = False
    contradiction_authority: Literal[False] = False


class ContradictionResolutionInput(StrictModel):
    schema_version: Literal[
        "contradiction-resolution-input-v1"
    ] = "contradiction-resolution-input-v1"
    task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_tension_report_id: str = Field(min_length=1)
    source_tension_report_sha256: str = Field(min_length=64, max_length=64)
    seeds: list[ContradictionResolutionSeed] = Field(default_factory=list)
    eligible_witness_ids: list[str] = Field(default_factory=list)

    evidence_tension_is_conflict_authority: Literal[False] = False
    contradiction_established: Literal[False] = False
    missing_eligible_tension_is_negative_evidence: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_seed_partition(self) -> "ContradictionResolutionInput":
        ids = [row.witness_id for row in self.seeds]
        if len(ids) != len(set(ids)):
            raise ValueError("contradiction-resolution witness IDs must be unique")
        expected = [row.witness_id for row in self.seeds if row.attempt_eligible]
        if self.eligible_witness_ids != expected:
            raise ValueError(
                "eligible_witness_ids must equal eligible seeds in stable order"
            )
        return self


class ContradictionResolutionDraft(StrictModel):
    local_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    tension_witness_ids: list[str] = Field(min_length=1)
    premise_statement_ids: list[str] = Field(min_length=2)
    gap_statement_ids: list[str] = Field(default_factory=list)
    side_a_statement_ids: list[str] = Field(min_length=1)
    side_b_statement_ids: list[str] = Field(min_length=1)
    baseline_model: ScientificModelDraft
    resolution_model: ScientificModelDraft
    apparent_contradiction: str = Field(min_length=1)
    resolution_principle: str = Field(min_length=1)
    resolution_kind: ResolutionKind
    distinguishing_context_variables: list[str] = Field(default_factory=list)
    proposed_resolution_constructs: list[str] = Field(default_factory=list)
    differential_predictions: list[DifferentialPredictionDraft] = Field(min_length=1)
    falsifiers: list[ReframeFalsifierDraft] = Field(min_length=1)
    discriminating_test: DiscriminatingTestDraft
    unresolved_questions: list[str] = Field(default_factory=list)


class ContradictionResolutionBatchDraft(StrictModel):
    schema_version: Literal[
        "contradiction-resolution-batch-draft-v1"
    ] = "contradiction-resolution-batch-draft-v1"
    candidates: list[ContradictionResolutionDraft] = Field(default_factory=list)
    abstention_reason: str | None = None


class ScientificContradictionResolutionCandidate(StrictModel):
    schema_version: Literal[
        "scientific-contradiction-resolution-candidate-v1"
    ] = "scientific-contradiction-resolution-candidate-v1"
    candidate_id: str = Field(min_length=1)
    operator_id: Literal["CONTRADICTION_RESOLUTION"] = "CONTRADICTION_RESOLUTION"
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    source_tension_report_id: str = Field(min_length=1)
    source_tension_report_sha256: str = Field(min_length=64, max_length=64)

    title: str = Field(min_length=1)
    tension_witness_ids: list[str] = Field(min_length=1)
    premise_statement_ids: list[str] = Field(min_length=2)
    gap_statement_ids: list[str] = Field(default_factory=list)
    side_a_statement_ids: list[str] = Field(min_length=1)
    side_b_statement_ids: list[str] = Field(min_length=1)
    baseline_model: ScientificModelDraft
    resolution_model: ScientificModelDraft
    apparent_contradiction: str = Field(min_length=1)
    resolution_principle: str = Field(min_length=1)
    resolution_kind: ResolutionKind
    distinguishing_context_variables: list[str] = Field(default_factory=list)
    proposed_resolution_constructs: list[str] = Field(default_factory=list)
    differential_predictions: list[DifferentialPredictionDraft] = Field(min_length=1)
    falsifiers: list[ReframeFalsifierDraft] = Field(min_length=1)
    discriminating_test: DiscriminatingTestDraft
    unresolved_questions: list[str] = Field(default_factory=list)
    provenance_normalizations: list[str] = Field(default_factory=list)

    epistemic_status: Literal["hypothesis_only"] = "hypothesis_only"
    requires_verification: Literal[True] = True
    shadow_only: Literal[True] = True
    tension_witness_is_conflict_authority: Literal[False] = False
    contradiction_claim_established: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_candidate(self) -> "ScientificContradictionResolutionCandidate":
        premise_ids = set(self.premise_statement_ids)
        if len(premise_ids) < 2:
            raise ValueError(
                "CONTRADICTION_RESOLUTION requires at least two grounded premises"
            )
        if set(self.side_a_statement_ids) & set(self.side_b_statement_ids):
            raise ValueError("contradiction sides must be disjoint")
        if not set(self.side_a_statement_ids) <= premise_ids:
            raise ValueError("side A statements must be selected grounded premises")
        if not set(self.side_b_statement_ids) <= premise_ids:
            raise ValueError("side B statements must be selected grounded premises")
        resolution_explained = set(self.resolution_model.explained_statement_ids)
        if not resolution_explained & set(self.side_a_statement_ids):
            raise ValueError("resolution model must explain side A")
        if not resolution_explained & set(self.side_b_statement_ids):
            raise ValueError("resolution model must explain side B")
        if self.baseline_model.summary.strip() == self.resolution_model.summary.strip():
            raise ValueError("baseline and resolution model summaries must differ")
        if any(
            row.baseline_expectation.strip() == row.alternative_expectation.strip()
            for row in self.differential_predictions
        ):
            raise ValueError("resolution predictions must differentiate models")
        if not self.discriminating_test.primary_observables:
            raise ValueError("resolution test requires primary observables")
        return self


class ContradictionResolutionRunReport(StrictModel):
    schema_version: Literal[
        "scientific-contradiction-resolution-shadow-report-v1"
    ] = "scientific-contradiction-resolution-shadow-report-v1"
    report_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    source_tension_report_id: str = Field(min_length=1)
    source_tension_report_sha256: str = Field(min_length=64, max_length=64)
    backend_name: str
    model_name: str
    decision: Literal[
        "generated",
        "abstained",
        "rejected_invalid_draft",
        "skipped_no_eligible_tension",
        "generation_failed",
    ]
    candidate_ids: list[str] = Field(default_factory=list)
    candidates: list[ScientificContradictionResolutionCandidate] = Field(
        default_factory=list
    )
    eligible_witness_ids: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    compile_issues: list[str] = Field(default_factory=list)
    rejected_candidate_count: int = Field(default=0, ge=0)
    llm_calls_performed: int = Field(ge=0, le=1)
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None

    shadow_only: Literal[True] = True
    source_tensions_are_conflict_authority: Literal[False] = False
    contradiction_claim_established: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> "ContradictionResolutionRunReport":
        if self.candidate_ids != [row.candidate_id for row in self.candidates]:
            raise ValueError("candidate_ids must match candidates in stable order")
        if self.decision == "skipped_no_eligible_tension" and self.llm_calls_performed:
            raise ValueError("tension-gated skip must not perform an LLM call")
        return self


@dataclass(frozen=True)
class ContradictionResolutionPrompt:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class ContradictionResolutionGeneration:
    draft: ContradictionResolutionBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class ContradictionResolutionBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(
        self, prompt: ContradictionResolutionPrompt
    ) -> ContradictionResolutionGeneration: ...


class InstructorOpenAICompatibleContradictionBackend:
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
                "CONTRADICTION_RESOLUTION generation requires installed openai and instructor packages."
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
        self, prompt: ContradictionResolutionPrompt
    ) -> ContradictionResolutionGeneration:
        from pipeline_core.llm.llm_telemetry import run_instructor_structured_call

        started = time.perf_counter()
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=ContradictionResolutionBatchDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "scientific_reframing_shadow",
                "stage": "contradiction_resolution",
                "call_kind": "scientific_contradiction_resolution_generation",
            },
            semantic_components={"operator_id": "CONTRADICTION_RESOLUTION"},
        )
        elapsed = event.elapsed_seconds
        if elapsed is None:
            elapsed = time.perf_counter() - started
        if not isinstance(draft, ContradictionResolutionBatchDraft):
            draft = ContradictionResolutionBatchDraft.model_validate(draft)
        return ContradictionResolutionGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=elapsed,
        )


_STRONG_TENSION_TYPES = {"DIRECTIONAL", "NULL_EFFECT", "POTENTIAL_CONFLICT"}

_POSITIVE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:associated|correlat(?:e|ed|ion)|contribut(?:e|ed|ion)|promot(?:e|ed)|favor(?:s|ed)?)\b",
        r"\b(?:increase|increased|higher|greater|stronger|enhanced|improved)\b",
        r"\b(?:match|matching|proximity|closer)\b",
    )
)
_NULL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:not|no)\s+(?:a\s+)?significant(?:ly)?\b",
        r"\b(?:does|did|do)\s+not\b",
        r"\b(?:negligible|insignificant|unchanged|no\s+effect|no\s+change)\b",
        r"\b(?:not\s+(?:a\s+)?(?:major|meaningful|important)\s+contributor)\b",
    )
)
_OFFSET_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:offset|mismatch|diverg(?:e|ed|ence)|decoupl(?:e|ed|ing))\b",
        r"\b(?:opposite|inverse|reversal|reversed)\b",
    )
)


def _matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _statement_index(evidence: ScientificReframeEvidencePacket):
    return {
        row.statement_id: row
        for row in [*evidence.premise_statements, *evidence.gap_statements]
    }


def _side_text(ids: list[str], by_statement: Mapping[str, Any]) -> str:
    return " ".join(
        by_statement[row].text for row in ids if row in by_statement
    )


def _semantic_incompatibility_signals(
    *,
    witness: EvidenceLevelTensionWitness,
    by_statement: Mapping[str, Any],
) -> list[str]:
    side_a = _side_text(witness.side_a_statement_ids, by_statement)
    side_b = _side_text(witness.side_b_statement_ids, by_statement)
    signals: list[str] = []

    a_positive = _matches(side_a, _POSITIVE_PATTERNS)
    b_positive = _matches(side_b, _POSITIVE_PATTERNS)
    a_null = _matches(side_a, _NULL_PATTERNS)
    b_null = _matches(side_b, _NULL_PATTERNS)
    a_offset = _matches(side_a, _OFFSET_PATTERNS)
    b_offset = _matches(side_b, _OFFSET_PATTERNS)

    if (a_positive and b_null) or (b_positive and a_null):
        signals.append("positive_vs_null_effect")
    if (a_positive and b_offset) or (b_positive and a_offset):
        signals.append("association_vs_offset_or_mismatch")
    if a_offset != b_offset and (a_positive or b_positive):
        signals.append("one_side_only_offset_or_divergence")
    return sorted(set(signals))


def build_contradiction_resolution_input(
    *,
    evidence: ScientificReframeEvidencePacket,
    tensions: ScientificEvidenceTensionReport,
) -> ContradictionResolutionInput:
    if tensions.source_task_id != evidence.task_id:
        raise ValueError("tension report task_id must equal evidence task_id")
    if tensions.source_context_id != evidence.source_context_id:
        raise ValueError("tension report context_id must equal evidence context_id")
    if tensions.source_context_sha256 != evidence.source_context_sha256:
        raise ValueError("tension report context SHA must equal evidence context SHA")

    by_statement = _statement_index(evidence)
    premise_ids = {row.statement_id for row in evidence.premise_statements}
    seeds: list[ContradictionResolutionSeed] = []
    for witness in tensions.witnesses:
        side_a_ids = [row for row in witness.side_a_statement_ids if row in premise_ids]
        side_b_ids = [row for row in witness.side_b_statement_ids if row in premise_ids]
        if not side_a_ids or not side_b_ids:
            continue
        signals = _semantic_incompatibility_signals(
            witness=witness,
            by_statement=by_statement,
        )
        bases: list[str] = []
        paired = bool(side_a_ids and side_b_ids)
        if paired:
            bases.append("paired_grounded_premise_sides")
        if witness.independent_family_signal:
            bases.append("independent_evidence_family_signal")
        strong_types = sorted(_STRONG_TENSION_TYPES & set(witness.tension_types))
        if strong_types:
            bases.append("strong_tension_type:" + ",".join(strong_types))
        if signals:
            bases.append("semantic_incompatibility_signal")

        eligible = bool(
            paired
            and witness.independent_family_signal
            and (strong_types or signals)
        )
        focal_text = (
            by_statement[witness.focal_statement_id].text
            if witness.focal_statement_id in by_statement
            else None
        )
        seeds.append(
            ContradictionResolutionSeed(
                witness_id=witness.witness_id,
                source_tension_id=witness.source_tension_id,
                source_tension_type=witness.source_tension_type,
                tension_types=list(witness.tension_types),
                side_a_statement_ids=side_a_ids,
                side_b_statement_ids=side_b_ids,
                side_a_texts=[by_statement[row].text for row in side_a_ids],
                side_b_texts=[by_statement[row].text for row in side_b_ids],
                focal_statement_id=(
                    witness.focal_statement_id
                    if witness.focal_statement_id in by_statement
                    else None
                ),
                focal_statement_text=focal_text,
                paper_ids=list(witness.paper_ids),
                independence_basis=witness.independence_basis,
                independent_family_signal=witness.independent_family_signal,
                paired_response_signal=witness.paired_response_signal,
                relevant_condition_names=list(witness.relevant_condition_names),
                semantic_incompatibility_signals=signals,
                eligibility_bases=bases,
                attempt_eligible=eligible,
            )
        )

    seeds.sort(key=lambda row: row.witness_id)
    eligible_ids = [row.witness_id for row in seeds if row.attempt_eligible]
    return ContradictionResolutionInput(
        task_id=evidence.task_id,
        source_context_id=evidence.source_context_id,
        source_context_sha256=evidence.source_context_sha256,
        question=evidence.question,
        source_tension_report_id=tensions.report_id,
        source_tension_report_sha256=_sha256(tensions.model_dump(mode="json")),
        seeds=seeds,
        eligible_witness_ids=eligible_ids,
    )


def build_contradiction_resolution_prompt(
    *,
    evidence: ScientificReframeEvidencePacket,
    resolution_input: ContradictionResolutionInput,
) -> ContradictionResolutionPrompt:
    eligible = [row for row in resolution_input.seeds if row.attempt_eligible]
    if not eligible:
        raise ValueError("cannot build CONTRADICTION_RESOLUTION prompt without eligible tension")

    premise_rows = [
        {
            "statement_id": row.statement_id,
            "text": row.text,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
        }
        for row in evidence.premise_statements
    ]
    gap_rows = [
        {
            "statement_id": row.statement_id,
            "text": row.text,
            "epistemic_role": row.epistemic_role,
            "claim_kind": row.claim_kind,
            "paper_ids": row.paper_ids,
        }
        for row in evidence.gap_statements
    ]
    seed_rows = [row.model_dump(mode="json") for row in eligible]

    system_prompt = """You are the shadow-only CONTRADICTION_RESOLUTION scientific reframing operator.

Your job is NOT to declare the literature contradictory. The supplied tension witnesses are diagnostic contrasts without conflict authority. Your job is to ask whether apparently incompatible reported observations can be jointly explained by a minimal conditional model.

Generate at most ONE candidate. Abstain if a reconciliation would require unsupported specificity.

A valid candidate must:
- select one or more eligible tension_witness_ids exactly as supplied;
- cite at least one grounded premise from side A and at least one from side B;
- state the apparent contradiction without promoting it to established conflict;
- give a naive/single-rule baseline model that cannot naturally accommodate both sides;
- give a minimal resolution model that explains at least one selected statement from BOTH sides;
- prefer already grounded context differences, scope differences, measurement conditions, or competing mechanisms before inventing a new latent construct;
- identify distinguishing context variables when they are supported or directly suggested by the evidence;
- provide at least one differential prediction, one falsifier, and a discriminating test;
- keep all proposed reconciliation constructs hypothesis-only;
- never treat tension witnesses, gaps, or synthesis statements as novelty or conflict authority.

This operator is distinct from LATENT_VARIABLE, REGIME_BOUNDARY, and PROXY_CHALLENGE. Do not merely rename a hidden variable, a regime boundary, or a proxy limitation. The core transformation is: seemingly incompatible evidence -> a minimal model specifying why both can hold under different conditions/scopes/mechanism balances.
"""

    payload = {
        "task": {
            "task_id": evidence.task_id,
            "question": evidence.question,
        },
        "eligible_tension_witnesses": seed_rows,
        "grounded_premises": premise_rows,
        "grounded_gaps": gap_rows,
        "output_contract_notes": {
            "max_candidates": 1,
            "tension_is_not_conflict_authority": True,
            "candidate_is_hypothesis_only": True,
            "requires_verification": True,
            "side_a_and_side_b_grounded_coverage_required": True,
        },
    }
    user_prompt = (
        "Resolve at most one apparent contradiction for this task using only the "
        "grounded statements below. Return an empty candidates list with an "
        "abstention_reason if no disciplined reconciliation is warranted.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return ContradictionResolutionPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


class ContradictionResolutionCompileError(ValueError):
    pass


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _one_edit_apart(left: str, right: str) -> bool:
    if left == right:
        return False
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    edits += int(j < len(right))
    return edits == 1


def _normalize_witness_ids(
    values: list[str],
    eligible_ids: list[str],
) -> tuple[list[str], list[str]]:
    eligible = set(eligible_ids)
    result: list[str] = []
    notes: list[str] = []
    for raw in _dedupe(values):
        if raw in eligible:
            result.append(raw)
            continue
        candidates = [row for row in eligible_ids if _one_edit_apart(raw, row)]
        if len(candidates) == 1:
            result.append(candidates[0])
            notes.append(
                "normalized one-edit tension witness ID copy error: "
                f"{raw} -> {candidates[0]}"
            )
            continue
        raise ContradictionResolutionCompileError(
            "candidate references unknown or ineligible tension witness ID: " + raw
        )
    return _dedupe(result), notes


def compile_contradiction_resolution_candidate(
    *,
    evidence: ScientificReframeEvidencePacket,
    resolution_input: ContradictionResolutionInput,
    draft: ContradictionResolutionDraft,
) -> ScientificContradictionResolutionCandidate:
    witness_ids, normalizations = _normalize_witness_ids(
        draft.tension_witness_ids,
        resolution_input.eligible_witness_ids,
    )
    if not witness_ids:
        raise ContradictionResolutionCompileError(
            "candidate must reference an eligible tension witness"
        )

    seed_by_id = {row.witness_id: row for row in resolution_input.seeds}
    selected_seeds = [seed_by_id[row] for row in witness_ids]
    allowed_side_a = {
        item for seed in selected_seeds for item in seed.side_a_statement_ids
    }
    allowed_side_b = {
        item for seed in selected_seeds for item in seed.side_b_statement_ids
    }
    known_premises = {row.statement_id for row in evidence.premise_statements}
    premise_order = [row.statement_id for row in evidence.premise_statements]
    known_gaps = {row.statement_id for row in evidence.gap_statements}
    gap_order = [row.statement_id for row in evidence.gap_statements]

    side_a = _dedupe(draft.side_a_statement_ids)
    side_b = _dedupe(draft.side_b_statement_ids)
    if not side_a or not side_b:
        raise ContradictionResolutionCompileError(
            "candidate must cite both contradiction sides"
        )
    if not set(side_a) <= allowed_side_a:
        unknown = sorted(set(side_a) - allowed_side_a)
        raise ContradictionResolutionCompileError(
            "candidate side A references statements outside selected witnesses: "
            + ", ".join(unknown)
        )
    if not set(side_b) <= allowed_side_b:
        unknown = sorted(set(side_b) - allowed_side_b)
        raise ContradictionResolutionCompileError(
            "candidate side B references statements outside selected witnesses: "
            + ", ".join(unknown)
        )
    if set(side_a) & set(side_b):
        raise ContradictionResolutionCompileError(
            "candidate contradiction sides must be disjoint"
        )

    # Require at least one selected witness to be represented on both sides.
    if not any(
        (set(side_a) & set(seed.side_a_statement_ids))
        and (set(side_b) & set(seed.side_b_statement_ids))
        for seed in selected_seeds
    ):
        raise ContradictionResolutionCompileError(
            "candidate does not cover both sides of any selected tension witness"
        )

    declared = set(_dedupe(draft.premise_statement_ids)) | set(side_a) | set(side_b)
    explained = set(draft.baseline_model.explained_statement_ids) | set(
        draft.resolution_model.explained_statement_ids
    )
    unknown_premises = sorted((declared | explained) - known_premises)
    if unknown_premises:
        raise ContradictionResolutionCompileError(
            "candidate references non-grounded premise IDs: "
            + ", ".join(unknown_premises)
        )
    selected_premises = [
        row for row in premise_order if row in (declared | explained)
    ]
    if len(selected_premises) < 2:
        raise ContradictionResolutionCompileError(
            "CONTRADICTION_RESOLUTION requires at least two grounded premises"
        )

    declared_gaps = set(_dedupe(draft.gap_statement_ids))
    unknown_gaps = sorted(declared_gaps - known_gaps)
    if unknown_gaps:
        raise ContradictionResolutionCompileError(
            "candidate references non-grounded gap IDs: " + ", ".join(unknown_gaps)
        )
    selected_gaps = [row for row in gap_order if row in declared_gaps]

    def normalize_model(model: ScientificModelDraft, name: str) -> ScientificModelDraft:
        ids = set(_dedupe(model.explained_statement_ids))
        canonical = [row for row in premise_order if row in ids]
        if not canonical:
            raise ContradictionResolutionCompileError(
                f"{name} must explain a grounded premise"
            )
        if not model.expected_observations:
            raise ContradictionResolutionCompileError(
                f"{name} requires expected observations"
            )
        if canonical != list(model.explained_statement_ids):
            normalizations.append(f"canonicalized {name}.explained_statement_ids")
        return model.model_copy(update={"explained_statement_ids": canonical})

    baseline = normalize_model(draft.baseline_model, "baseline_model")
    resolution = normalize_model(draft.resolution_model, "resolution_model")
    if not set(resolution.explained_statement_ids) & set(side_a):
        raise ContradictionResolutionCompileError(
            "resolution model must explain at least one selected side A statement"
        )
    if not set(resolution.explained_statement_ids) & set(side_b):
        raise ContradictionResolutionCompileError(
            "resolution model must explain at least one selected side B statement"
        )
    if baseline.summary.strip() == resolution.summary.strip():
        raise ContradictionResolutionCompileError(
            "baseline and resolution summaries are identical"
        )

    predictions = [
        row
        for row in draft.differential_predictions
        if row.baseline_expectation.strip() != row.alternative_expectation.strip()
    ]
    if not predictions:
        raise ContradictionResolutionCompileError(
            "candidate requires a differential prediction"
        )
    if len({row.local_id for row in predictions}) != len(predictions):
        raise ContradictionResolutionCompileError(
            "duplicate differential prediction local_id"
        )
    if not draft.falsifiers:
        raise ContradictionResolutionCompileError("candidate requires a falsifier")
    if len({row.local_id for row in draft.falsifiers}) != len(draft.falsifiers):
        raise ContradictionResolutionCompileError("duplicate falsifier local_id")
    if not draft.discriminating_test.primary_observables:
        raise ContradictionResolutionCompileError(
            "discriminating test requires primary observables"
        )

    normalized = draft.model_copy(
        update={
            "tension_witness_ids": witness_ids,
            "premise_statement_ids": selected_premises,
            "gap_statement_ids": selected_gaps,
            "side_a_statement_ids": [row for row in premise_order if row in set(side_a)],
            "side_b_statement_ids": [row for row in premise_order if row in set(side_b)],
            "baseline_model": baseline,
            "resolution_model": resolution,
            "differential_predictions": predictions,
        }
    )
    payload = normalized.model_dump(mode="json")
    candidate_id = "scientific_contradiction_resolution:" + hashlib.sha256(
        (
            evidence.task_id
            + "|"
            + evidence.source_context_sha256
            + "|"
            + resolution_input.source_tension_report_sha256
            + "|"
            + _canonical_json(payload)
        ).encode("utf-8")
    ).hexdigest()[:20]

    return ScientificContradictionResolutionCandidate(
        candidate_id=candidate_id,
        source_task_id=evidence.task_id,
        source_context_id=evidence.source_context_id,
        source_context_sha256=evidence.source_context_sha256,
        source_tension_report_id=resolution_input.source_tension_report_id,
        source_tension_report_sha256=resolution_input.source_tension_report_sha256,
        title=normalized.title,
        tension_witness_ids=normalized.tension_witness_ids,
        premise_statement_ids=normalized.premise_statement_ids,
        gap_statement_ids=normalized.gap_statement_ids,
        side_a_statement_ids=normalized.side_a_statement_ids,
        side_b_statement_ids=normalized.side_b_statement_ids,
        baseline_model=normalized.baseline_model,
        resolution_model=normalized.resolution_model,
        apparent_contradiction=normalized.apparent_contradiction,
        resolution_principle=normalized.resolution_principle,
        resolution_kind=normalized.resolution_kind,
        distinguishing_context_variables=normalized.distinguishing_context_variables,
        proposed_resolution_constructs=normalized.proposed_resolution_constructs,
        differential_predictions=normalized.differential_predictions,
        falsifiers=normalized.falsifiers,
        discriminating_test=normalized.discriminating_test,
        unresolved_questions=normalized.unresolved_questions,
        provenance_normalizations=normalizations,
    )


class ContradictionResolutionShadowRuntime:
    def __init__(self, backend: ContradictionResolutionBackend) -> None:
        self.backend = backend

    def run(
        self,
        *,
        evidence: ScientificReframeEvidencePacket,
        resolution_input: ContradictionResolutionInput,
    ) -> tuple[ContradictionResolutionRunReport, ContradictionResolutionPrompt | None]:
        if evidence.task_id != resolution_input.task_id:
            raise ValueError("resolution input task_id does not match evidence")
        if not resolution_input.eligible_witness_ids:
            return (
                self._report(
                    evidence=evidence,
                    resolution_input=resolution_input,
                    decision="skipped_no_eligible_tension",
                    candidates=[],
                    abstention_reason=(
                        "No paired independent evidence tension met the shadow-only "
                        "CONTRADICTION_RESOLUTION attempt gate."
                    ),
                    llm_calls=0,
                ),
                None,
            )

        prompt = build_contradiction_resolution_prompt(
            evidence=evidence,
            resolution_input=resolution_input,
        )
        try:
            generation = self.backend.generate(prompt)
        except Exception as exc:
            return (
                self._report(
                    evidence=evidence,
                    resolution_input=resolution_input,
                    decision="generation_failed",
                    candidates=[],
                    abstention_reason=f"{type(exc).__name__}: {exc}",
                    llm_calls=1,
                ),
                prompt,
            )

        candidates: list[ScientificContradictionResolutionCandidate] = []
        issues: list[str] = []
        for draft in generation.draft.candidates[:1]:
            try:
                candidates.append(
                    compile_contradiction_resolution_candidate(
                        evidence=evidence,
                        resolution_input=resolution_input,
                        draft=draft,
                    )
                )
            except Exception as exc:
                issues.append(
                    f"candidate {draft.local_id}: {type(exc).__name__} {exc}"
                )

        if candidates:
            decision = "generated"
            abstention = None
        elif generation.draft.candidates:
            decision = "rejected_invalid_draft"
            abstention = "All generated candidates failed contradiction-resolution compilation."
        else:
            decision = "abstained"
            abstention = generation.draft.abstention_reason or (
                "Model returned no contradiction-resolution candidate."
            )

        return (
            self._report(
                evidence=evidence,
                resolution_input=resolution_input,
                decision=decision,
                candidates=candidates,
                abstention_reason=abstention,
                compile_issues=issues,
                rejected_candidate_count=len(issues),
                llm_calls=1,
                generation=generation,
            ),
            prompt,
        )

    def _report(
        self,
        *,
        evidence: ScientificReframeEvidencePacket,
        resolution_input: ContradictionResolutionInput,
        decision: str,
        candidates: list[ScientificContradictionResolutionCandidate],
        abstention_reason: str | None,
        llm_calls: int,
        compile_issues: list[str] | None = None,
        rejected_candidate_count: int = 0,
        generation: ContradictionResolutionGeneration | None = None,
    ) -> ContradictionResolutionRunReport:
        candidate_ids = [row.candidate_id for row in candidates]
        report_payload = {
            "task_id": evidence.task_id,
            "context_sha": evidence.source_context_sha256,
            "tension_sha": resolution_input.source_tension_report_sha256,
            "decision": decision,
            "candidate_ids": candidate_ids,
        }
        report_id = "scientific_contradiction_resolution_shadow:" + hashlib.sha256(
            _canonical_json(report_payload).encode("utf-8")
        ).hexdigest()[:20]
        return ContradictionResolutionRunReport(
            report_id=report_id,
            source_task_id=evidence.task_id,
            source_context_id=evidence.source_context_id,
            source_context_sha256=evidence.source_context_sha256,
            source_tension_report_id=resolution_input.source_tension_report_id,
            source_tension_report_sha256=resolution_input.source_tension_report_sha256,
            backend_name=self.backend.backend_name,
            model_name=self.backend.model_name,
            decision=decision,
            candidate_ids=candidate_ids,
            candidates=candidates,
            eligible_witness_ids=list(resolution_input.eligible_witness_ids),
            abstention_reason=abstention_reason,
            compile_issues=list(compile_issues or []),
            rejected_candidate_count=rejected_candidate_count,
            llm_calls_performed=llm_calls,
            input_tokens=(generation.input_tokens if generation else None),
            output_tokens=(generation.output_tokens if generation else None),
            response_id=(generation.response_id if generation else None),
            elapsed_seconds=(generation.elapsed_seconds if generation else None),
        )
