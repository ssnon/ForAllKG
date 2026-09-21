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

from pipeline_core.corpus.semantic_ir.annotation import SemanticAnnotation
from pipeline_core.discovery.reframing.proxy_enrichment import (
    EXTRACTOR_VERSION,
    ProxySemanticAnnotationValue,
    ProxySemanticEnrichmentReport,
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


ProxyFailureMode = Literal[
    "non_equivalence",
    "context_dependence",
    "construct_undercoverage",
    "measurement_conflation",
    "surrogate_breakdown",
    "other",
]


class ProxyChallengeSeed(StrictModel):
    annotation_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    measurement_id: str = Field(min_length=1)
    metric: str | None = None
    proxy_role: str = Field(min_length=1)
    target_construct: str = Field(min_length=1)
    interchangeability: str = Field(min_length=1)
    interpretation_summary: str = Field(min_length=1)
    support_kind: str = Field(min_length=1)
    support_quotes: list[str] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    context_dependencies: list[str] = Field(default_factory=list)
    matched_task_terms: list[str] = Field(default_factory=list)
    task_relevant_hint: bool = False

    source_interpretation_only: Literal[True] = True
    positive_premise_authority: Literal[False] = False
    scientific_equivalence_authority: Literal[False] = False


class ProxyChallengeInput(StrictModel):
    schema_version: Literal[
        "proxy-challenge-input-v1"
    ] = "proxy-challenge-input-v1"
    task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_enrichment_report_id: str = Field(min_length=1)
    source_annotation_sha256: str = Field(min_length=64, max_length=64)
    seeds: list[ProxyChallengeSeed] = Field(default_factory=list)
    task_relevant_seed_ids: list[str] = Field(default_factory=list)

    enrichment_coverage_complete: Literal[True] = True
    annotation_is_positive_premise: Literal[False] = False
    annotation_is_scientific_equivalence_authority: Literal[False] = False
    missing_annotation_is_negative_evidence: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_seed_partition(self) -> "ProxyChallengeInput":
        seed_ids = [row.annotation_id for row in self.seeds]
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError("proxy challenge seed annotation IDs must be unique")
        expected = [row.annotation_id for row in self.seeds if row.task_relevant_hint]
        if self.task_relevant_seed_ids != expected:
            raise ValueError(
                "task_relevant_seed_ids must equal relevant seeds in stable order"
            )
        return self


class ProxyChallengeDraft(StrictModel):
    local_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    semantic_seed_annotation_ids: list[str] = Field(min_length=1)
    premise_statement_ids: list[str] = Field(min_length=2)
    gap_statement_ids: list[str] = Field(default_factory=list)
    baseline_model: ScientificModelDraft
    alternative_model: ScientificModelDraft
    challenged_proxy_assumption: str = Field(min_length=1)
    challenged_observable: str = Field(min_length=1)
    target_construct: str = Field(min_length=1)
    proxy_failure_mode: ProxyFailureMode
    differential_predictions: list[DifferentialPredictionDraft] = Field(min_length=1)
    falsifiers: list[ReframeFalsifierDraft] = Field(min_length=1)
    discriminating_test: DiscriminatingTestDraft
    unresolved_questions: list[str] = Field(default_factory=list)


class ProxyChallengeBatchDraft(StrictModel):
    schema_version: Literal[
        "proxy-challenge-batch-draft-v1"
    ] = "proxy-challenge-batch-draft-v1"
    candidates: list[ProxyChallengeDraft] = Field(default_factory=list)
    abstention_reason: str | None = None


class ScientificProxyChallengeCandidate(StrictModel):
    schema_version: Literal[
        "scientific-proxy-challenge-candidate-v1"
    ] = "scientific-proxy-challenge-candidate-v1"
    candidate_id: str = Field(min_length=1)
    operator_id: Literal["PROXY_CHALLENGE"] = "PROXY_CHALLENGE"
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    source_enrichment_report_id: str = Field(min_length=1)
    source_annotation_sha256: str = Field(min_length=64, max_length=64)

    title: str = Field(min_length=1)
    semantic_seed_annotation_ids: list[str] = Field(min_length=1)
    premise_statement_ids: list[str] = Field(min_length=2)
    gap_statement_ids: list[str] = Field(default_factory=list)
    baseline_model: ScientificModelDraft
    alternative_model: ScientificModelDraft
    challenged_proxy_assumption: str = Field(min_length=1)
    challenged_observable: str = Field(min_length=1)
    target_construct: str = Field(min_length=1)
    proxy_failure_mode: ProxyFailureMode
    differential_predictions: list[DifferentialPredictionDraft] = Field(min_length=1)
    falsifiers: list[ReframeFalsifierDraft] = Field(min_length=1)
    discriminating_test: DiscriminatingTestDraft
    unresolved_questions: list[str] = Field(default_factory=list)
    provenance_normalizations: list[str] = Field(default_factory=list)

    epistemic_status: Literal["hypothesis_only"] = "hypothesis_only"
    requires_verification: Literal[True] = True
    shadow_only: Literal[True] = True
    proxy_annotation_is_positive_premise: Literal[False] = False
    proxy_annotation_is_scientific_equivalence_authority: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_candidate(self) -> "ScientificProxyChallengeCandidate":
        premise_ids = set(self.premise_statement_ids)
        if len(premise_ids) < 2:
            raise ValueError("PROXY_CHALLENGE requires at least two grounded premises")
        if len(self.semantic_seed_annotation_ids) != len(
            set(self.semantic_seed_annotation_ids)
        ):
            raise ValueError("semantic seed annotation IDs must be unique")
        if len(self.gap_statement_ids) != len(set(self.gap_statement_ids)):
            raise ValueError("gap statement IDs must be unique")
        for name, model in (
            ("baseline_model", self.baseline_model),
            ("alternative_model", self.alternative_model),
        ):
            if not model.explained_statement_ids:
                raise ValueError(f"{name} must explain grounded premises")
            if not set(model.explained_statement_ids) <= premise_ids:
                raise ValueError(f"{name} references non-selected premises")
            if not model.expected_observations:
                raise ValueError(f"{name} requires expected observations")
        if len(set(self.alternative_model.explained_statement_ids)) < 2:
            raise ValueError(
                "PROXY_CHALLENGE alternative model must explain at least two premises"
            )
        if self.baseline_model.summary.strip() == self.alternative_model.summary.strip():
            raise ValueError("baseline and alternative model summaries must differ")
        if any(
            row.baseline_expectation.strip() == row.alternative_expectation.strip()
            for row in self.differential_predictions
        ):
            raise ValueError("proxy challenge predictions must differentiate models")
        if not self.discriminating_test.primary_observables:
            raise ValueError("proxy challenge test requires primary observables")
        return self


class ProxyChallengeRunReport(StrictModel):
    schema_version: Literal[
        "scientific-proxy-challenge-shadow-report-v1"
    ] = "scientific-proxy-challenge-shadow-report-v1"
    report_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    source_enrichment_report_id: str = Field(min_length=1)
    source_annotation_sha256: str = Field(min_length=64, max_length=64)
    backend_name: str
    model_name: str
    decision: Literal[
        "generated",
        "abstained",
        "rejected_invalid_draft",
        "skipped_no_task_relevant_seed",
        "generation_failed",
    ]
    candidate_ids: list[str] = Field(default_factory=list)
    candidates: list[ScientificProxyChallengeCandidate] = Field(default_factory=list)
    task_relevant_seed_ids: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    compile_issues: list[str] = Field(default_factory=list)
    rejected_candidate_count: int = Field(default=0, ge=0)
    llm_calls_performed: int = Field(ge=0, le=1)
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None

    shadow_only: Literal[True] = True
    source_annotations_are_positive_premises: Literal[False] = False
    source_annotations_have_scientific_equivalence_authority: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> "ProxyChallengeRunReport":
        if self.candidate_ids != [row.candidate_id for row in self.candidates]:
            raise ValueError("candidate_ids must match candidates in stable order")
        if self.decision == "skipped_no_task_relevant_seed" and self.llm_calls_performed:
            raise ValueError("seed-gated skip must not perform an LLM call")
        return self


@dataclass(frozen=True)
class ProxyChallengePrompt:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class ProxyChallengeGeneration:
    draft: ProxyChallengeBatchDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


@runtime_checkable
class ProxyChallengeBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(self, prompt: ProxyChallengePrompt) -> ProxyChallengeGeneration: ...


class InstructorOpenAICompatibleProxyChallengeBackend:
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
                "PROXY_CHALLENGE generation requires installed openai and instructor packages."
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

    def generate(self, prompt: ProxyChallengePrompt) -> ProxyChallengeGeneration:
        from pipeline_core.llm.llm_telemetry import run_instructor_structured_call

        started = time.perf_counter()
        draft, event = run_instructor_structured_call(
            self._get_client().chat.completions,
            model=self.model_name,
            response_model=ProxyChallengeBatchDraft,
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
                "stage": "proxy_challenge",
                "call_kind": "scientific_proxy_challenge_generation",
            },
            semantic_components={"operator_id": "PROXY_CHALLENGE"},
        )
        elapsed = event.elapsed_seconds
        if elapsed is None:
            elapsed = time.perf_counter() - started
        if not isinstance(draft, ProxyChallengeBatchDraft):
            draft = ProxyChallengeBatchDraft.model_validate(draft)
        return ProxyChallengeGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
            elapsed_seconds=elapsed,
        )


_RELEVANCE_STOPWORDS = {
    "about", "after", "against", "also", "among", "around", "based", "before",
    "between", "change", "changes", "condition", "conditions", "construct",
    "different", "effect", "effects", "enhancement", "evidence", "from", "gold",
    "indicate", "indicates", "intensity", "measurement", "measurements", "metal",
    "nanoparticle", "nanoparticles", "observed", "particle", "particles", "property",
    "raman", "reported", "response", "sers", "signal", "signals", "silver", "substrate",
    "substrates", "surface", "through", "using", "value", "values", "wavelength",
    "with", "without", "their", "there", "these", "this", "those", "toward",
    "under", "used", "uses", "were", "while", "which", "would",
    "the", "and", "are", "can", "does", "how", "not", "one", "whether",
    "associated", "because", "interpretation", "interpreted", "specific",
    "increased", "decreased", "higher", "lower", "presence",
    "band", "spectral", "shift", "optical", "excitation", "concentration",
    "nir-ii", "resonance", "than", "that", "for", "observable", "distinct",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
        if token not in _RELEVANCE_STOPWORDS and not token.isdigit()
    }


def _annotation_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_proxy_annotations(path: str | Path) -> list[SemanticAnnotation]:
    rows: list[SemanticAnnotation] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = SemanticAnnotation.model_validate_json(line)
        if row.capability != "proxy_semantics":
            raise ValueError(
                f"annotation line {line_number} has capability {row.capability!r}, expected proxy_semantics"
            )
        if row.epistemic_status != "source_interpretation":
            raise ValueError("proxy challenge accepts source_interpretation annotations only")
        rows.append(row)
    return rows


def build_proxy_challenge_input(
    *,
    evidence: ScientificReframeEvidencePacket,
    enrichment_report: ProxySemanticEnrichmentReport,
    annotations: list[SemanticAnnotation],
    annotation_path: str | Path,
) -> ProxyChallengeInput:
    if enrichment_report.scope_id != evidence.task_id:
        raise ValueError("proxy enrichment scope_id must equal evidence task_id")
    if not enrichment_report.coverage_complete_for_plan:
        raise ValueError("PROXY_CHALLENGE requires complete targeted enrichment coverage")
    if enrichment_report.failed_target_count:
        raise ValueError("PROXY_CHALLENGE requires zero failed enrichment targets")
    if enrichment_report.total_annotation_count != len(annotations):
        raise ValueError("annotation sidecar count does not match enrichment report")
    if enrichment_report.extractor_version != EXTRACTOR_VERSION:
        raise ValueError("proxy annotation extractor version does not match runtime")

    # Task relevance is intentionally anchored to the user-facing scientific
    # question, not to every grounded premise.  The proxy sidecars were
    # recovered from the same evidence neighborhood, so premise-wide lexical
    # overlap is almost guaranteed and previously admitted unrelated seeds
    # (for example biocompatibility or molecular-orientation annotations)
    # through generic words.  A seed must overlap the task's actual question
    # surface; gap/premise content remains available later to generation as
    # grounded scientific evidence, but cannot manufacture seed relevance.
    task_tokens = _tokens(evidence.question)
    seeds: list[ProxyChallengeSeed] = []
    for annotation in annotations:
        value = ProxySemanticAnnotationValue.model_validate(annotation.value)
        chunk_refs = [row for row in annotation.source_refs if row.object_kind == "chunk"]
        if not chunk_refs:
            raise ValueError(
                f"proxy annotation {annotation.annotation_id} lacks a source chunk ref"
            )
        source_ref = chunk_refs[0]
        semantic_text = "\n".join(
            part
            for part in (
                value.metric or "",
                value.target_construct,
                value.interpretation_summary,
                value.distinct_target_rationale,
                *value.limitations,
                *value.context_dependencies,
            )
            if part
        )
        matched = sorted(task_tokens & _tokens(semantic_text))
        seeds.append(
            ProxyChallengeSeed(
                annotation_id=annotation.annotation_id,
                paper_id=annotation.subject_ref.paper_id,
                chunk_id=str(annotation.subject_ref.chunk_id or source_ref.object_id),
                measurement_id=value.measurement_id,
                metric=value.metric,
                proxy_role=value.proxy_role,
                target_construct=value.target_construct,
                interchangeability=value.interchangeability,
                interpretation_summary=value.interpretation_summary,
                support_kind=value.support_kind,
                support_quotes=list(value.support_quotes),
                limitations=list(value.limitations),
                context_dependencies=list(value.context_dependencies),
                matched_task_terms=matched,
                task_relevant_hint=bool(matched),
            )
        )

    return ProxyChallengeInput(
        task_id=evidence.task_id,
        source_context_id=evidence.source_context_id,
        source_context_sha256=evidence.source_context_sha256,
        question=evidence.question,
        source_enrichment_report_id=enrichment_report.report_id,
        source_annotation_sha256=_annotation_sha256(Path(annotation_path)),
        seeds=seeds,
        task_relevant_seed_ids=[
            row.annotation_id for row in seeds if row.task_relevant_hint
        ],
    )


_SYSTEM_PROMPT = """You are the PROXY_CHALLENGE operator in a shadow-only scientific reframing lane.

Your task is to test whether a measured observable is being treated as if it were an interchangeable or sufficient proxy for a distinct scientific construct when the grounded evidence supports a falsifiable alternative.

Hard rules:
1. Grounded premise statements are the ONLY positive scientific premises.
2. Proxy-semantic annotations are source-interpretation sidecars. They may help distinguish observables from constructs, but they are NOT positive premises, equivalence authority, or proof that the task actually contains the same proxy relation.
3. Use at least one supplied task-relevant semantic seed, but do not transplant its paper-specific conclusion into another system. Copy every semantic seed annotation_id exactly as supplied; do not abbreviate, reconstruct, or invent IDs.
4. A useful proxy challenge must identify a specific observable, a distinct target construct, and a concrete failure mode. Generic claims that 'all measurements are imperfect' are invalid.
5. The baseline model must state how the observable is implicitly or explicitly being used as a sufficient/interchangeable indicator. The alternative must explain at least two grounded premises without claiming confirmation.
6. Every candidate needs at least one differential prediction where the baseline and alternative make different observable expectations, plus a falsifier and a discriminating test.
7. Prefer tests that measure the challenged observable and the target construct independently or add an orthogonal measurement.
8. Gap statements may motivate the question but are not positive evidence.
9. Do not claim novelty, truth, causality, or scientific authority. Candidates remain hypothesis-only and require verification.
10. If the evidence does not support a genuine proxy challenge, return zero candidates and explain why. Do not fill a quota.
"""


def build_proxy_challenge_prompt(
    *,
    evidence: ScientificReframeEvidencePacket,
    proxy_input: ProxyChallengeInput,
) -> ProxyChallengePrompt:
    payload = {
        "task_id": evidence.task_id,
        "question": evidence.question,
        "grounded_positive_premises": [
            {
                "statement_id": row.statement_id,
                "text": row.text,
                "claim_kind": row.claim_kind,
                "paper_ids": row.paper_ids,
            }
            for row in evidence.premise_statements
        ],
        "research_gaps_nonpremise": [
            {
                "statement_id": row.statement_id,
                "text": row.text,
                "claim_kind": row.claim_kind,
                "paper_ids": row.paper_ids,
            }
            for row in evidence.gap_statements
        ],
        "proxy_semantic_seeds": [
            row.model_dump(mode="json")
            for row in proxy_input.seeds
            if row.task_relevant_hint
        ],
        "semantic_seed_policy": {
            "annotations_are_positive_premises": False,
            "annotations_establish_equivalence": False,
            "seed_relevance_is_diagnostic_only": True,
        },
    }
    return ProxyChallengePrompt(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=(
            "PROXY_CHALLENGE INPUT\n=====================\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nReturn a structured ProxyChallengeBatchDraft. Generate no more than two candidates."
        ),
    )


class ProxyChallengeCompileError(ValueError):
    pass


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            rows.append(text)
    return rows


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    """Return True only when two strings are exactly one edit apart.

    This is deliberately narrower than generic fuzzy matching.  It exists only
    to recover a single-character copy error in an opaque annotation ID emitted
    by the LLM.  Scientific text, measurement identity, and task relevance are
    never fuzzy-matched.
    """

    if left == right:
        return False
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    i = j = edits = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    # The only remaining difference may be one trailing character.
    return True


def _resolve_semantic_seed_ids(
    *,
    requested_ids: list[str],
    proxy_input: ProxyChallengeInput,
) -> tuple[list[str], list[str]]:
    relevant = list(proxy_input.task_relevant_seed_ids)
    relevant_set = set(relevant)
    all_seed_ids = {row.annotation_id for row in proxy_input.seeds}
    resolved: list[str] = []
    normalizations: list[str] = []

    for requested in _dedupe(requested_ids):
        if requested in relevant_set:
            resolved.append(requested)
            continue
        # An exact ID that exists but was deterministically classified as
        # task-irrelevant must never be repaired into a different relevant ID.
        if requested in all_seed_ids:
            raise ProxyChallengeCompileError(
                "candidate references task-irrelevant semantic seed ID: " + requested
            )
        near = [
            candidate
            for candidate in relevant
            if _edit_distance_at_most_one(requested, candidate)
        ]
        if len(near) != 1:
            raise ProxyChallengeCompileError(
                "candidate references unknown or task-irrelevant semantic seed IDs: "
                + requested
            )
        canonical = near[0]
        resolved.append(canonical)
        normalizations.append(
            "normalized one-edit semantic seed ID copy error: "
            f"{requested} -> {canonical}"
        )

    return _dedupe(resolved), normalizations


def compile_proxy_challenge_candidate(
    *,
    evidence: ScientificReframeEvidencePacket,
    proxy_input: ProxyChallengeInput,
    draft: ProxyChallengeDraft,
) -> ScientificProxyChallengeCandidate:
    premise_order = [row.statement_id for row in evidence.premise_statements]
    gap_order = [row.statement_id for row in evidence.gap_statements]
    premise_known = set(premise_order)
    gap_known = set(gap_order)
    seed_ids, seed_normalizations = _resolve_semantic_seed_ids(
        requested_ids=draft.semantic_seed_annotation_ids,
        proxy_input=proxy_input,
    )
    if not seed_ids:
        raise ProxyChallengeCompileError("candidate requires a semantic seed")

    declared_premises = set(_dedupe(draft.premise_statement_ids))
    explained = set(draft.baseline_model.explained_statement_ids) | set(
        draft.alternative_model.explained_statement_ids
    )
    unknown_premises = sorted((declared_premises | explained) - premise_known)
    if unknown_premises:
        raise ProxyChallengeCompileError(
            "candidate references non-grounded premise IDs: "
            + ", ".join(unknown_premises)
        )
    selected_premises = [
        statement_id
        for statement_id in premise_order
        if statement_id in (declared_premises | explained)
    ]
    if len(selected_premises) < 2:
        raise ProxyChallengeCompileError(
            "PROXY_CHALLENGE requires at least two grounded premises"
        )

    declared_gaps = set(_dedupe(draft.gap_statement_ids))
    unknown_gaps = sorted(declared_gaps - gap_known)
    if unknown_gaps:
        raise ProxyChallengeCompileError(
            "candidate references non-grounded gap IDs: " + ", ".join(unknown_gaps)
        )
    selected_gaps = [row for row in gap_order if row in declared_gaps]

    normalizations: list[str] = list(seed_normalizations)

    def normalize_model(model: ScientificModelDraft, name: str) -> ScientificModelDraft:
        ids = set(_dedupe(model.explained_statement_ids))
        canonical = [row for row in premise_order if row in ids]
        if not canonical:
            raise ProxyChallengeCompileError(f"{name} must explain a grounded premise")
        if not model.expected_observations:
            raise ProxyChallengeCompileError(f"{name} requires expected observations")
        if canonical != list(model.explained_statement_ids):
            normalizations.append(f"canonicalized {name}.explained_statement_ids")
        return model.model_copy(update={"explained_statement_ids": canonical})

    baseline = normalize_model(draft.baseline_model, "baseline_model")
    alternative = normalize_model(draft.alternative_model, "alternative_model")
    if len(set(alternative.explained_statement_ids)) < 2:
        raise ProxyChallengeCompileError(
            "PROXY_CHALLENGE alternative model must explain at least two premises"
        )
    if baseline.summary.strip() == alternative.summary.strip():
        raise ProxyChallengeCompileError("baseline and alternative summaries are identical")

    predictions = [
        row
        for row in draft.differential_predictions
        if row.baseline_expectation.strip() != row.alternative_expectation.strip()
    ]
    if not predictions:
        raise ProxyChallengeCompileError("candidate requires a differential prediction")
    if len({row.local_id for row in predictions}) != len(predictions):
        raise ProxyChallengeCompileError("duplicate differential prediction local_id")
    if not draft.falsifiers:
        raise ProxyChallengeCompileError("candidate requires a falsifier")
    if len({row.local_id for row in draft.falsifiers}) != len(draft.falsifiers):
        raise ProxyChallengeCompileError("duplicate falsifier local_id")
    if not draft.discriminating_test.primary_observables:
        raise ProxyChallengeCompileError(
            "discriminating test requires primary observables"
        )

    normalized = draft.model_copy(
        update={
            "semantic_seed_annotation_ids": seed_ids,
            "premise_statement_ids": selected_premises,
            "gap_statement_ids": selected_gaps,
            "baseline_model": baseline,
            "alternative_model": alternative,
            "differential_predictions": predictions,
        }
    )
    payload = json.dumps(
        normalized.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    candidate_id = "scientific_proxy_challenge:" + hashlib.sha256(
        (
            evidence.task_id
            + "|"
            + evidence.source_context_sha256
            + "|"
            + proxy_input.source_annotation_sha256
            + "|"
            + payload
        ).encode("utf-8")
    ).hexdigest()[:20]
    return ScientificProxyChallengeCandidate(
        candidate_id=candidate_id,
        source_task_id=evidence.task_id,
        source_context_id=evidence.source_context_id,
        source_context_sha256=evidence.source_context_sha256,
        source_enrichment_report_id=proxy_input.source_enrichment_report_id,
        source_annotation_sha256=proxy_input.source_annotation_sha256,
        title=normalized.title,
        semantic_seed_annotation_ids=list(normalized.semantic_seed_annotation_ids),
        premise_statement_ids=list(normalized.premise_statement_ids),
        gap_statement_ids=list(normalized.gap_statement_ids),
        baseline_model=normalized.baseline_model,
        alternative_model=normalized.alternative_model,
        challenged_proxy_assumption=normalized.challenged_proxy_assumption,
        challenged_observable=normalized.challenged_observable,
        target_construct=normalized.target_construct,
        proxy_failure_mode=normalized.proxy_failure_mode,
        differential_predictions=list(normalized.differential_predictions),
        falsifiers=list(normalized.falsifiers),
        discriminating_test=normalized.discriminating_test,
        unresolved_questions=list(normalized.unresolved_questions),
        provenance_normalizations=normalizations,
    )


class ProxyChallengeShadowRuntime:
    def __init__(self, backend: ProxyChallengeBackend) -> None:
        self.backend = backend

    def run(
        self,
        *,
        evidence: ScientificReframeEvidencePacket,
        proxy_input: ProxyChallengeInput,
    ) -> tuple[ProxyChallengeRunReport, ProxyChallengePrompt | None]:
        if proxy_input.task_id != evidence.task_id:
            raise ValueError("proxy input task_id must equal evidence task_id")
        if proxy_input.source_context_id != evidence.source_context_id:
            raise ValueError("proxy input context_id must equal evidence context_id")
        if proxy_input.source_context_sha256 != evidence.source_context_sha256:
            raise ValueError("proxy input context sha must equal evidence context sha")

        if not proxy_input.task_relevant_seed_ids:
            report = self._report(
                evidence=evidence,
                proxy_input=proxy_input,
                decision="skipped_no_task_relevant_seed",
                candidates=[],
                abstention_reason=(
                    "No proxy-semantic annotation had deterministic task-term overlap; "
                    "missing relevance is not negative scientific evidence."
                ),
                llm_calls=0,
            )
            return report, None

        prompt = build_proxy_challenge_prompt(evidence=evidence, proxy_input=proxy_input)
        try:
            generation = self.backend.generate(prompt)
        except Exception as exc:
            report = self._report(
                evidence=evidence,
                proxy_input=proxy_input,
                decision="generation_failed",
                candidates=[],
                abstention_reason="Structured PROXY_CHALLENGE generation failed.",
                compile_issues=[f"{type(exc).__name__}: {exc}"],
                llm_calls=1,
            )
            return report, prompt

        rows = list(generation.draft.candidates)
        rejected = 0
        issues: list[str] = []
        if len(rows) > 2:
            rejected += len(rows) - 2
            issues.append(f"ignored {len(rows) - 2} candidate(s) beyond quota of 2")
            rows = rows[:2]
        candidates: list[ScientificProxyChallengeCandidate] = []
        for row in rows:
            try:
                candidates.append(
                    compile_proxy_challenge_candidate(
                        evidence=evidence,
                        proxy_input=proxy_input,
                        draft=row,
                    )
                )
            except ProxyChallengeCompileError as exc:
                rejected += 1
                issues.append(f"candidate {row.local_id}: {exc}")

        if candidates:
            decision = "generated"
            abstention_reason = None
        elif generation.draft.candidates:
            decision = "rejected_invalid_draft"
            abstention_reason = "All generated candidates failed proxy challenge compilation."
        else:
            decision = "abstained"
            abstention_reason = (
                generation.draft.abstention_reason
                or "Model returned no proxy challenge candidates."
            )

        report = self._report(
            evidence=evidence,
            proxy_input=proxy_input,
            decision=decision,
            candidates=candidates,
            abstention_reason=abstention_reason,
            compile_issues=issues,
            rejected_candidate_count=rejected,
            llm_calls=1,
            generation=generation,
        )
        return report, prompt

    def _report(
        self,
        *,
        evidence: ScientificReframeEvidencePacket,
        proxy_input: ProxyChallengeInput,
        decision: str,
        candidates: list[ScientificProxyChallengeCandidate],
        abstention_reason: str | None,
        compile_issues: list[str] | None = None,
        rejected_candidate_count: int = 0,
        llm_calls: int,
        generation: ProxyChallengeGeneration | None = None,
    ) -> ProxyChallengeRunReport:
        seed = json.dumps(
            [
                evidence.task_id,
                evidence.source_context_sha256,
                proxy_input.source_enrichment_report_id,
                proxy_input.source_annotation_sha256,
                decision,
                *[row.candidate_id for row in candidates],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return ProxyChallengeRunReport(
            report_id="scientific_proxy_challenge_shadow:"
            + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20],
            source_task_id=evidence.task_id,
            source_context_id=evidence.source_context_id,
            source_context_sha256=evidence.source_context_sha256,
            source_enrichment_report_id=proxy_input.source_enrichment_report_id,
            source_annotation_sha256=proxy_input.source_annotation_sha256,
            backend_name=self.backend.backend_name,
            model_name=self.backend.model_name,
            decision=decision,  # type: ignore[arg-type]
            candidate_ids=[row.candidate_id for row in candidates],
            candidates=candidates,
            task_relevant_seed_ids=list(proxy_input.task_relevant_seed_ids),
            abstention_reason=abstention_reason,
            compile_issues=list(compile_issues or []),
            rejected_candidate_count=rejected_candidate_count,
            llm_calls_performed=llm_calls,
            input_tokens=(generation.input_tokens if generation else None),
            output_tokens=(generation.output_tokens if generation else None),
            response_id=(generation.response_id if generation else None),
            elapsed_seconds=(generation.elapsed_seconds if generation else None),
        )
