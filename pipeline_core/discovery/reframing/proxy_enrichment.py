from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.semantic_ir.annotation import SemanticAnnotation
from pipeline_core.corpus.semantic_ir.backfill import BackfillPlan
from pipeline_core.corpus.semantic_ir.schema import (
    SemanticIRBundle,
    SemanticIRRecord,
    SemanticObjectRef,
)


EXTRACTOR_VERSION = "proxy-semantic-backfill-v1.2"
PROMPT_VERSION = "proxy-semantic-backfill-prompt-v1.2"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ProxyRole = Literal[
    "explicit_proxy",
    "implicit_proxy_candidate",
    "surrogate_for_construct",
    "proxy_limitation",
    "observable_construct_distinction",
]

InterchangeabilityStatus = Literal[
    "explicitly_equated",
    "implicitly_treated_as_interchangeable",
    "context_limited",
    "explicitly_distinguished",
    "unclear",
]

EvidenceBasis = Literal[
    "explicit_source_statement",
    "source_interpretation",
]

SupportKind = Literal[
    "explicit_proxy_language",
    "explicit_inference_to_distinct_construct",
    "explicit_non_equivalence_or_limitation",
    "explicit_interchangeability_or_substitution",
]


class ProxySemanticRelationDraft(StrictModel):
    measurement_id: str = Field(min_length=1)
    proxy_role: ProxyRole
    target_construct: str = Field(min_length=1)
    interchangeability: InterchangeabilityStatus
    interpretation_summary: str = Field(min_length=1)
    distinct_target_rationale: str = Field(min_length=1)
    support_kind: SupportKind
    support_quotes: list[str] = Field(min_length=1, max_length=3)
    limitations: list[str] = Field(default_factory=list)
    context_dependencies: list[str] = Field(default_factory=list)
    evidence_basis: EvidenceBasis


class ProxySemanticChunkDraft(StrictModel):
    schema_version: Literal[
        "proxy-semantic-chunk-draft-v1.1"
    ] = "proxy-semantic-chunk-draft-v1.1"
    items: list[ProxySemanticRelationDraft] = Field(default_factory=list)


class ProxySemanticAnnotationValue(StrictModel):
    schema_version: Literal[
        "proxy-semantic-annotation-value-v1.1"
    ] = "proxy-semantic-annotation-value-v1.1"
    measurement_id: str
    metric: str | None = None
    subject_id: str | None = None
    source_expression: str | None = None
    proxy_role: ProxyRole
    target_construct: str
    interchangeability: InterchangeabilityStatus
    interpretation_summary: str
    distinct_target_rationale: str
    support_kind: SupportKind
    support_quotes: list[str] = Field(min_length=1, max_length=3)
    limitations: list[str] = Field(default_factory=list)
    context_dependencies: list[str] = Field(default_factory=list)
    evidence_basis: EvidenceBasis

    hypothesis_generation_authority: Literal[False] = False
    negative_evidence_authority: Literal[False] = False
    scientific_equivalence_authority: Literal[False] = False


class ProxySemanticCandidateRejection(StrictModel):
    item_index: int = Field(ge=0)
    measurement_id: str = Field(min_length=1)
    proxy_role: ProxyRole
    target_construct: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)

    candidate_rejected_not_negative_evidence: Literal[True] = True
    candidate_rejected_not_scientific_authority: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False


class ProxySemanticChunkReview(StrictModel):
    schema_version: Literal[
        "proxy-semantic-chunk-review-v1"
    ] = "proxy-semantic-chunk-review-v1"
    review_id: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    source_chunk_ref: SemanticObjectRef
    source_chunk_sha256: str = Field(min_length=64, max_length=64)
    measurement_ids: list[str] = Field(min_length=1)
    annotation_ids: list[str] = Field(default_factory=list)
    emitted_annotation_count: int = Field(ge=0)
    draft_candidate_count: int | None = Field(default=None, ge=0)
    rejected_candidate_count: int = Field(default=0, ge=0)
    rejected_candidates: list[ProxySemanticCandidateRejection] = Field(default_factory=list)
    extractor_version: Literal[EXTRACTOR_VERSION] = EXTRACTOR_VERSION
    prompt_version: Literal[PROMPT_VERSION] = PROMPT_VERSION
    backend_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    provider_input_tokens: int | None = Field(default=None, ge=0)
    provider_output_tokens: int | None = Field(default=None, ge=0)
    response_id: str | None = None

    completed: Literal[True] = True
    source_interpretation_only: Literal[True] = True
    absence_of_annotation_is_not_negative_evidence: Literal[True] = True
    candidate_rejection_is_not_negative_evidence: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    selection_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ProxySemanticChunkReview":
        if self.source_chunk_ref.object_kind != "chunk":
            raise ValueError("proxy enrichment reviews must reference source chunks")
        if self.emitted_annotation_count != len(self.annotation_ids):
            raise ValueError("emitted_annotation_count must equal len(annotation_ids)")
        if self.rejected_candidate_count != len(self.rejected_candidates):
            raise ValueError("rejected_candidate_count must equal len(rejected_candidates)")
        if self.draft_candidate_count is not None and self.draft_candidate_count != (
            self.emitted_annotation_count + self.rejected_candidate_count
        ):
            raise ValueError(
                "draft_candidate_count must equal emitted + rejected candidates"
            )
        if len(self.measurement_ids) != len(set(self.measurement_ids)):
            raise ValueError("measurement_ids must be unique")
        if len(self.annotation_ids) != len(set(self.annotation_ids)):
            raise ValueError("annotation_ids must be unique")
        return self


class ProxySemanticEnrichmentError(StrictModel):
    source_chunk_ref: SemanticObjectRef
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)


class ProxySemanticEnrichmentReport(StrictModel):
    schema_version: Literal[
        "proxy-semantic-enrichment-report-v1"
    ] = "proxy-semantic-enrichment-report-v1"
    report_id: str = Field(min_length=1)
    scope_id: str = Field(min_length=1)
    operator_id: Literal["PROXY_CHALLENGE"] = "PROXY_CHALLENGE"
    source_plan_id: str = Field(min_length=1)
    extractor_version: Literal[EXTRACTOR_VERSION] = EXTRACTOR_VERSION
    prompt_version: Literal[PROMPT_VERSION] = PROMPT_VERSION
    annotation_path: str
    review_path: str
    plan_target_count: int = Field(ge=0)
    considered_target_count: int = Field(ge=0)
    completed_target_count: int = Field(ge=0)
    resumed_target_count: int = Field(ge=0)
    new_target_count: int = Field(ge=0)
    pending_target_count: int = Field(ge=0)
    failed_target_count: int = Field(ge=0)
    new_annotation_count: int = Field(ge=0)
    total_annotation_count: int = Field(ge=0)
    new_rejected_candidate_count: int = Field(default=0, ge=0)
    total_rejected_candidate_count: int = Field(default=0, ge=0)
    llm_calls_performed: int = Field(ge=0)
    max_new_calls: int | None = Field(default=None, ge=0)
    errors: list[ProxySemanticEnrichmentError] = Field(default_factory=list)

    coverage_complete_for_plan: bool
    execution_performed: bool
    append_only_sidecars: Literal[True] = True
    absence_of_annotation_is_not_negative_evidence: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False
    production_selection_changed: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    proxy_semantics_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ProxySemanticEnrichmentReport":
        if self.failed_target_count != len(self.errors):
            raise ValueError("failed_target_count must equal len(errors)")
        if self.new_target_count > self.llm_calls_performed:
            raise ValueError(
                "each newly completed target requires an LLM call"
            )
        if self.completed_target_count != self.resumed_target_count + self.new_target_count:
            raise ValueError(
                "completed_target_count must equal resumed + new targets"
            )
        if self.pending_target_count != self.plan_target_count - self.completed_target_count:
            raise ValueError(
                "pending_target_count must equal plan targets minus completed targets"
            )
        if self.coverage_complete_for_plan != (
            self.completed_target_count == self.plan_target_count
            and self.failed_target_count == 0
        ):
            raise ValueError("coverage_complete_for_plan is inconsistent with counts")
        return self


@dataclass(frozen=True)
class ProxySemanticPrompt:
    source_chunk_ref: SemanticObjectRef
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class ProxyDraftGeneration:
    draft: ProxySemanticChunkDraft
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None


@runtime_checkable
class ProxySemanticBackend(Protocol):
    backend_name: str
    model_name: str

    def generate(self, prompt: ProxySemanticPrompt) -> ProxyDraftGeneration: ...


class InstructorOpenAICompatibleProxyBackend:
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
                "Proxy semantic enrichment requires installed 'openai' and 'instructor' packages."
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

    def generate(self, prompt: ProxySemanticPrompt) -> ProxyDraftGeneration:
        from pipeline_core.llm.llm_telemetry import run_instructor_structured_call

        client = self._get_client()
        draft, event = run_instructor_structured_call(
            client.chat.completions,
            model=self.model_name,
            response_model=ProxySemanticChunkDraft,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            temperature=self.temperature,
            max_retries=self.parse_retries,
            telemetry_path=self.telemetry_path,
            telemetry_context={
                **self.telemetry_context,
                "pipeline": "semantic_enrichment",
                "stage": "proxy_semantics",
                "call_kind": "proxy_semantic_backfill",
                "paper_id": prompt.source_chunk_ref.paper_id,
                "chunk_id": prompt.source_chunk_ref.chunk_id,
            },
            semantic_components={
                "capability": "proxy_semantics",
                "source_chunk_id": prompt.source_chunk_ref.object_id,
            },
        )
        if not isinstance(draft, ProxySemanticChunkDraft):
            draft = ProxySemanticChunkDraft.model_validate(draft)
        return ProxyDraftGeneration(
            draft=draft,
            input_tokens=event.provider_input_tokens,
            output_tokens=event.provider_output_tokens,
            response_id=event.response_id,
        )


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _chunk_key(ref: SemanticObjectRef) -> tuple[str, str | None, str, str]:
    if ref.object_kind != "chunk":
        raise ValueError("proxy enrichment targets must reference source chunks")
    return ref.identity_key()


def _bundle_for_chunk(
    *,
    bundles: dict[str, SemanticIRBundle],
    source_chunk_ref: SemanticObjectRef,
) -> SemanticIRBundle:
    candidates = [
        bundle
        for bundle in bundles.values()
        if bundle.paper_id == source_chunk_ref.paper_id
        and any(
            source.identity_key() == source_chunk_ref.identity_key()
            for source in bundle.source_chunks
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one semantic IR bundle for source chunk "
            + str(source_chunk_ref.identity_key())
        )
    return candidates[0]


def load_source_chunk_payload(
    *,
    bundle: SemanticIRBundle,
    source_chunk_ref: SemanticObjectRef,
) -> tuple[dict[str, Any], str]:
    if source_chunk_ref.object_kind != "chunk":
        raise ValueError("source_chunk_ref must reference a chunk")
    if not source_chunk_ref.source_path:
        raise ValueError("source chunk ref has no persisted source_path")
    if not bundle.source_attempt_directory:
        raise ValueError("semantic IR bundle has no source_attempt_directory")

    recorded = Path(source_chunk_ref.source_path)
    path = (
        recorded
        if recorded.is_absolute()
        else Path(bundle.source_attempt_directory) / recorded
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    if str(payload.get("paper_id", "")) != source_chunk_ref.paper_id:
        raise ValueError("source chunk paper_id does not match semantic reference")
    if str(payload.get("chunk_id", "")) != str(source_chunk_ref.chunk_id):
        raise ValueError("source chunk chunk_id does not match semantic reference")
    return payload, _sha256_bytes(raw)


def measurement_records_for_chunk(
    *,
    bundle: SemanticIRBundle,
    source_chunk_ref: SemanticObjectRef,
) -> list[SemanticIRRecord]:
    key = source_chunk_ref.identity_key()
    rows = [
        record
        for record in bundle.records
        if record.ref.object_kind == "measurement"
        and record.source_chunk_ref is not None
        and record.source_chunk_ref.identity_key() == key
    ]
    by_id: dict[str, SemanticIRRecord] = {}
    for row in rows:
        if row.ref.object_id in by_id:
            raise ValueError(
                f"duplicate measurement id in source chunk: {row.ref.object_id}"
            )
        by_id[row.ref.object_id] = row
    return [by_id[key] for key in sorted(by_id)]


_LIMITATION_DISCOVERY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bdifficult\s+to\s+determine\b",
        r"\b(?:can|could|does|do|did)\s+not\s+(?:fully\s+)?determine\b",
        r"\bnot\s+sufficient\b",
        r"\binsufficient\b",
        r"\bdiverg(?:e|es|ed|ing)\b",
        r"\bdiffer(?:s|ed|ing)?\s+from\b",
        r"\bmismatch\b",
        r"\bdiscrepanc(?:y|ies)\b",
        r"\bmay\s+not\s+represent\b",
        r"\bdoes\s+not\s+represent\b",
        r"\bnot\s+equivalent\b",
        r"\bcannot\s+be\s+equated\b",
        r"\bdistinct\s+from\b",
        r"\blimit(?:ation|ed|s)?\b",
    )
)


def _limitation_candidate_hints(source_payload: Mapping[str, Any]) -> list[str]:
    """Surface explicit non-equivalence wording without creating authority.

    These strings are attention hints only. They are copied from the persisted
    source surface so the LLM does not overlook a limitation sentence buried in
    a measurement-heavy chunk. The strict compiler still requires the emitted
    support quote itself to be an exact source substring and to substantiate the
    declared support kind.
    """

    parts: list[str] = []
    for key in ("left_context", "core_text", "right_context"):
        value = source_payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if not text:
        return []
    sentences = [row.strip() for row in re.split(r"(?<=[.!?])\s+", text) if row.strip()]
    hints: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        if not any(pattern.search(sentence) for pattern in _LIMITATION_DISCOVERY_PATTERNS):
            continue
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        hints.append(sentence)
        if len(hints) >= 8:
            break
    return hints


def build_proxy_semantic_prompt(
    *,
    source_chunk_ref: SemanticObjectRef,
    source_payload: dict[str, Any],
    measurements: list[SemanticIRRecord],
) -> ProxySemanticPrompt:
    if not measurements:
        raise ValueError("proxy enrichment requires at least one measurement")

    measurement_rows = []
    for record in measurements:
        payload = record.payload
        measurement_rows.append(
            {
                "measurement_id": record.ref.object_id,
                "metric": payload.get("metric"),
                "metric_id": payload.get("metric_id"),
                "subject_id": payload.get("subject_id"),
                "source_expression": payload.get("source_expression"),
                "value": payload.get("value"),
                "unit": payload.get("unit"),
                "conditions": payload.get("conditions"),
            }
        )

    source_surface = {
        "paper_id": source_payload.get("paper_id"),
        "chunk_id": source_payload.get("chunk_id"),
        "section": source_payload.get("section"),
        "left_context": source_payload.get("left_context"),
        "core_text": source_payload.get("core_text"),
        "right_context": source_payload.get("right_context"),
        "asset_context": source_payload.get("asset_context"),
    }
    limitation_hints = _limitation_candidate_hints(source_payload)

    system_prompt = f"""You extract task-local scientific proxy semantics from one persisted source chunk.

This is {PROMPT_VERSION}. Treat the supplied source text as the only scientific authority.

A proxy relation exists only when an observable/measurement is used to infer, stand in for, operationalize, or approximate a scientifically distinct target construct, or when the source explicitly/interpretively exposes a limitation or distinction between the observable and that construct.

Rules:
- Use ONLY measurement_id values supplied in MEASUREMENTS.
- Do not invent measurements, claims, mechanisms, targets, numerical values, or missing context.
- Do not emit an item merely because a measurement correlates with, affects, predicts, quantifies, or summarizes an outcome.
- Do not emit an item just because two measurements are reported together.
- Do not emit tautological operationalizations such as enhancement factor -> enhancement performance, R^2 -> linearity, or LSPR wavelength -> generic plasmonic properties unless the source itself states a scientifically consequential stand-in, inference, substitution, or non-equivalence.
- A study-specific analyte, range, or measurement condition alone is NOT enough to make an otherwise tautological metric a context-limited proxy.
- Every emitted item MUST cite 1-3 short exact support_quotes copied from SOURCE_CHUNK. The quoted wording must itself support the declared support_kind; generic measurement reporting is insufficient.
- distinct_target_rationale must explain why the target construct is scientifically distinct from the measured metric, not merely a broader synonym.
- Use support_kind='explicit_proxy_language' only when the source explicitly uses proxy/surrogate/indicator/marker/representative language.
- Use support_kind='explicit_inference_to_distinct_construct' only when the source explicitly uses the observable to indicate/reveal/infer a different construct.
- Use support_kind='explicit_non_equivalence_or_limitation' only when the source explicitly states that the observable cannot fully determine, differs from, or is limited as a representation of the construct.
- Use support_kind='explicit_interchangeability_or_substitution' only when the source explicitly equates, substitutes, or treats observables/constructs as interchangeable.
- Do not label absence of a proxy relation. If the source does not support one of those four source-grounded relations, emit no item for that measurement.
- 'implicit_proxy_candidate' still requires explicit source wording that licenses an inference/substitution to a distinct construct; it is not permission to invent a proxy interpretation.
- 'proxy_limitation' or 'observable_construct_distinction' must preserve explicit source-supported non-equivalence or limitation.
- Before returning an empty item list, inspect every LIMITATION_CANDIDATE_HINT. If a hint explicitly distinguishes a supplied measurement/observable from a scientifically distinct target quantity or says that the target cannot be fully determined from the observed quantity, emit a conservative proxy_limitation or observable_construct_distinction item. Do not emit merely because a limitation cue exists.
- LIMITATION_CANDIDATE_HINTS are attention aids copied from the source, not prevalidated proxy relations and not scientific authority. The emitted item still needs an exact support quote and a distinct target construct.
- Keep limitations and context dependencies source-grounded. Empty lists are allowed.
- Output is a source interpretation sidecar, never canonical evidence, scientific equivalence authority, novelty authority, or hypothesis-selection authority.
"""
    user_prompt = (
        "SOURCE_CHUNK\n============\n"
        + _canonical_json(source_surface)
        + "\n\nMEASUREMENTS\n============\n"
        + _canonical_json(measurement_rows)
        + "\n\nLIMITATION_CANDIDATE_HINTS\n==========================\n"
        + _canonical_json(limitation_hints)
        + "\n\nReturn ProxySemanticChunkDraft JSON. It is valid to return {\"items\": []}.\n"
    )
    return ProxySemanticPrompt(
        source_chunk_ref=source_chunk_ref,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _annotation_id(
    *,
    scope_id: str,
    measurement_ref: SemanticObjectRef,
    item: ProxySemanticRelationDraft,
) -> str:
    return _stable_id(
        "semantic_annotation",
        scope_id,
        EXTRACTOR_VERSION,
        *measurement_ref.identity_key(),
        item.proxy_role,
        item.interchangeability,
        item.target_construct.strip().lower(),
    )


_SUPPORT_CUE_PATTERNS: dict[SupportKind, tuple[re.Pattern[str], ...]] = {
    "explicit_proxy_language": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\bproxy\b",
            r"\bsurrogate\b",
            r"\bindicator\b",
            r"\bmarker\b",
            r"\brepresent(?:s|ed|ing|ative)?\b",
            r"\bstand(?:s|ing)?\s+in\s+for\b",
        )
    ),
    "explicit_inference_to_distinct_construct": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\bindicat(?:e|es|ed|ing)\b",
            r"\bsuggest(?:s|ed|ing)?\b",
            r"\breveal(?:s|ed|ing)?\b",
            r"\bimply(?:ies|ied|ing)?\b",
            r"\breflect(?:s|ed|ing)?\b",
            r"\bused\s+to\s+(?:infer|estimate|assess|evaluate)\b",
        )
    ),
    "explicit_non_equivalence_or_limitation": _LIMITATION_DISCOVERY_PATTERNS,
    "explicit_interchangeability_or_substitution": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\binterchangeable\b",
            r"\bequivalent\b",
            r"\bequat(?:e|ed|es|ing)\b",
            r"\bsubstitut(?:e|ed|es|ing|ion)\b",
            r"\bstand(?:s|ing)?\s+in\s+for\b",
            r"\bproxy\s+for\b",
            r"\bsurrogate\s+for\b",
        )
    ),
}


def _normalized_source_text(source_payload: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("left_context", "core_text", "right_context"):
        value = source_payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    asset_context = source_payload.get("asset_context")
    if asset_context:
        parts.append(_canonical_json(asset_context))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _validate_support_quotes(
    *,
    item: ProxySemanticRelationDraft,
    source_payload: Mapping[str, Any],
) -> None:
    source_text = _normalized_source_text(source_payload)
    source_folded = source_text.casefold()
    exact_quotes: list[str] = []
    for raw in item.support_quotes:
        quote = re.sub(r"\s+", " ", raw).strip()
        if not quote:
            raise ValueError("proxy semantic support quote must be non-empty")
        if quote.casefold() not in source_folded:
            raise ValueError(
                "proxy semantic support quote is not an exact source substring after whitespace normalization: "
                + quote
            )
        exact_quotes.append(quote)

    patterns = _SUPPORT_CUE_PATTERNS[item.support_kind]
    if not any(pattern.search(quote) for quote in exact_quotes for pattern in patterns):
        raise ValueError(
            "proxy semantic support quote does not substantiate support_kind "
            + item.support_kind
        )

    if item.proxy_role in {"proxy_limitation", "observable_construct_distinction"} and (
        item.support_kind != "explicit_non_equivalence_or_limitation"
    ):
        raise ValueError(
            f"{item.proxy_role} requires explicit_non_equivalence_or_limitation support"
        )
    if item.proxy_role == "implicit_proxy_candidate" and item.support_kind not in {
        "explicit_inference_to_distinct_construct",
        "explicit_interchangeability_or_substitution",
    }:
        raise ValueError(
            "implicit_proxy_candidate requires explicit inference or substitution support"
        )
    if item.proxy_role in {"explicit_proxy", "surrogate_for_construct"} and item.support_kind not in {
        "explicit_proxy_language",
        "explicit_inference_to_distinct_construct",
        "explicit_interchangeability_or_substitution",
    }:
        raise ValueError(
            f"{item.proxy_role} requires explicit stand-in, inference, or substitution support"
        )


def compile_proxy_semantic_annotations(
    *,
    scope_id: str,
    source_chunk_ref: SemanticObjectRef,
    source_payload: Mapping[str, Any],
    measurements: list[SemanticIRRecord],
    draft: ProxySemanticChunkDraft,
) -> list[SemanticAnnotation]:
    measurement_by_id = {row.ref.object_id: row for row in measurements}
    annotations: list[SemanticAnnotation] = []
    seen_relation_keys: set[tuple[str, str, str, str]] = set()

    for item in draft.items:
        record = measurement_by_id.get(item.measurement_id)
        if record is None:
            raise ValueError(
                "proxy semantic draft referenced unknown measurement_id: "
                + item.measurement_id
            )
        _validate_support_quotes(item=item, source_payload=source_payload)
        relation_key = (
            item.measurement_id,
            item.proxy_role,
            item.interchangeability,
            item.target_construct.strip().lower(),
        )
        if relation_key in seen_relation_keys:
            raise ValueError("proxy semantic draft contains a duplicate relation")
        seen_relation_keys.add(relation_key)

        payload = record.payload
        value = ProxySemanticAnnotationValue(
            measurement_id=record.ref.object_id,
            metric=(str(payload["metric"]) if payload.get("metric") is not None else None),
            subject_id=(
                str(payload["subject_id"])
                if payload.get("subject_id") is not None
                else None
            ),
            source_expression=(
                str(payload["source_expression"])
                if payload.get("source_expression") is not None
                else None
            ),
            proxy_role=item.proxy_role,
            target_construct=item.target_construct,
            interchangeability=item.interchangeability,
            interpretation_summary=item.interpretation_summary,
            distinct_target_rationale=item.distinct_target_rationale,
            support_kind=item.support_kind,
            support_quotes=item.support_quotes,
            limitations=item.limitations,
            context_dependencies=item.context_dependencies,
            evidence_basis=item.evidence_basis,
        )
        annotations.append(
            SemanticAnnotation(
                annotation_id=_annotation_id(
                    scope_id=scope_id,
                    measurement_ref=record.ref,
                    item=item,
                ),
                capability="proxy_semantics",
                subject_ref=record.ref,
                value=value.model_dump(mode="json"),
                source_refs=[source_chunk_ref],
                extractor_version=EXTRACTOR_VERSION,
                epistemic_status="source_interpretation",
            )
        )

    annotations.sort(key=lambda row: row.annotation_id)
    return annotations


def compile_proxy_semantic_annotations_isolated(
    *,
    scope_id: str,
    source_chunk_ref: SemanticObjectRef,
    source_payload: Mapping[str, Any],
    measurements: list[SemanticIRRecord],
    draft: ProxySemanticChunkDraft,
) -> tuple[list[SemanticAnnotation], list[ProxySemanticCandidateRejection]]:
    """Compile candidates independently without weakening the strict gate.

    A malformed or semantically unsupported candidate is diagnostic output, not
    a chunk-level execution failure.  The strict compiler remains the authority
    for each individual candidate; this wrapper merely isolates failures so a
    bad sibling candidate cannot discard valid annotations from the same LLM
    response.
    """

    accepted: list[SemanticAnnotation] = []
    rejected: list[ProxySemanticCandidateRejection] = []
    seen_annotation_ids: set[str] = set()
    for item_index, item in enumerate(draft.items):
        try:
            single = ProxySemanticChunkDraft(items=[item])
            compiled = compile_proxy_semantic_annotations(
                scope_id=scope_id,
                source_chunk_ref=source_chunk_ref,
                source_payload=source_payload,
                measurements=measurements,
                draft=single,
            )
            if len(compiled) != 1:
                raise ValueError(
                    "isolated proxy semantic candidate must compile to exactly one annotation"
                )
            annotation = compiled[0]
            if annotation.annotation_id in seen_annotation_ids:
                raise ValueError("proxy semantic draft contains a duplicate relation")
            seen_annotation_ids.add(annotation.annotation_id)
            accepted.append(annotation)
        except Exception as exc:
            rejected.append(
                ProxySemanticCandidateRejection(
                    item_index=item_index,
                    measurement_id=item.measurement_id,
                    proxy_role=item.proxy_role,
                    target_construct=item.target_construct,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    accepted.sort(key=lambda row: row.annotation_id)
    return accepted, rejected


def _read_jsonl_models(path: Path, model: type[BaseModel], id_field: str) -> dict[str, BaseModel]:
    rows: dict[str, BaseModel] = {}
    if not path.is_file():
        return rows
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
            row = model.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"invalid JSONL row {path}:{line_number}: {exc}") from exc
        row_id = str(getattr(row, id_field))
        previous = rows.get(row_id)
        if previous is not None and previous.model_dump(mode="json") != row.model_dump(mode="json"):
            raise ValueError(f"conflicting duplicate {id_field}={row_id} in {path}")
        rows[row_id] = row
    return rows


def _append_unique_models(
    *,
    path: Path,
    rows: list[BaseModel],
    id_field: str,
    existing: dict[str, BaseModel],
) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            row_id = str(getattr(row, id_field))
            previous = existing.get(row_id)
            if previous is not None:
                if previous.model_dump(mode="json") != row.model_dump(mode="json"):
                    raise ValueError(
                        f"existing sidecar conflicts with {id_field}={row_id}"
                    )
                continue
            handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
            handle.flush()
            existing[row_id] = row
            appended += 1
    return appended


def _review_id(*, scope_id: str, plan_id: str, source_chunk_ref: SemanticObjectRef) -> str:
    return _stable_id(
        "proxy_semantic_review",
        scope_id,
        plan_id,
        EXTRACTOR_VERSION,
        *source_chunk_ref.identity_key(),
    )


def execute_proxy_semantic_enrichment(
    *,
    scope_id: str,
    plan: BackfillPlan,
    bundles: dict[str, SemanticIRBundle],
    backend: ProxySemanticBackend,
    annotation_path: str | Path,
    review_path: str | Path,
    max_new_calls: int | None = None,
) -> ProxySemanticEnrichmentReport:
    if max_new_calls is not None and max_new_calls < 0:
        raise ValueError("max_new_calls must be >= 0")
    for target in plan.targets:
        if set(target.missing_capabilities) != {"proxy_semantics"}:
            raise ValueError(
                "proxy enrichment executor may consume only pure proxy_semantics targets"
            )

    annotation_file = Path(annotation_path)
    review_file = Path(review_path)
    existing_annotations = _read_jsonl_models(
        annotation_file,
        SemanticAnnotation,
        "annotation_id",
    )
    existing_reviews = _read_jsonl_models(
        review_file,
        ProxySemanticChunkReview,
        "review_id",
    )

    resumed = 0
    successful_new = 0
    llm_calls = 0
    new_annotations = 0
    new_rejected_candidates = 0
    errors: list[ProxySemanticEnrichmentError] = []
    considered = 0

    for target in plan.targets:
        review_id = _review_id(
            scope_id=scope_id,
            plan_id=plan.plan_id,
            source_chunk_ref=target.source_chunk_ref,
        )
        existing_review = existing_reviews.get(review_id)
        if existing_review is None and max_new_calls is not None and llm_calls >= max_new_calls:
            continue

        considered += 1
        try:
            bundle = _bundle_for_chunk(
                bundles=bundles,
                source_chunk_ref=target.source_chunk_ref,
            )
            source_payload, source_sha256 = load_source_chunk_payload(
                bundle=bundle,
                source_chunk_ref=target.source_chunk_ref,
            )
            measurements = measurement_records_for_chunk(
                bundle=bundle,
                source_chunk_ref=target.source_chunk_ref,
            )
            if not measurements:
                raise ValueError(
                    "proxy semantic backfill target contains no measurement records"
                )
            measurement_ids = sorted(row.ref.object_id for row in measurements)
            if existing_review is not None:
                if existing_review.source_chunk_sha256 != source_sha256:
                    raise ValueError(
                        "existing proxy semantic review is stale: source chunk hash changed"
                    )
                if existing_review.measurement_ids != measurement_ids:
                    raise ValueError(
                        "existing proxy semantic review is stale: measurement identity set changed"
                    )
                resumed += 1
                continue

            prompt = build_proxy_semantic_prompt(
                source_chunk_ref=target.source_chunk_ref,
                source_payload=source_payload,
                measurements=measurements,
            )
            llm_calls += 1
            generation = backend.generate(prompt)
            annotations, candidate_rejections = compile_proxy_semantic_annotations_isolated(
                scope_id=scope_id,
                source_chunk_ref=target.source_chunk_ref,
                source_payload=source_payload,
                measurements=measurements,
                draft=generation.draft,
            )
            new_rejected_candidates += len(candidate_rejections)
            appended = _append_unique_models(
                path=annotation_file,
                rows=list(annotations),
                id_field="annotation_id",
                existing=existing_annotations,
            )
            new_annotations += appended

            review = ProxySemanticChunkReview(
                review_id=review_id,
                scope_id=scope_id,
                plan_id=plan.plan_id,
                source_chunk_ref=target.source_chunk_ref,
                source_chunk_sha256=source_sha256,
                measurement_ids=measurement_ids,
                annotation_ids=[row.annotation_id for row in annotations],
                emitted_annotation_count=len(annotations),
                draft_candidate_count=len(generation.draft.items),
                rejected_candidate_count=len(candidate_rejections),
                rejected_candidates=candidate_rejections,
                backend_name=backend.backend_name,
                model_name=backend.model_name,
                provider_input_tokens=generation.input_tokens,
                provider_output_tokens=generation.output_tokens,
                response_id=generation.response_id,
            )
            _append_unique_models(
                path=review_file,
                rows=[review],
                id_field="review_id",
                existing=existing_reviews,
            )
            successful_new += 1
        except Exception as exc:
            errors.append(
                ProxySemanticEnrichmentError(
                    source_chunk_ref=target.source_chunk_ref,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    completed = resumed + successful_new
    pending = plan.target_count - completed
    report_id = _stable_id(
        "proxy_semantic_enrichment_report",
        scope_id,
        plan.plan_id,
        EXTRACTOR_VERSION,
        str(annotation_file.resolve()),
        str(review_file.resolve()),
        completed,
        len(errors),
    )
    return ProxySemanticEnrichmentReport(
        report_id=report_id,
        scope_id=scope_id,
        source_plan_id=plan.plan_id,
        annotation_path=str(annotation_file.resolve()),
        review_path=str(review_file.resolve()),
        plan_target_count=plan.target_count,
        considered_target_count=considered,
        completed_target_count=completed,
        resumed_target_count=resumed,
        new_target_count=successful_new,
        pending_target_count=pending,
        failed_target_count=len(errors),
        new_annotation_count=new_annotations,
        total_annotation_count=len(existing_annotations),
        new_rejected_candidate_count=new_rejected_candidates,
        total_rejected_candidate_count=sum(
            int(row.rejected_candidate_count) for row in existing_reviews.values()
        ),
        llm_calls_performed=llm_calls,
        max_new_calls=max_new_calls,
        coverage_complete_for_plan=(completed == plan.target_count and not errors),
        execution_performed=bool(llm_calls),
        errors=errors,
    )
