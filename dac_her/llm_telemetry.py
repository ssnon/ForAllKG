from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "llm-call-usage-v1.1"
ARTIFACT_RESOLUTION_SCHEMA_VERSION = "llm-artifact-resolution-v1"
_TOKENIZER_NAME = "cl100k_base"

# These section names already occur in the extraction/recovery prompts. The
# parser is diagnostic-only: it never changes the prompt sent to the model.
_SECTION_COMPONENTS = {
    "ASSET_CONTEXT": "asset_context",
    "VOCABULARY_CONTEXT": "vocabulary",
    "LEFT_CONTEXT": "left_context",
    "CORE_TEXT": "source",
    "RIGHT_CONTEXT": "right_context",
    "PREVIOUS VALIDATION ERROR": "issues",
    "CURRENT_GRAPH_DRAFT_JSON": "current_draft",
    "CURRENT GRAPH DRAFT JSON": "current_draft",
    "ALL STRUCTURED_VALIDATION_ISSUES": "issues",
    "STRUCTURED VALIDATION ISSUES": "issues",
    "PREVIOUS_PATCH_FEEDBACK": "previous_stage_output",
    "PREVIOUS PATCH FEEDBACK": "previous_stage_output",
    # Hypothesis-planning / critic prompts. These are diagnostic views of the
    # serialized user surface and are therefore not added to estimated_sum.
    "ELIGIBLE POSITIVE PREMISES": "premises",
    "ELIGIBLE PREMISES": "premises",
    "SELECTED PREMISES": "premises",
    "VALIDATED AXIS-EVIDENCE AUDIT": "axis_audit",
    "AXIS-EVIDENCE AUDIT": "axis_audit",
    "BLUEPRINT": "blueprint",
    "RETRIEVED PRIOR-ART CANDIDATES": "prior_art_candidates",
    "POLICY": "policy",
    "RESEARCH GAPS": "gaps",
    "MECHANISM ROUTES": "mechanism_routes",
    "MECHANISTIC MOTIFS": "mechanistic_motifs",
    "REPORTED DESIGN LEVERS": "design_levers",
    "RESTRICTED / NON-PREMISE STATEMENTS": "restricted_context",
    "PARTIAL-PAPER ABSENCE SAFETY": "safety_context",
    "ISSUES": "issues",
    "PREVIOUS DRAFT": "previous_stage_output",
}
_CONTEXT_LABELS = {
    "PAPER_ID": "paper_id",
    "CHUNK_ID": "chunk_id",
    "HYPOTHESIS_ID": "hypothesis_id",
    "ROUTE_ID": "route_id",
    "AXIS_ID": "axis_id",
    "CLAIM_ID": "claim_id",
}


@dataclass(frozen=True)
class TokenComponentEstimate:
    estimated_tokens: int
    fingerprint: str
    estimator: str
    counted_in_estimated_sum: bool = False


@dataclass(frozen=True)
class LLMCallUsageEvent:
    schema_version: str = SCHEMA_VERSION
    record_type: str = "call"
    call_id: str = field(default_factory=lambda: f"llm_call:{uuid.uuid4().hex}")
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    run_id: str | None = None
    parent_call_id: str | None = None
    pipeline: str | None = None
    stage: str | None = None
    call_kind: str | None = None
    response_model: str | None = None
    attempt: int | None = None

    paper_id: str | None = None
    chunk_id: str | None = None
    hypothesis_id: str | None = None
    route_id: str | None = None
    axis_id: str | None = None
    claim_id: str | None = None

    requested_model: str | None = None
    served_model: str | None = None
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    provider_total_tokens: int | None = None
    provider_usage_scope: str | None = None
    configured_max_retries: int | None = None

    estimated_components: dict[str, TokenComponentEstimate] = field(
        default_factory=dict
    )
    estimated_sum: int | None = None
    token_estimate_gap: int | None = None

    elapsed_seconds: float | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    call_outcome: str | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMArtifactResolutionEvent:
    schema_version: str = ARTIFACT_RESOLUTION_SCHEMA_VERSION
    record_type: str = "artifact_resolution"
    resolution_id: str = field(
        default_factory=lambda: f"llm_resolution:{uuid.uuid4().hex}"
    )
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    call_id: str = ""
    run_id: str | None = None
    paper_id: str | None = None
    chunk_id: str | None = None
    artifact_outcome: str = "unknown"
    terminal_contribution: str = "unknown"
    final_materialization_status: str | None = None
    record_status: str | None = None
    acceptance_mode: str | None = None
    resolution_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def component_fingerprint(value: Any) -> str:
    text = _canonical_text(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _token_encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding(_TOKENIZER_NAME)


def estimate_tokens(value: Any) -> tuple[int, str]:
    """Return a local diagnostic estimate, never a billing value."""

    text = _canonical_text(value)
    if not text:
        return 0, _TOKENIZER_NAME

    try:
        encoding = _token_encoding()
        return len(encoding.encode(text)), _TOKENIZER_NAME
    except Exception:  # pragma: no cover - fallback for minimal environments
        # Stable dependency-free fallback. Provider-reported usage remains the
        # source of truth; this is only for relative component diagnostics.
        return max(1, math.ceil(len(text.encode("utf-8")) / 4)), "utf8_bytes_div4"


def estimate_component(
    value: Any,
    *,
    counted_in_estimated_sum: bool = False,
) -> TokenComponentEstimate:
    tokens, estimator = estimate_tokens(value)
    return TokenComponentEstimate(
        estimated_tokens=tokens,
        fingerprint=component_fingerprint(value),
        estimator=estimator,
        counted_in_estimated_sum=counted_in_estimated_sum,
    )


def _extract_labeled_sections(prompt: str) -> dict[str, str]:
    lines = prompt.splitlines()
    found: dict[str, list[str]] = {}
    current: str | None = None

    for line in lines:
        stripped = line.strip()
        candidate = stripped[:-1].strip() if stripped.endswith(":") else stripped
        label = candidate.upper()
        if label in _SECTION_COMPONENTS:
            current = _SECTION_COMPONENTS[label]
            found.setdefault(current, [])
            continue
        if current is not None:
            found[current].append(line)

    return {
        key: "\n".join(value).strip()
        for key, value in found.items()
        if "\n".join(value).strip()
    }




def _extract_semantic_review_payload(prompt: str) -> dict[str, Any]:
    marker = "SEMANTIC REVIEW INPUT"
    output_marker = "OUTPUT REQUIREMENTS"
    if marker not in prompt or output_marker not in prompt:
        return {}
    try:
        body = prompt.split(marker, 1)[1]
        # Skip the underline immediately following the heading.
        body = body.split("\n", 2)[2]
        payload_text = body.split(output_marker, 1)[0].strip()
        payload = json.loads(payload_text)
    except (IndexError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    mapping = {
        "eligible_positive_premises": "premises",
        "research_gaps": "gaps",
        "restricted_nonpremise_statements": "restricted_context",
        "mechanism_routes": "mechanism_routes",
        "deterministic_diagnostics": "deterministic_diagnostics",
        "hypothesis_portfolio": "hypothesis_portfolio",
        "task": "task_context",
    }
    return {
        component: payload[key]
        for key, component in mapping.items()
        if key in payload
    }


def infer_prompt_context(prompt: str | None) -> dict[str, Any]:
    """Infer stable identifiers already serialized in labeled prompts."""

    if not prompt:
        return {}
    lines = prompt.splitlines()
    inferred: dict[str, Any] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if ":" in stripped:
            left, right = stripped.split(":", 1)
            key = _CONTEXT_LABELS.get(left.strip().upper())
            if key is not None and right.strip():
                inferred[key] = right.strip()
                continue
        label = stripped.rstrip(":").upper()
        key = _CONTEXT_LABELS.get(label)
        if key is None or not stripped.endswith(":"):
            continue
        for candidate in lines[index + 1 :]:
            value = candidate.strip()
            if value:
                inferred[key] = value
                break
    # Some structured planning prompts serialize metadata inside a JSON/Python
    # mapping rather than as a dedicated label line (for example AXIS
    # {"axis_id": "..."}). Pick up those stable identifiers diagnostically.
    for label, key in _CONTEXT_LABELS.items():
        if key in inferred:
            continue
        pattern = rf"[\"']{re.escape(label.lower())}[\"']\s*:\s*[\"']([^\"']+)[\"']"
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            inferred[key] = match.group(1).strip()

    if "CORE_TEXT:" in prompt and "PAPER_ID:" in prompt:
        inferred.setdefault("pipeline", "extraction")
    return inferred


def estimate_prompt_components(
    *,
    system_prompt: str | None,
    user_prompt: str | None,
    response_schema: Any | None = None,
    semantic_components: Mapping[str, Any] | None = None,
    extra_messages: list[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, TokenComponentEstimate], int]:
    """Estimate serialized input components without changing call semantics.

    ``estimated_sum`` is intentionally based only on the non-overlapping top
    level surfaces (system + user prompt + schema + extra messages). Semantic
    subcomponents such as source/vocabulary/current_draft are nested diagnostic
    views of the user/extra-message surface and are therefore *not* added twice.
    """

    components: dict[str, TokenComponentEstimate] = {}
    top_level_names: list[str] = []

    if system_prompt is not None:
        components["system"] = estimate_component(system_prompt, counted_in_estimated_sum=True)
        top_level_names.append("system")
    if user_prompt is not None:
        components["user_prompt"] = estimate_component(user_prompt, counted_in_estimated_sum=True)
        top_level_names.append("user_prompt")
    if response_schema is not None:
        components["schema"] = estimate_component(response_schema, counted_in_estimated_sum=True)
        top_level_names.append("schema")
    if extra_messages:
        components["extra_messages"] = estimate_component(extra_messages, counted_in_estimated_sum=True)
        top_level_names.append("extra_messages")

    inferred = _extract_labeled_sections(user_prompt or "")
    for name, value in inferred.items():
        components.setdefault(name, estimate_component(value))

    semantic_payload = _extract_semantic_review_payload(user_prompt or "")
    for name, value in semantic_payload.items():
        components[name] = estimate_component(value)

    for name, value in (semantic_components or {}).items():
        if value is None:
            continue
        components[str(name)] = estimate_component(value)

    estimated_sum = sum(
        components[name].estimated_tokens
        for name in top_level_names
        if name in components
    )
    return components, estimated_sum


def completion_metadata(completion: Any | None) -> dict[str, Any]:
    if completion is None:
        return {
            "served_model": None,
            "provider_input_tokens": None,
            "provider_output_tokens": None,
            "provider_total_tokens": None,
            "finish_reason": None,
            "response_id": None,
        }

    usage = getattr(completion, "usage", None)
    choices = getattr(completion, "choices", None) or []
    choice = choices[0] if choices else None
    return {
        "served_model": getattr(completion, "model", None),
        "provider_input_tokens": (
            getattr(usage, "prompt_tokens", None) if usage is not None else None
        ),
        "provider_output_tokens": (
            getattr(usage, "completion_tokens", None) if usage is not None else None
        ),
        "provider_total_tokens": (
            getattr(usage, "total_tokens", None) if usage is not None else None
        ),
        "finish_reason": getattr(choice, "finish_reason", None),
        "response_id": getattr(completion, "id", None),
    }


def normalize_stage_name(
    stage: str | None,
    *,
    response_model: str | None = None,
) -> str | None:
    """Normalize response-model-shaped labels into pipeline stage names."""

    raw = str(stage or "").strip()
    model = str(response_model or "").strip()
    candidate = raw or model
    mapping = {
        "KnowledgeGraphDraft": "graph_generation",
        "KnowledgeGraphPatch": "semantic_patch",
    }
    if candidate in mapping:
        return mapping[candidate]
    lowered = candidate.lower()
    if "semantic" in lowered and "patch" in lowered:
        return "semantic_patch"
    if "micro" in lowered and ("extract" in lowered or "reextract" in lowered):
        return "micro_reextract"
    return candidate or None


def build_usage_event(
    *,
    requested_model: str | None,
    completion: Any | None,
    system_prompt: str | None,
    user_prompt: str | None,
    response_schema: Any | None = None,
    semantic_components: Mapping[str, Any] | None = None,
    extra_messages: list[Mapping[str, Any]] | None = None,
    elapsed_seconds: float | None = None,
    outcome: str | None = "success",
    rejection_reason: str | None = None,
    context: Mapping[str, Any] | None = None,
    provider_usage_scope: str | None = None,
    configured_max_retries: int | None = None,
) -> LLMCallUsageEvent:
    components, estimated_sum = estimate_prompt_components(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=response_schema,
        semantic_components=semantic_components,
        extra_messages=extra_messages,
    )
    provider = completion_metadata(completion)
    provider_input = provider["provider_input_tokens"]
    gap = (
        int(provider_input) - estimated_sum
        if provider_input is not None
        else None
    )

    allowed_context = {
        "run_id",
        "parent_call_id",
        "pipeline",
        "stage",
        "call_kind",
        "response_model",
        "attempt",
        "paper_id",
        "chunk_id",
        "hypothesis_id",
        "route_id",
        "axis_id",
        "claim_id",
    }
    merged_context = infer_prompt_context(user_prompt)
    merged_context.update(dict(context or {}))
    cleaned = {
        key: value
        for key, value in merged_context.items()
        if key in allowed_context
    }

    cleaned["stage"] = normalize_stage_name(
        cleaned.get("stage"),
        response_model=cleaned.get("response_model"),
    )

    return LLMCallUsageEvent(
        **cleaned,
        requested_model=requested_model,
        served_model=provider["served_model"],
        provider_input_tokens=provider_input,
        provider_output_tokens=provider["provider_output_tokens"],
        provider_total_tokens=provider["provider_total_tokens"],
        provider_usage_scope=provider_usage_scope,
        configured_max_retries=configured_max_retries,
        estimated_components=components,
        estimated_sum=estimated_sum,
        token_estimate_gap=gap,
        elapsed_seconds=elapsed_seconds,
        finish_reason=provider["finish_reason"],
        response_id=provider["response_id"],
        call_outcome=outcome,
        rejection_reason=rejection_reason,
    )


def resolve_telemetry_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path)
    value = os.getenv("GRAPHAGENTS_LLM_TELEMETRY_PATH", "").strip()
    return Path(value) if value else None


def append_usage_event(
    path: str | Path | None,
    event: LLMCallUsageEvent,
) -> bool:
    """Append one JSONL event without allowing telemetry I/O to fail science.

    The ledger is observational. A missing/unwritable telemetry destination
    must never change extraction, recovery, or hypothesis acceptance behavior.
    """

    resolved = resolve_telemetry_path(path)
    if resolved is None:
        return False
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            resolved,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        return True
    except OSError as error:
        warnings.warn(
            f"LLM telemetry append failed for {resolved}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


def append_artifact_resolution(
    path: str | Path | None,
    event: LLMArtifactResolutionEvent,
) -> bool:
    resolved = resolve_telemetry_path(path)
    if resolved is None:
        return False
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            resolved,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        return True
    except OSError as error:
        warnings.warn(
            f"LLM telemetry resolution append failed for {resolved}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


def _attempt_call_id(attempt: Mapping[str, Any]) -> str | None:
    telemetry = attempt.get("telemetry_event")
    if not isinstance(telemetry, Mapping):
        return None
    value = telemetry.get("call_id")
    return str(value) if value else None


def append_extraction_artifact_resolutions(
    path: str | Path | None,
    *,
    run_id: str,
    paper_id: str,
    materialization_status: str,
    active_records: Any,
    quarantined_records: Any,
    failed_records: Any,
    allowed_call_ids: set[str] | None = None,
) -> int:
    """Resolve extraction calls after deterministic paper quality is known."""

    if resolve_telemetry_path(path) is None:
        return 0

    written = 0
    paper_rejected = materialization_status == "rejected"

    def emit_for_record(record: Mapping[str, Any], *, family: str) -> None:
        nonlocal written
        attempts = record.get("attempt_usages")
        if not isinstance(attempts, list):
            return
        usable_attempts = [row for row in attempts if isinstance(row, Mapping)]
        call_rows = [
            (index, row, _attempt_call_id(row))
            for index, row in enumerate(usable_attempts)
        ]
        call_rows = [row for row in call_rows if row[2]]
        if not call_rows:
            return

        record_status = str(record.get("status") or family)
        repaired = (
            len(call_rows) > 1
            or int(record.get("patch_attempts") or 0) > 0
            or int(record.get("micro_reextract_attempts") or 0) > 0
            or int(record.get("post_micro_patch_attempts") or 0) > 0
        )

        if paper_rejected:
            artifact_outcome = "rejected"
        elif family == "active":
            if materialization_status in {"partial_acceptable", "partial_critical"}:
                artifact_outcome = "accepted_partial"
            else:
                artifact_outcome = (
                    "accepted_after_repair" if repaired else "accepted"
                )
        elif family == "quarantined":
            artifact_outcome = "quarantined"
        elif family == "failed":
            artifact_outcome = "failed"
        else:
            artifact_outcome = "unknown"

        last_call_id = str(call_rows[-1][2])
        for _, attempt, call_id_raw in call_rows:
            call_id = str(call_id_raw)
            if allowed_call_ids is not None and call_id not in allowed_call_ids:
                continue
            attempt_event = attempt.get("telemetry_event")
            call_outcome = (
                str(
                    attempt_event.get("call_outcome")
                    or attempt_event.get("outcome")
                    or ""
                )
                if isinstance(attempt_event, Mapping)
                else ""
            )
            attempt_failed = bool(
                attempt.get("error_type")
                or call_outcome in {
                    "error",
                    "validation_error",
                    "provider_error",
                }
            )
            if artifact_outcome in {
                "accepted",
                "accepted_after_repair",
                "accepted_partial",
            }:
                if attempt_failed:
                    contribution = "discarded"
                elif call_id == last_call_id:
                    contribution = "terminal"
                else:
                    contribution = "non_terminal"
            else:
                contribution = "discarded"

            reason = (
                f"record_status={record_status}; "
                f"materialization_status={materialization_status}; "
                f"record_family={family}"
            )
            event = LLMArtifactResolutionEvent(
                call_id=call_id,
                run_id=run_id,
                paper_id=paper_id,
                chunk_id=str(record.get("chunk_id") or "") or None,
                artifact_outcome=artifact_outcome,
                terminal_contribution=contribution,
                final_materialization_status=materialization_status,
                record_status=record_status,
                acceptance_mode=(
                    str(record.get("acceptance_mode"))
                    if record.get("acceptance_mode") is not None
                    else None
                ),
                resolution_reason=reason,
            )
            if append_artifact_resolution(path, event):
                written += 1

    for row in active_records:
        if isinstance(row, Mapping):
            emit_for_record(row, family="active")
    for row in quarantined_records:
        if isinstance(row, Mapping):
            emit_for_record(row, family="quarantined")
    for row in failed_records:
        if isinstance(row, Mapping):
            emit_for_record(row, family="failed")
    return written


def instructor_create_with_completion(
    completions_api: Any,
    **kwargs: Any,
) -> tuple[Any, Any | None]:
    """Use Instructor's raw-completion API when available, with fallback."""

    create_with_completion = getattr(
        completions_api,
        "create_with_completion",
        None,
    )
    if callable(create_with_completion):
        result = create_with_completion(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        # Do not issue a second provider request merely because a third-party
        # compatible client returned a non-standard shape.
        return result, getattr(result, "_raw_response", None)
    return completions_api.create(**kwargs), None


def run_instructor_structured_call(
    completions_api: Any,
    *,
    model: str,
    response_model: Any,
    messages: list[Mapping[str, Any]],
    temperature: float,
    max_retries: int,
    telemetry_path: str | Path | None = None,
    telemetry_context: Mapping[str, Any] | None = None,
    semantic_components: Mapping[str, Any] | None = None,
    request_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, LLMCallUsageEvent]:
    """Execute one existing Instructor call while recording usage telemetry.

    The model request payload is unchanged apart from using Instructor's
    create_with_completion API when available so the provider usage object can
    be observed. Prompt text is never written to telemetry.
    """

    system_prompt: str | None = None
    user_prompt: str | None = None
    extra_messages: list[Mapping[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        if role == "system" and system_prompt is None and isinstance(content, str):
            system_prompt = content
            continue
        if role == "user" and user_prompt is None and isinstance(content, str):
            user_prompt = content
            continue
        extra_messages.append(message)

    schema = response_model.model_json_schema()
    telemetry_context = {
        "response_model": getattr(response_model, "__name__", str(response_model)),
        **dict(telemetry_context or {}),
    }
    started = time.perf_counter()
    try:
        call_kwargs = {
            "model": model,
            "response_model": response_model,
            "messages": messages,
            "temperature": temperature,
            "max_retries": max_retries,
            **dict(request_kwargs or {}),
        }
        draft, completion = instructor_create_with_completion(
            completions_api,
            **call_kwargs,
        )
    except Exception as error:
        elapsed = time.perf_counter() - started
        event = build_usage_event(
            requested_model=model,
            completion=None,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=schema,
            semantic_components=semantic_components,
            extra_messages=extra_messages,
            elapsed_seconds=elapsed,
            outcome="error",
            rejection_reason=type(error).__name__,
            context=telemetry_context,
            provider_usage_scope="returned_completion",
            configured_max_retries=max_retries,
        )
        append_usage_event(telemetry_path, event)
        raise

    elapsed = time.perf_counter() - started
    event = build_usage_event(
        requested_model=model,
        completion=completion,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=schema,
        semantic_components=semantic_components,
        extra_messages=extra_messages,
        elapsed_seconds=elapsed,
        outcome="success",
        context=telemetry_context,
        provider_usage_scope="returned_completion",
        configured_max_retries=max_retries,
    )
    append_usage_event(telemetry_path, event)
    return draft, event
