from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


AUDIT_SCHEMA_VERSION = "prior-art-review-audit-v1"
AUDIT_PATH_ENV = "GRAPHAGENTS_PRIOR_ART_REVIEW_AUDIT_PATH"

_AUDIT_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "graphagents_prior_art_review_audit_context",
    default={},
)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


@contextlib.contextmanager
def prior_art_review_audit_scope(
    **context: Any,
) -> Iterator[None]:
    """Attach diagnostic-only assessment context to review audit rows.

    This ContextVar is deliberately outside the scientific objects. It cannot
    change retrieval, ranking, prompts, model parameters, compilation, or
    acceptance decisions.
    """

    current = dict(_AUDIT_CONTEXT.get())
    current.update(
        {
            str(key): value
            for key, value in context.items()
            if value is not None
        }
    )
    token = _AUDIT_CONTEXT.set(current)
    try:
        yield
    finally:
        _AUDIT_CONTEXT.reset(token)


def current_prior_art_review_audit_context() -> dict[str, Any]:
    return dict(_AUDIT_CONTEXT.get())


def _event_payload(event: Any) -> dict[str, Any]:
    if event is None:
        return {}
    if hasattr(event, "to_dict"):
        try:
            payload = event.to_dict()
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    if isinstance(event, dict):
        return dict(event)
    return {}


def record_prior_art_review_call(
    *,
    system_prompt: str,
    user_prompt: str,
    response_schema: Any,
    result: Any,
    model: str,
    instructor_mode: str,
    temperature: float,
    claim_id: str,
    hypothesis_id: str,
    claim_text: str,
    works: list[Mapping[str, Any]],
    telemetry_event: Any = None,
) -> None:
    """Append one successful review call to the audit JSONL.

    The function is best-effort by design. An audit filesystem failure must
    never change the scientific pipeline outcome.
    """

    raw_path = os.getenv(AUDIT_PATH_ENV)
    if not raw_path:
        return

    try:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        schema_payload = (
            response_schema.model_json_schema()
            if hasattr(response_schema, "model_json_schema")
            else response_schema
        )
        semantic_request = {
            "model": str(model),
            "instructor_mode": str(instructor_mode),
            "temperature": float(temperature),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_schema": schema_payload,
        }
        event = _event_payload(telemetry_event)
        result_payload = (
            result.model_dump(mode="json")
            if hasattr(result, "model_dump")
            else result
        )
        context = current_prior_art_review_audit_context()

        record = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "record_type": "prior_art_review_call",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request_fingerprint": _sha256_json(semantic_request),
            "prompt_fingerprint": _sha256_text(
                system_prompt + "\n---\n" + user_prompt
            ),
            "schema_fingerprint": _sha256_json(schema_payload),
            "response_fingerprint": _sha256_json(result_payload),
            "model": str(model),
            "instructor_mode": str(instructor_mode),
            "temperature": float(temperature),
            "claim_id": str(claim_id),
            "hypothesis_id": str(hypothesis_id),
            "claim_text": str(claim_text),
            "candidate_work_ids": [
                str(row.get("work_id"))
                for row in works
                if row.get("work_id") is not None
            ],
            "candidate_work_count": len(works),
            "response_payload": result_payload,
            "assessment_context": context,
            "telemetry_call_id": event.get("call_id"),
            "provider_input_tokens": event.get("provider_input_tokens"),
            "provider_output_tokens": event.get("provider_output_tokens"),
            "provider_total_tokens": event.get("provider_total_tokens"),
            # Present after PR-O1; safely absent on older telemetry schemas.
            "provider_cost_credits": event.get("provider_cost_credits"),
            "provider_cached_input_tokens": event.get(
                "provider_cached_input_tokens"
            ),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
    except Exception as exc:  # pragma: no cover - defensive observability guard
        warnings.warn(
            f"Prior-art review audit write failed and was ignored: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def load_prior_art_review_audit(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if (
            isinstance(row, dict)
            and row.get("record_type") == "prior_art_review_call"
        ):
            rows.append(row)
    return rows


def summarize_prior_art_review_audit(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    seen: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    by_kind: dict[str, dict[str, Any]] = {}
    transitions: dict[tuple[str, str], int] = {}

    duplicate_calls = 0
    duplicate_cost = 0.0
    duplicate_cost_observations = 0
    duplicate_input = 0
    duplicate_output = 0
    duplicate_total = 0

    targeted_calls = 0
    targeted_non_focal_calls = 0
    targeted_non_focal_duplicates = 0
    targeted_non_focal_duplicate_cost = 0.0
    targeted_non_focal_duplicate_cost_observations = 0

    for index, row in enumerate(rows):
        fingerprint = str(row.get("request_fingerprint") or "")
        if not fingerprint:
            continue
        groups.setdefault(fingerprint, []).append(row)

        context = row.get("assessment_context")
        context = context if isinstance(context, dict) else {}
        kind = str(context.get("assessment_kind") or "unknown")
        stats = by_kind.setdefault(
            kind,
            {
                "calls": 0,
                "new_unique_requests": 0,
                "duplicate_calls": 0,
                "provider_cost_credits": 0.0,
                "provider_cost_observations": 0,
                "duplicate_cost_credits": 0.0,
                "duplicate_cost_observations": 0,
            },
        )
        stats["calls"] += 1
        cost = row.get("provider_cost_credits")
        if isinstance(cost, (int, float)):
            stats["provider_cost_credits"] += float(cost)
            stats["provider_cost_observations"] += 1

        first = seen.get(fingerprint)
        is_duplicate = first is not None
        if is_duplicate:
            duplicate_calls += 1
            stats["duplicate_calls"] += 1
            first_context = first.get("assessment_context")
            first_context = (
                first_context if isinstance(first_context, dict) else {}
            )
            first_kind = str(
                first_context.get("assessment_kind") or "unknown"
            )
            transitions[(first_kind, kind)] = (
                transitions.get((first_kind, kind), 0) + 1
            )
            if isinstance(cost, (int, float)):
                duplicate_cost += float(cost)
                duplicate_cost_observations += 1
                stats["duplicate_cost_credits"] += float(cost)
                stats["duplicate_cost_observations"] += 1
            for field, name in (
                ("provider_input_tokens", "input"),
                ("provider_output_tokens", "output"),
                ("provider_total_tokens", "total"),
            ):
                value = row.get(field)
                if isinstance(value, int):
                    if name == "input":
                        duplicate_input += value
                    elif name == "output":
                        duplicate_output += value
                    else:
                        duplicate_total += value
        else:
            seen[fingerprint] = row
            stats["new_unique_requests"] += 1

        if kind == "alpha6_targeted_reassessment":
            targeted_calls += 1
            focal = context.get("focal_hypothesis_id")
            non_focal = bool(focal) and str(row.get("hypothesis_id")) != str(
                focal
            )
            if non_focal:
                targeted_non_focal_calls += 1
                if is_duplicate:
                    targeted_non_focal_duplicates += 1
                    if isinstance(cost, (int, float)):
                        targeted_non_focal_duplicate_cost += float(cost)
                        targeted_non_focal_duplicate_cost_observations += 1

    duplicate_groups = [
        group for group in groups.values() if len(group) > 1
    ]
    stable_groups = 0
    divergent_groups = 0
    divergent_requests: list[dict[str, Any]] = []
    for group in duplicate_groups:
        responses = {
            str(row.get("response_fingerprint") or "")
            for row in group
        }
        responses.discard("")
        if len(responses) <= 1:
            stable_groups += 1
        else:
            divergent_groups += 1
            first = group[0]
            divergent_requests.append(
                {
                    "request_fingerprint": first.get(
                        "request_fingerprint"
                    ),
                    "claim_id": first.get("claim_id"),
                    "hypothesis_id": first.get("hypothesis_id"),
                    "calls": len(group),
                    "unique_response_fingerprints": len(responses),
                    "assessment_kinds": [
                        (
                            row.get("assessment_context", {})
                            or {}
                        ).get("assessment_kind")
                        for row in group
                    ],
                }
            )

    total_cost = sum(
        float(row["provider_cost_credits"])
        for row in rows
        if isinstance(row.get("provider_cost_credits"), (int, float))
    )
    total_cost_observations = sum(
        isinstance(row.get("provider_cost_credits"), (int, float))
        for row in rows
    )

    return {
        "schema_version": "prior-art-review-audit-summary-v1",
        "quality_policy": (
            "This is diagnostic only. Exact-repeat reuse is not authorized by "
            "this report; any future reuse requires zero-loss validation."
        ),
        "calls": len(rows),
        "unique_request_fingerprints": len(groups),
        "duplicate_calls": duplicate_calls,
        "duplicate_fraction_of_calls": (
            duplicate_calls / len(rows) if rows else 0.0
        ),
        "provider_cost_credits": total_cost,
        "provider_cost_observations": total_cost_observations,
        "duplicate_cost_credits": duplicate_cost,
        "duplicate_cost_observations": duplicate_cost_observations,
        "duplicate_cost_fraction_of_observed_review_cost": (
            duplicate_cost / total_cost if total_cost else None
        ),
        "duplicate_input_tokens": duplicate_input,
        "duplicate_output_tokens": duplicate_output,
        "duplicate_total_tokens": duplicate_total,
        "duplicate_request_groups": len(duplicate_groups),
        "response_stable_duplicate_groups": stable_groups,
        "response_divergent_duplicate_groups": divergent_groups,
        "response_stability_fraction_of_duplicate_groups": (
            stable_groups / len(duplicate_groups)
            if duplicate_groups
            else None
        ),
        "targeted_reassessment_calls": targeted_calls,
        "targeted_reassessment_non_focal_calls": targeted_non_focal_calls,
        "targeted_reassessment_non_focal_duplicate_calls": (
            targeted_non_focal_duplicates
        ),
        "targeted_reassessment_non_focal_duplicate_cost_credits": (
            targeted_non_focal_duplicate_cost
        ),
        "targeted_reassessment_non_focal_duplicate_cost_observations": (
            targeted_non_focal_duplicate_cost_observations
        ),
        "by_assessment_kind": {
            key: value
            for key, value in sorted(by_kind.items())
        },
        "duplicate_transitions": [
            {
                "first_assessment_kind": first,
                "repeat_assessment_kind": repeat,
                "calls": count,
            }
            for (first, repeat), count in sorted(transitions.items())
        ],
        "divergent_requests": divergent_requests,
    }
