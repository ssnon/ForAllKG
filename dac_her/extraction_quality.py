from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from dac_her.chunking import count_tokens
from dac_her.extraction_policy import ExtractionPolicy


QUALITY_COMPLETE = "complete"
QUALITY_PARTIAL_ACCEPTABLE = "partial_acceptable"
QUALITY_PARTIAL_CRITICAL = "partial_critical"
QUALITY_REJECTED = "rejected"

DEFAULT_USABLE_STATUSES = frozenset({
    QUALITY_COMPLETE,
    QUALITY_PARTIAL_ACCEPTABLE,
})

PROVENANCE_ISSUE_MARKERS = (
    "PROVENANCE",
    "EVIDENCE_POINTER",
    "PAPER_ID_MISMATCH",
    "CHUNK_ID_MISMATCH",
    "DOCUMENT_ID_MISMATCH",
    "DOCUMENT_ROLE_MISMATCH",
    "PAGE_ID",
    "ASSET_ID",
)


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _record_source_tokens(record: Mapping[str, Any]) -> int | None:
    direct = _as_int(record.get("source_tokens_estimated"))
    if direct is not None:
        return direct

    source_path_value = record.get("source_path")
    if not source_path_value:
        return None

    path = Path(str(source_path_value))
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    core_text = payload.get("core_text")
    if not isinstance(core_text, str):
        return None

    return count_tokens(core_text)


def quarantine_reason_class(record: Mapping[str, Any]) -> str:
    issue_counts = record.get("validation_issue_counts")
    if not isinstance(issue_counts, dict):
        issue_counts = {}

    issue_codes = [str(code).upper() for code in issue_counts]
    if any(
        marker in code
        for code in issue_codes
        for marker in PROVENANCE_ISSUE_MARKERS
    ):
        return "source_integrity"

    error_type = str(record.get("error_type") or "")
    reason = str(record.get("recovery_reason") or "")
    combined = f"{error_type} {reason}".lower()

    if any(
        marker in combined
        for marker in (
            "maxsplitdepthexceeded",
            "unsplittablerecovery",
            "cannot be split safely",
            "max_split_depth",
            "split budget",
            "micro-reextract",
            "micro re-extract",
            "recovery budget",
            "budgets were exhausted",
        )
    ):
        return "recovery_exhausted"

    issue_total = sum(
        int(value)
        for value in issue_counts.values()
        if isinstance(value, int) and value > 0
    )
    if issue_total:
        return "local_validation" if issue_total <= 2 else "complex_validation"

    return "unknown"


def quarantine_tier(record: Mapping[str, Any]) -> str:
    reason_class = quarantine_reason_class(record)
    tokens = _record_source_tokens(record)

    if reason_class == "source_integrity" or tokens is None:
        return "Q3_SOURCE_INTEGRITY"

    if reason_class in {
        "recovery_exhausted",
        "complex_validation",
    }:
        return "Q2_RECOVERY_OR_COMPLEX"

    return "Q1_LOCAL_VALIDATION"


def annotate_quarantined_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        tokens = _record_source_tokens(record)
        if tokens is not None:
            record["source_tokens_estimated"] = tokens
        record["quarantine_reason_class"] = quarantine_reason_class(record)
        record["quarantine_tier"] = quarantine_tier(record)
        annotated.append(record)
    return annotated


def _group_coverage(
    active_records: list[Mapping[str, Any]],
    quarantined_records: list[Mapping[str, Any]],
    failed_records: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "active_chunks": 0,
            "quarantined_chunks": 0,
            "failed_chunks": 0,
            "active_tokens": 0,
            "quarantined_tokens": 0,
            "failed_tokens": 0,
            "unknown_token_records": 0,
        }
    )

    def add(records: list[Mapping[str, Any]], bucket: str) -> None:
        for record in records:
            document_id = str(record.get("document_id") or "unknown")
            grouped[document_id][f"{bucket}_chunks"] += 1
            tokens = _record_source_tokens(record)
            if tokens is None:
                grouped[document_id]["unknown_token_records"] += 1
            else:
                grouped[document_id][f"{bucket}_tokens"] += tokens

    add(active_records, "active")
    add(quarantined_records, "quarantined")
    add(failed_records, "failed")

    result: dict[str, dict[str, Any]] = {}
    for document_id, values in sorted(grouped.items()):
        known_total = (
            values["active_tokens"]
            + values["quarantined_tokens"]
            + values["failed_tokens"]
        )
        result[document_id] = {
            **values,
            "known_source_tokens": known_total,
            "source_token_coverage": (
                values["active_tokens"] / known_total
                if known_total > 0
                else None
            ),
            "coverage_exact": values["unknown_token_records"] == 0,
        }
    return result


def evaluate_extraction_quality(
    *,
    active_records: Iterable[Mapping[str, Any]],
    quarantined_records: Iterable[Mapping[str, Any]],
    failed_records: Iterable[Mapping[str, Any]],
    policy: ExtractionPolicy,
) -> dict[str, Any]:
    active = [dict(record) for record in active_records]
    quarantined = annotate_quarantined_records(quarantined_records)
    failed = [dict(record) for record in failed_records]

    def token_summary(records: list[Mapping[str, Any]]) -> tuple[int, int]:
        known = 0
        unknown = 0
        for record in records:
            tokens = _record_source_tokens(record)
            if tokens is None:
                unknown += 1
            else:
                known += tokens
        return known, unknown

    active_tokens, active_unknown = token_summary(active)
    quarantine_tokens, quarantine_unknown = token_summary(quarantined)
    failed_tokens, failed_unknown = token_summary(failed)

    known_total_tokens = active_tokens + quarantine_tokens + failed_tokens
    unknown_token_records = active_unknown + quarantine_unknown + failed_unknown
    coverage_exact = unknown_token_records == 0

    source_token_coverage = (
        active_tokens / known_total_tokens
        if known_total_tokens > 0
        else None
    )
    quarantine_token_fraction = (
        quarantine_tokens / known_total_tokens
        if known_total_tokens > 0
        else None
    )
    failed_token_fraction = (
        failed_tokens / known_total_tokens
        if known_total_tokens > 0
        else None
    )

    active_count = len(active)
    quarantine_count = len(quarantined)
    failed_count = len(failed)
    terminal_count = active_count + quarantine_count + failed_count

    strict_complete = failed_count == 0 and quarantine_count == 0

    if strict_complete:
        status = QUALITY_COMPLETE
        reason = "No failed or quarantined terminal chunks."
    elif failed_count > 0:
        status = QUALITY_REJECTED
        reason = (
            "At least one unresolved hard/technical failure remains; "
            "paper-level graph materialization is rejected."
        )
    elif active_count == 0:
        status = QUALITY_REJECTED
        reason = "No strict-valid active chunks remain."
    elif not coverage_exact or source_token_coverage is None:
        status = QUALITY_PARTIAL_CRITICAL
        reason = (
            "Partial extraction has unknown source-token coverage; "
            "explicit override is required."
        )
    elif (
        source_token_coverage
        >= policy.partial_acceptable_min_source_token_coverage
        and quarantine_token_fraction
        <= policy.partial_acceptable_max_quarantine_token_fraction
    ):
        status = QUALITY_PARTIAL_ACCEPTABLE
        reason = (
            "Strict-valid source-token coverage meets the default "
            "partial-graph engineering gate."
        )
    elif source_token_coverage >= policy.partial_critical_min_source_token_coverage:
        status = QUALITY_PARTIAL_CRITICAL
        reason = (
            "Strict-valid evidence remains usable, but source-token coverage "
            "is below the default automatic materialization gate."
        )
    else:
        status = QUALITY_REJECTED
        reason = (
            "Strict-valid source-token coverage is below the minimum "
            "partial-graph threshold."
        )

    tier_counts = Counter(
        str(record.get("quarantine_tier", "UNKNOWN"))
        for record in quarantined
    )
    reason_counts = Counter(
        str(record.get("quarantine_reason_class", "unknown"))
        for record in quarantined
    )

    return {
        "schema_version": 1,
        "graph_materialization_status": status,
        "strict_complete": strict_complete,
        "graph_usable_by_default": status in DEFAULT_USABLE_STATUSES,
        "graph_usable_with_explicit_override": status == QUALITY_PARTIAL_CRITICAL,
        "positive_evidence_queries_allowed": status != QUALITY_REJECTED,
        "coverage_sensitive_queries_allowed": strict_complete,
        "absence_claims_allowed": strict_complete,
        "classification_reason": reason,
        "active_chunk_count": active_count,
        "quarantined_chunk_count": quarantine_count,
        "failed_chunk_count": failed_count,
        "terminal_chunk_count": terminal_count,
        "active_chunk_fraction": (
            active_count / terminal_count if terminal_count else 0.0
        ),
        "known_source_tokens": known_total_tokens,
        "active_source_tokens": active_tokens,
        "quarantined_source_tokens": quarantine_tokens,
        "failed_source_tokens": failed_tokens,
        "unknown_token_records": unknown_token_records,
        "coverage_exact": coverage_exact,
        "source_token_coverage": source_token_coverage,
        "quarantine_token_fraction": quarantine_token_fraction,
        "failed_token_fraction": failed_token_fraction,
        "quarantine_tier_counts": dict(sorted(tier_counts.items())),
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        "documents": _group_coverage(active, quarantined, failed),
        "policy": {
            "partial_acceptable_min_source_token_coverage": (
                policy.partial_acceptable_min_source_token_coverage
            ),
            "partial_acceptable_max_quarantine_token_fraction": (
                policy.partial_acceptable_max_quarantine_token_fraction
            ),
            "partial_critical_min_source_token_coverage": (
                policy.partial_critical_min_source_token_coverage
            ),
            "unresolved_failed_chunks_force_rejection": True,
        },
        "quarantined_chunks": quarantined,
    }


def quality_from_active_payload(
    active_payload: Mapping[str, Any],
    *,
    policy: ExtractionPolicy | None = None,
) -> dict[str, Any]:
    existing = active_payload.get("quality")
    if isinstance(existing, dict) and existing.get("graph_materialization_status"):
        return dict(existing)

    policy = policy or ExtractionPolicy()
    return evaluate_extraction_quality(
        active_records=active_payload.get("chunks") or [],
        quarantined_records=active_payload.get("quarantined_chunks") or [],
        failed_records=active_payload.get("failed_chunks") or [],
        policy=policy,
    )


def graph_quality_attributes(quality: Mapping[str, Any]) -> dict[str, Any]:
    def numeric_or_sentinel(name: str) -> float:
        value = quality.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        return -1.0

    return {
        "extraction_complete": bool(quality.get("strict_complete", False)),
        "extraction_quality_status": str(
            quality.get("graph_materialization_status", "unknown")
        ),
        "extraction_source_token_coverage": numeric_or_sentinel(
            "source_token_coverage"
        ),
        "extraction_quarantine_token_fraction": numeric_or_sentinel(
            "quarantine_token_fraction"
        ),
        "extraction_active_chunk_count": int(
            quality.get("active_chunk_count", 0)
        ),
        "extraction_quarantined_chunk_count": int(
            quality.get("quarantined_chunk_count", 0)
        ),
        "extraction_failed_chunk_count": int(
            quality.get("failed_chunk_count", 0)
        ),
        "extraction_coverage_exact": bool(
            quality.get("coverage_exact", False)
        ),
        "extraction_absence_claims_allowed": bool(
            quality.get("absence_claims_allowed", False)
        ),
        "extraction_coverage_sensitive_queries_allowed": bool(
            quality.get("coverage_sensitive_queries_allowed", False)
        ),
    }


def projection_quality_summary(graph: Any) -> dict[str, Any]:
    return {
        "extraction_quality_status": str(
            graph.graph.get("extraction_quality_status", "unknown")
        ),
        "extraction_complete": bool(
            graph.graph.get("extraction_complete", False)
        ),
        "extraction_source_token_coverage": float(
            graph.graph.get("extraction_source_token_coverage", -1.0)
        ),
        "extraction_quarantine_token_fraction": float(
            graph.graph.get("extraction_quarantine_token_fraction", -1.0)
        ),
        "extraction_active_chunk_count": int(
            graph.graph.get("extraction_active_chunk_count", 0)
        ),
        "extraction_quarantined_chunk_count": int(
            graph.graph.get("extraction_quarantined_chunk_count", 0)
        ),
        "extraction_failed_chunk_count": int(
            graph.graph.get("extraction_failed_chunk_count", 0)
        ),
        "extraction_coverage_exact": bool(
            graph.graph.get("extraction_coverage_exact", False)
        ),
        "extraction_absence_claims_allowed": bool(
            graph.graph.get("extraction_absence_claims_allowed", False)
        ),
        "extraction_coverage_sensitive_queries_allowed": bool(
            graph.graph.get(
                "extraction_coverage_sensitive_queries_allowed",
                False,
            )
        ),
    }
