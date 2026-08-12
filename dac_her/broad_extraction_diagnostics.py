from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


DIAGNOSTICS_SCHEMA_VERSION = "graphagentsdac-broad-extraction-diagnostics-v3-run-bound"

_RELATION_MISMATCH_CODES = frozenset({
    "RELATION_SOURCE_TYPE_MISMATCH",
    "RELATION_TARGET_TYPE_MISMATCH",
})


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(dict(payload))
    except (OSError, json.JSONDecodeError):
        return rows
    return rows


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _latest_run_dir(paper_root: Path) -> Path | None:
    pointer = _read_json(paper_root / "latest_run.json")
    if pointer is None:
        return None
    raw = pointer.get("run_directory")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None


def _terminal_records(active_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("chunks", "quarantined_chunks", "failed_chunks"):
        value = active_payload.get(key)
        if isinstance(value, list):
            rows.extend(dict(row) for row in value if isinstance(row, dict))
    return rows


def _aggregate_attempt_usage(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    call_kind_counts: Counter[str] = Counter()
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    llm_calls = 0

    for record in records:
        usages = record.get("attempt_usages")
        if not isinstance(usages, list):
            continue
        for raw_usage in usages:
            if not isinstance(raw_usage, dict):
                continue
            usage = dict(raw_usage)
            llm_calls += 1
            call_kind = str(usage.get("call_kind") or "unknown")
            call_kind_counts[call_kind] += 1
            in_tokens = _int(usage.get("input_tokens"))
            out_tokens = _int(usage.get("output_tokens"))
            explicit_total = _int(usage.get("total_tokens"))
            input_tokens += in_tokens
            output_tokens += out_tokens
            total_tokens += explicit_total or (in_tokens + out_tokens)

    return {
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "call_kind_counts": _counter_dict(call_kind_counts),
    }


def _terminal_issue_counts(
    records: Iterable[Mapping[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        raw = record.get("validation_issue_counts")
        if not isinstance(raw, dict):
            continue
        for code, count in raw.items():
            counter[str(code)] += _int(count)
    return counter


def _observed_validation_issue_counts(run_dir: Path) -> Counter[str]:
    """Count issue appearances across saved validation reports.

    These counts intentionally include repeated appearances during recovery.
    They are useful for identifying the validator families that consume repair
    budget, but are not interpreted as distinct scientific errors.
    """
    counter: Counter[str] = Counter()
    validation_dir = run_dir / "validation"
    if not validation_dir.exists():
        return counter
    for path in sorted(validation_dir.glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        issues = payload.get("issues")
        if not isinstance(issues, list):
            continue
        for item in issues:
            if isinstance(item, dict) and item.get("code"):
                counter[str(item["code"])] += 1
    return counter


def _relation_mismatch_patterns(run_dir: Path) -> list[dict[str, Any]]:
    """Aggregate relation endpoint mismatches across saved validation reports."""
    counts: Counter[tuple[str, str, str, str, tuple[str, ...]]] = Counter()
    validation_dir = run_dir / "validation"
    if not validation_dir.exists():
        return []
    for path in sorted(validation_dir.glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        issues = payload.get("issues")
        if not isinstance(issues, list):
            continue
        for raw in issues:
            if not isinstance(raw, dict):
                continue
            code = str(raw.get("code") or "")
            if code not in _RELATION_MISMATCH_CODES:
                continue
            relation = str(raw.get("relation") or "")
            side = (
                "source"
                if code == "RELATION_SOURCE_TYPE_MISMATCH"
                else "target"
            )
            actual = raw.get("actual")
            actual_type = (
                str(actual.get("type") or "")
                if isinstance(actual, dict)
                else ""
            )
            expected = raw.get("expected")
            expected_types: tuple[str, ...] = ()
            if isinstance(expected, dict):
                values = expected.get("types")
                if isinstance(values, list):
                    expected_types = tuple(sorted(str(v) for v in values))
            counts[(code, relation, side, actual_type, expected_types)] += 1

    rows = [
        {
            "code": code,
            "relation": relation,
            "side": side,
            "actual_type": actual_type,
            "expected_types": list(expected_types),
            "count": count,
        }
        for (code, relation, side, actual_type, expected_types), count
        in counts.items()
    ]
    rows.sort(
        key=lambda row: (
            -int(row["count"]),
            str(row["relation"]),
            str(row["side"]),
            str(row["actual_type"]),
        )
    )
    return rows


def _isolated_node_patterns(run_dir: Path) -> list[dict[str, Any]]:
    """Summarize which node families repeatedly become disconnected."""
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = {}
    validation_dir = run_dir / "validation"
    if not validation_dir.exists():
        return []
    for path in sorted(validation_dir.glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        issues = payload.get("issues")
        if not isinstance(issues, list):
            continue
        for raw in issues:
            if not isinstance(raw, dict) or str(raw.get("code") or "") != "ISOLATED_NODE":
                continue
            collection = str(raw.get("node_collection") or "")
            actual = raw.get("actual")
            actual_type = ""
            if isinstance(actual, dict):
                actual_type = str(
                    actual.get("type")
                    or actual.get("node_type")
                    or actual.get("collection")
                    or ""
                )
            signature = (collection, actual_type)
            counts[signature] += 1
            node_id = str(raw.get("node_id") or "").strip()
            if node_id:
                bucket = examples.setdefault(signature, [])
                if node_id not in bucket and len(bucket) < 5:
                    bucket.append(node_id)
    rows = [
        {
            "node_collection": collection,
            "actual_type": actual_type,
            "count": count,
            "example_node_ids": examples.get((collection, actual_type), []),
        }
        for (collection, actual_type), count in counts.items()
    ]
    rows.sort(
        key=lambda row: (
            -int(row["count"]),
            str(row["actual_type"]),
            str(row["node_collection"]),
        )
    )
    return rows


def _recovery_reason_counts(
    records: Iterable[Mapping[str, Any]],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        for key in ("recovery_reason", "quarantine_reason_class", "error_type"):
            value = str(record.get(key) or "").strip()
            if value:
                counter[f"{key}:{value}"] += 1
    return counter


def _projection_summary(paper_root: Path) -> dict[str, Any] | None:
    return _read_json(
        paper_root / "graphagents" / "mechanism" / "summary.json"
    )


def inspect_broad_paper(
    *,
    data_root: str | Path,
    paper_id: str,
    preflight_outlier: bool = False,
) -> dict[str, Any]:
    data_root = Path(data_root)
    paper_root = data_root / "extracted" / paper_id
    run_dir = _latest_run_dir(paper_root)
    base: dict[str, Any] = {
        "paper_id": paper_id,
        "paper_root": str(paper_root),
        "run_found": run_dir is not None,
        "run_directory": str(run_dir) if run_dir is not None else None,
    }
    if preflight_outlier:
        return {
            **base,
            "historical_run_found": run_dir is not None,
            "run_found": False,
            "run_directory": None,
            "diagnostic_scope": "preflight_excluded_current_pipeline",
            "input_guard_status": "ABSTRACT_LENGTH_OUTLIER",
            "graph_materialization_status": "abstract_length_outlier",
            "graph_usable": False,
            "strict_complete": False,
            "active_chunk_count": 0,
            "quarantined_chunk_count": 0,
            "failed_chunk_count": 0,
            "source_token_coverage": None,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "call_kind_counts": {},
            "generation_attempts": 0,
            "patch_attempts": 0,
            "micro_reextract_attempts": 0,
            "post_micro_patch_attempts": 0,
            "patch_operation_count": 0,
            "terminal_validation_issue_counts": {},
            "observed_validation_issue_counts": {},
            "relation_mismatch_patterns": [],
            "isolated_node_patterns": [],
            "recovery_reason_counts": {
                "input_guard:ABSTRACT_LENGTH_OUTLIER": 1
            },
            "projection_found": False,
            "projection_current": False,
            "stale_projection_found": False,
            "projection_nodes": 0,
            "projection_edges": 0,
            "direct_mechanism_edges": 0,
            "mechanism_bearing": False,
        }
    if run_dir is None:
        base.update({
            "graph_materialization_status": "missing",
            "graph_usable": False,
            "strict_complete": False,
            "active_chunk_count": 0,
            "quarantined_chunk_count": 0,
            "failed_chunk_count": 0,
            "source_token_coverage": None,
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "call_kind_counts": {},
            "terminal_validation_issue_counts": {},
            "observed_validation_issue_counts": {},
            "relation_mismatch_patterns": [],
            "isolated_node_patterns": [],
            "recovery_reason_counts": {},
            "projection_found": False,
            "projection_current": False,
            "stale_projection_found": False,
            "projection_nodes": 0,
            "projection_edges": 0,
            "direct_mechanism_edges": 0,
            "mechanism_bearing": False,
        })
        return base

    active_payload = _read_json(run_dir / "active_chunks.json") or {}
    quality = active_payload.get("quality")
    if not isinstance(quality, dict):
        quality = _read_json(run_dir / "extraction_quality.json") or {}
    summary = _read_json(run_dir / "summary.json") or {}
    records = _terminal_records(active_payload)
    usage = _aggregate_attempt_usage(records)
    terminal_issues = _terminal_issue_counts(records)
    observed_issues = _observed_validation_issue_counts(run_dir)
    relation_mismatch_patterns = _relation_mismatch_patterns(run_dir)
    isolated_node_patterns = _isolated_node_patterns(run_dir)
    recovery_reasons = _recovery_reason_counts(records)
    recovery_budget = {
        # Generation count is derived from attempt_usages. Older terminal
        # records do not always persist generation_attempts explicitly.
        "generation_attempts": _int(
            usage["call_kind_counts"].get("graph_generation", 0)
        ),
        "patch_attempts": sum(_int(row.get("patch_attempts")) for row in records),
        "micro_reextract_attempts": sum(
            _int(row.get("micro_reextract_attempts")) for row in records
        ),
        "post_micro_patch_attempts": sum(
            _int(row.get("post_micro_patch_attempts")) for row in records
        ),
        "patch_operation_count": sum(
            _int(row.get("patch_operation_count")) for row in records
        ),
    }

    materialization = str(
        active_payload.get("graph_materialization_status")
        or quality.get("graph_materialization_status")
        or summary.get("graph_materialization_status")
        or "unknown"
    )
    graph_usable = materialization in {"complete", "partial_acceptable"}
    run_id = str(active_payload.get("run_id") or summary.get("run_id") or "")
    run_fingerprint = str(
        active_payload.get("run_fingerprint")
        or summary.get("run_fingerprint")
        or ""
    )
    projection = _projection_summary(paper_root)
    projection_source_run_id = str(
        projection.get("source_extraction_run_id") if projection else ""
    )
    projection_source_fingerprint = str(
        projection.get("source_extraction_run_fingerprint") if projection else ""
    )
    projection_current = bool(
        projection
        and run_id
        and projection_source_run_id == run_id
        and (
            not run_fingerprint
            or projection_source_fingerprint == run_fingerprint
        )
    )
    stale_projection_found = projection is not None and not projection_current
    direct_mechanism_edges = _int(
        projection.get("direct_mechanism_edges")
        if projection_current and projection
        else 0
    )

    result = {
        **base,
        "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "graph_materialization_status": materialization,
        "graph_usable": graph_usable,
        "strict_complete": bool(
            active_payload.get("complete")
            or quality.get("strict_complete")
            or summary.get("complete")
        ),
        "paper_status": str(active_payload.get("paper_status") or summary.get("paper_status") or ""),
        "active_chunk_count": _int(
            active_payload.get("active_chunk_count")
            or quality.get("active_chunk_count")
        ),
        "quarantined_chunk_count": len(
            active_payload.get("quarantined_chunks") or []
        ),
        "failed_chunk_count": len(active_payload.get("failed_chunks") or []),
        "source_token_coverage": _float(
            quality.get("source_token_coverage")
        ),
        "quarantine_tier_counts": dict(
            quality.get("quarantine_tier_counts") or {}
        ),
        **usage,
        **recovery_budget,
        "terminal_validation_issue_counts": _counter_dict(terminal_issues),
        "observed_validation_issue_counts": _counter_dict(observed_issues),
        "relation_mismatch_patterns": relation_mismatch_patterns,
        "isolated_node_patterns": isolated_node_patterns,
        "recovery_reason_counts": _counter_dict(recovery_reasons),
        "projection_found": projection is not None,
        "projection_current": projection_current,
        "stale_projection_found": stale_projection_found,
        "projection_source_extraction_run_id": projection_source_run_id,
        "projection_nodes": _int(
            projection.get("nodes") if projection_current and projection else 0
        ),
        "projection_edges": _int(
            projection.get("edges") if projection_current and projection else 0
        ),
        "direct_mechanism_edges": direct_mechanism_edges,
        "mechanism_bearing": direct_mechanism_edges > 0,
    }
    result["llm_calls_per_direct_mechanism_edge"] = (
        usage["llm_calls"] / direct_mechanism_edges
        if direct_mechanism_edges > 0
        else None
    )
    result["tokens_per_direct_mechanism_edge"] = (
        usage["total_tokens"] / direct_mechanism_edges
        if direct_mechanism_edges > 0
        else None
    )
    return result


def aggregate_broad_extraction_diagnostics(
    paper_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in paper_rows]
    status_counts: Counter[str] = Counter(
        str(row.get("graph_materialization_status") or "unknown")
        for row in rows
    )
    terminal_issues: Counter[str] = Counter()
    observed_issues: Counter[str] = Counter()
    recovery_reasons: Counter[str] = Counter()
    call_kinds: Counter[str] = Counter()
    relation_mismatch_counts: Counter[
        tuple[str, str, str, str, tuple[str, ...]]
    ] = Counter()
    isolated_node_counts: Counter[tuple[str, str]] = Counter()
    isolated_examples: dict[tuple[str, str], list[str]] = {}

    for row in rows:
        for code, count in dict(
            row.get("terminal_validation_issue_counts") or {}
        ).items():
            terminal_issues[str(code)] += _int(count)
        for code, count in dict(
            row.get("observed_validation_issue_counts") or {}
        ).items():
            observed_issues[str(code)] += _int(count)
        for key, count in dict(row.get("recovery_reason_counts") or {}).items():
            recovery_reasons[str(key)] += _int(count)
        for key, count in dict(row.get("call_kind_counts") or {}).items():
            call_kinds[str(key)] += _int(count)
        for pattern in row.get("relation_mismatch_patterns") or []:
            if not isinstance(pattern, dict):
                continue
            signature = (
                str(pattern.get("code") or ""),
                str(pattern.get("relation") or ""),
                str(pattern.get("side") or ""),
                str(pattern.get("actual_type") or ""),
                tuple(
                    sorted(
                        str(value)
                        for value in (pattern.get("expected_types") or [])
                    )
                ),
            )
            relation_mismatch_counts[signature] += _int(pattern.get("count"))
        for pattern in row.get("isolated_node_patterns") or []:
            if not isinstance(pattern, dict):
                continue
            signature = (
                str(pattern.get("node_collection") or ""),
                str(pattern.get("actual_type") or ""),
            )
            isolated_node_counts[signature] += _int(pattern.get("count"))
            bucket = isolated_examples.setdefault(signature, [])
            for node_id in pattern.get("example_node_ids") or []:
                value = str(node_id)
                if value and value not in bucket and len(bucket) < 5:
                    bucket.append(value)

    run_found = sum(bool(row.get("run_found")) for row in rows)
    usable = sum(bool(row.get("graph_usable")) for row in rows)
    mechanism_bearing = sum(bool(row.get("mechanism_bearing")) for row in rows)
    projected = sum(bool(row.get("projection_current")) for row in rows)
    stale_projection_ids = [
        str(row.get("paper_id"))
        for row in rows
        if row.get("stale_projection_found")
    ]
    preflight_outlier_ids = [
        str(row.get("paper_id"))
        for row in rows
        if str(row.get("input_guard_status") or "") == "ABSTRACT_LENGTH_OUTLIER"
    ]
    total_calls = sum(_int(row.get("llm_calls")) for row in rows)
    input_tokens = sum(_int(row.get("input_tokens")) for row in rows)
    output_tokens = sum(_int(row.get("output_tokens")) for row in rows)
    total_tokens = sum(_int(row.get("total_tokens")) for row in rows)
    direct_mechanism_edges = sum(
        _int(row.get("direct_mechanism_edges")) for row in rows
    )
    rejected_paper_ids = [
        str(row.get("paper_id"))
        for row in rows
        if str(row.get("graph_materialization_status")) == "rejected"
    ]
    most_expensive_papers = [
        {
            "paper_id": str(row.get("paper_id")),
            "graph_materialization_status": str(
                row.get("graph_materialization_status") or "unknown"
            ),
            "llm_calls": _int(row.get("llm_calls")),
            "total_tokens": _int(row.get("total_tokens")),
            "direct_mechanism_edges": _int(row.get("direct_mechanism_edges")),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                -_int(item.get("total_tokens")),
                -_int(item.get("llm_calls")),
                str(item.get("paper_id")),
            ),
        )[:10]
    ]
    generation_attempts = sum(_int(row.get("generation_attempts")) for row in rows)
    patch_attempts = sum(_int(row.get("patch_attempts")) for row in rows)
    micro_reextract_attempts = sum(
        _int(row.get("micro_reextract_attempts")) for row in rows
    )
    post_micro_patch_attempts = sum(
        _int(row.get("post_micro_patch_attempts")) for row in rows
    )
    patch_operation_count = sum(
        _int(row.get("patch_operation_count")) for row in rows
    )
    wasted_rows = [
        row
        for row in rows
        if row.get("run_found") and not row.get("graph_usable")
    ]
    wasted_llm_calls = sum(_int(row.get("llm_calls")) for row in wasted_rows)
    wasted_tokens = sum(_int(row.get("total_tokens")) for row in wasted_rows)
    relation_mismatch_patterns = [
        {
            "code": code,
            "relation": relation,
            "side": side,
            "actual_type": actual_type,
            "expected_types": list(expected_types),
            "count": count,
        }
        for (code, relation, side, actual_type, expected_types), count
        in relation_mismatch_counts.items()
    ]
    relation_mismatch_patterns.sort(
        key=lambda row: (
            -int(row["count"]),
            str(row["relation"]),
            str(row["side"]),
            str(row["actual_type"]),
        )
    )
    isolated_node_patterns = [
        {
            "node_collection": collection,
            "actual_type": actual_type,
            "count": count,
            "example_node_ids": isolated_examples.get(
                (collection, actual_type), []
            ),
        }
        for (collection, actual_type), count in isolated_node_counts.items()
    ]
    isolated_node_patterns.sort(
        key=lambda row: (
            -int(row["count"]),
            str(row["actual_type"]),
            str(row["node_collection"]),
        )
    )

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "requested_paper_count": len(rows),
        "run_found_count": run_found,
        "run_missing_count": len(rows) - run_found,
        "graph_usable_paper_count": usable,
        "graph_usable_paper_fraction": usable / len(rows) if rows else 0.0,
        "projection_paper_count": projected,
        "stale_projection_count": len(stale_projection_ids),
        "stale_projection_paper_ids": stale_projection_ids,
        "preflight_outlier_count": len(preflight_outlier_ids),
        "preflight_outlier_paper_ids": preflight_outlier_ids,
        "mechanism_bearing_paper_count": mechanism_bearing,
        "mechanism_bearing_fraction_of_requested": (
            mechanism_bearing / len(rows) if rows else 0.0
        ),
        "mechanism_bearing_fraction_of_usable": (
            mechanism_bearing / usable if usable else 0.0
        ),
        "materialization_status_counts": _counter_dict(status_counts),
        "rejected_paper_ids": rejected_paper_ids,
        "most_expensive_papers": most_expensive_papers,
        "llm_calls": total_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "call_kind_counts": _counter_dict(call_kinds),
        "wasted_llm_calls": wasted_llm_calls,
        "wasted_call_fraction": (
            wasted_llm_calls / total_calls if total_calls else 0.0
        ),
        "wasted_tokens": wasted_tokens,
        "wasted_token_fraction": (
            wasted_tokens / total_tokens if total_tokens else 0.0
        ),
        "generation_attempts": generation_attempts,
        "patch_attempts": patch_attempts,
        "micro_reextract_attempts": micro_reextract_attempts,
        "post_micro_patch_attempts": post_micro_patch_attempts,
        "patch_operation_count": patch_operation_count,
        "llm_calls_per_requested_paper": (
            total_calls / len(rows) if rows else 0.0
        ),
        "llm_calls_per_usable_paper": (
            total_calls / usable if usable else None
        ),
        "tokens_per_usable_paper": (
            total_tokens / usable if usable else None
        ),
        "direct_mechanism_edges": direct_mechanism_edges,
        "llm_calls_per_direct_mechanism_edge": (
            total_calls / direct_mechanism_edges
            if direct_mechanism_edges
            else None
        ),
        "tokens_per_direct_mechanism_edge": (
            total_tokens / direct_mechanism_edges
            if direct_mechanism_edges
            else None
        ),
        "terminal_validation_issue_counts": _counter_dict(terminal_issues),
        "observed_validation_issue_counts": _counter_dict(observed_issues),
        "relation_mismatch_patterns": relation_mismatch_patterns,
        "isolated_node_patterns": isolated_node_patterns,
        "recovery_reason_counts": _counter_dict(recovery_reasons),
    }


def write_broad_extraction_diagnostics(
    *,
    data_root: str | Path,
    paper_ids: Iterable[str],
    output_dir: str | Path,
    preflight_outlier_ids: Iterable[str] = (),
) -> tuple[Path, Path, Path]:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_ids = {str(value) for value in preflight_outlier_ids}
    rows = [
        inspect_broad_paper(
            data_root=data_root,
            paper_id=str(paper_id),
            preflight_outlier=str(paper_id) in outlier_ids,
        )
        for paper_id in paper_ids
    ]
    report = aggregate_broad_extraction_diagnostics(rows)
    report["data_root"] = str(data_root)
    report["paper_ids"] = [str(row["paper_id"]) for row in rows]

    report_path = output_dir / "extraction_diagnostics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows_path = output_dir / "paper_diagnostics.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    issue_path = output_dir / "validation_issue_counts.json"
    issue_path.write_text(
        json.dumps(
            {
                "terminal_validation_issue_counts": report[
                    "terminal_validation_issue_counts"
                ],
                "observed_validation_issue_counts": report[
                    "observed_validation_issue_counts"
                ],
                "relation_mismatch_patterns": report[
                    "relation_mismatch_patterns"
                ],
                "isolated_node_patterns": report["isolated_node_patterns"],
                "recovery_reason_counts": report["recovery_reason_counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path, rows_path, issue_path
