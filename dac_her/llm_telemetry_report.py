from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dac_her.llm_telemetry import normalize_stage_name


def load_usage_events(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_resolution(row: dict[str, Any]) -> bool:
    return (
        row.get("record_type") == "artifact_resolution"
        or str(row.get("schema_version") or "").startswith(
            "llm-artifact-resolution-"
        )
    )


def summarize_usage_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    raw_rows = list(events)
    resolutions: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if _is_resolution(row):
            call_id = str(row.get("call_id") or "")
            if call_id:
                # Ledger is append-only. Last resolution wins for a call ID.
                resolutions[call_id] = row
            continue
        rows.append(row)

    by_stage_internal: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "source_bearing_calls": 0,
            "source_bearing_input_tokens": 0,
            "estimated_source_tokens": 0,
        }
    )
    repeated: dict[tuple[str, str], dict[str, Any]] = {}
    component_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"observations": 0, "estimated_tokens": 0}
    )
    pipelines = Counter()
    provider_usage_scopes = Counter()
    call_outcomes = Counter()
    artifact_outcomes = Counter()
    terminal_contributions = Counter()
    response_models = Counter()

    total_input = total_output = total_tokens = 0
    source_scoped_input = 0
    source_scoped_source_estimate = 0
    source_scoped_calls = 0
    gaps: list[int] = []
    resolved_call_count = 0

    for event in rows:
        input_tokens = int(event.get("provider_input_tokens") or 0)
        output_tokens = int(event.get("provider_output_tokens") or 0)
        provider_total = int(event.get("provider_total_tokens") or 0)
        total_input += input_tokens
        total_output += output_tokens
        total_tokens += provider_total
        if event.get("token_estimate_gap") is not None:
            gaps.append(int(event["token_estimate_gap"]))

        pipeline = str(event.get("pipeline") or "unknown")
        response_model = str(event.get("response_model") or "") or None
        stage = normalize_stage_name(
            str(event.get("stage") or event.get("call_kind") or "unknown"),
            response_model=response_model,
        ) or "unknown"
        pipelines[pipeline] += 1
        provider_usage_scopes[
            str(event.get("provider_usage_scope") or "unknown")
        ] += 1
        call_outcome = str(
            event.get("call_outcome")
            or event.get("outcome")  # backwards compatibility with v1 rows
            or "unknown"
        )
        call_outcomes[call_outcome] += 1
        if response_model:
            response_models[response_model] += 1

        call_id = str(event.get("call_id") or "")
        resolution = resolutions.get(call_id) if call_id else None
        if resolution is not None:
            resolved_call_count += 1
            artifact_outcome = str(
                resolution.get("artifact_outcome") or "unknown"
            )
            terminal_contribution = str(
                resolution.get("terminal_contribution") or "unknown"
            )
        else:
            artifact_outcome = "unknown"
            terminal_contribution = "unknown"
        artifact_outcomes[artifact_outcome] += 1
        terminal_contributions[terminal_contribution] += 1

        bucket = by_stage_internal[f"{pipeline}:{stage}"]
        bucket["calls"] += 1
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["total_tokens"] += provider_total

        components = event.get("estimated_components") or {}
        source_component = components.get("source")
        if isinstance(source_component, dict):
            source_estimate = int(source_component.get("estimated_tokens") or 0)
            source_scoped_calls += 1
            source_scoped_input += input_tokens
            source_scoped_source_estimate += source_estimate
            bucket["source_bearing_calls"] += 1
            bucket["source_bearing_input_tokens"] += input_tokens
            bucket["estimated_source_tokens"] += source_estimate

        for name, component in components.items():
            if not isinstance(component, dict):
                continue
            estimated_tokens = int(component.get("estimated_tokens") or 0)
            component_totals[str(name)]["observations"] += 1
            component_totals[str(name)]["estimated_tokens"] += estimated_tokens

            fingerprint = component.get("fingerprint")
            if not fingerprint:
                continue
            key = (str(name), str(fingerprint))
            entry = repeated.setdefault(
                key,
                {
                    "calls": 0,
                    "estimated_tokens_per_call": estimated_tokens,
                    "counted_in_estimated_sum": bool(
                        component.get("counted_in_estimated_sum", False)
                    ),
                },
            )
            entry["calls"] += 1

    repeated_serialization = [
        {
            "component": name,
            "fingerprint": fingerprint,
            **values,
            "estimated_serialized_tokens": (
                values["calls"] * values["estimated_tokens_per_call"]
            ),
        }
        for (name, fingerprint), values in repeated.items()
        if values["calls"] > 1
    ]
    repeated_serialization.sort(
        key=lambda row: row["estimated_serialized_tokens"],
        reverse=True,
    )

    component_estimates = {
        name: {
            **values,
            "share_of_provider_input": (
                values["estimated_tokens"] / total_input
                if total_input > 0
                else None
            ),
        }
        for name, values in sorted(component_totals.items())
    }

    by_stage: dict[str, dict[str, Any]] = {}
    source_overhead_by_stage: dict[str, dict[str, Any]] = {}
    for key, values in sorted(by_stage_internal.items()):
        source_tokens = values["estimated_source_tokens"]
        source_input = values["source_bearing_input_tokens"]
        ratio = source_input / source_tokens if source_tokens > 0 else None
        by_stage[key] = {
            "calls": values["calls"],
            "input_tokens": values["input_tokens"],
            "output_tokens": values["output_tokens"],
            "total_tokens": values["total_tokens"],
            "source_bearing_calls": values["source_bearing_calls"],
            "estimated_source_tokens": source_tokens,
            "provider_input_to_estimated_source_ratio": ratio,
        }
        if values["source_bearing_calls"]:
            source_overhead_by_stage[key] = {
                "calls": values["source_bearing_calls"],
                "provider_input_tokens": source_input,
                "estimated_source_tokens": source_tokens,
                "provider_input_to_estimated_source_ratio": ratio,
            }

    graph_generation_key = "extraction:graph_generation"
    graph_generation_source_overhead = source_overhead_by_stage.get(
        graph_generation_key
    )

    return {
        "schema_version": "llm-telemetry-summary-v1.1",
        "calls": len(rows),
        "artifact_resolution_records": sum(
            1 for row in raw_rows if _is_resolution(row)
        ),
        "resolved_calls": resolved_call_count,
        "unresolved_calls": len(rows) - resolved_call_count,
        "provider_input_tokens": total_input,
        "provider_output_tokens": total_output,
        "provider_total_tokens": total_tokens,
        "pipeline_call_counts": dict(sorted(pipelines.items())),
        "provider_usage_scope_counts": dict(sorted(provider_usage_scopes.items())),
        "call_outcome_counts": dict(sorted(call_outcomes.items())),
        # Compatibility alias for existing consumers of v1 summaries.
        "outcome_counts": dict(sorted(call_outcomes.items())),
        "artifact_outcome_counts": dict(sorted(artifact_outcomes.items())),
        "terminal_contribution_counts": dict(
            sorted(terminal_contributions.items())
        ),
        "response_model_counts": dict(sorted(response_models.items())),
        "by_stage": by_stage,
        "source_overhead_by_stage": source_overhead_by_stage,
        "graph_generation_source_overhead": graph_generation_source_overhead,
        "component_estimates": component_estimates,
        "provider_input_to_estimated_source_ratio": (
            source_scoped_input / source_scoped_source_estimate
            if source_scoped_source_estimate > 0
            else None
        ),
        "source_scoped_overhead": {
            "calls": source_scoped_calls,
            "provider_input_tokens": source_scoped_input,
            "estimated_source_tokens": source_scoped_source_estimate,
        },
        "token_estimate_gap": {
            "observations": len(gaps),
            "mean": (sum(gaps) / len(gaps) if gaps else None),
            "min": (min(gaps) if gaps else None),
            "max": (max(gaps) if gaps else None),
        },
        # Component rows can overlap (e.g. user_prompt contains source). The
        # counted_in_estimated_sum flag identifies additive top-level surfaces.
        "repeated_serialization": repeated_serialization,
    }


def summarize_usage_file(path: str | Path) -> dict[str, Any]:
    return summarize_usage_events(load_usage_events(path))
