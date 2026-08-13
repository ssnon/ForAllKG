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

    by_stage_internal: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "provider_cost_credits": 0.0,
            "provider_cost_observations": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "cache_detail_observations": 0,
            "cache_observed_input_tokens": 0,
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
    total_cost_credits = 0.0
    cost_observations = 0
    total_cached_input_tokens = 0
    total_cache_write_tokens = 0
    cache_detail_observations = 0
    cache_observed_input_tokens = 0
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

        provider_cost = event.get("provider_cost_credits")
        if provider_cost is not None:
            try:
                cost_value = float(provider_cost)
            except (TypeError, ValueError):
                cost_value = None
            if cost_value is not None:
                total_cost_credits += cost_value
                cost_observations += 1
                bucket["provider_cost_credits"] += cost_value
                bucket["provider_cost_observations"] += 1

        cached_raw = event.get("provider_cached_input_tokens")
        cache_write_raw = event.get("provider_cache_write_tokens")
        # Missing cache detail is not equivalent to a measured zero. Coverage
        # is tracked explicitly so old/non-supporting provider rows do not make
        # cache hit rates look artificially low.
        if cached_raw is not None or cache_write_raw is not None:
            try:
                cached_tokens = int(cached_raw or 0)
            except (TypeError, ValueError):
                cached_tokens = 0
            try:
                cache_write_tokens = int(cache_write_raw or 0)
            except (TypeError, ValueError):
                cache_write_tokens = 0
            total_cached_input_tokens += cached_tokens
            total_cache_write_tokens += cache_write_tokens
            cache_detail_observations += 1
            cache_observed_input_tokens += input_tokens
            bucket["cached_input_tokens"] += cached_tokens
            bucket["cache_write_tokens"] += cache_write_tokens
            bucket["cache_detail_observations"] += 1
            bucket["cache_observed_input_tokens"] += input_tokens

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
        stage_cost_observations = int(values["provider_cost_observations"])
        stage_cache_observations = int(values["cache_detail_observations"])
        stage_cost = float(values["provider_cost_credits"])
        stage_cache_input = int(values["cache_observed_input_tokens"])
        by_stage[key] = {
            "calls": values["calls"],
            "input_tokens": values["input_tokens"],
            "output_tokens": values["output_tokens"],
            "total_tokens": values["total_tokens"],
            "provider_cost_credits": (
                stage_cost if stage_cost_observations else None
            ),
            "provider_cost_observations": stage_cost_observations,
            "provider_cost_coverage_fraction": (
                stage_cost_observations / values["calls"]
                if values["calls"]
                else None
            ),
            "cost_per_observed_call_credits": (
                stage_cost / stage_cost_observations
                if stage_cost_observations
                else None
            ),
            "cost_share_of_observed_total": (
                stage_cost / total_cost_credits
                if stage_cost_observations and total_cost_credits > 0
                else None
            ),
            "provider_cached_input_tokens": (
                int(values["cached_input_tokens"])
                if stage_cache_observations
                else None
            ),
            "provider_cache_write_tokens": (
                int(values["cache_write_tokens"])
                if stage_cache_observations
                else None
            ),
            "cache_detail_observations": stage_cache_observations,
            "cache_detail_coverage_fraction": (
                stage_cache_observations / values["calls"]
                if values["calls"]
                else None
            ),
            "cache_read_fraction_of_observed_input": (
                int(values["cached_input_tokens"]) / stage_cache_input
                if stage_cache_input > 0
                else None
            ),
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

    # "Query-side" here means LLM stages outside the extraction pipeline.
    # It is intentionally a reporting view only; no stage is skipped or reordered.
    query_stage_keys = [
        key
        for key in by_stage
        if key.split(":", 1)[0] not in {"extraction", "unknown"}
    ]
    query_calls = sum(int(by_stage[key]["calls"]) for key in query_stage_keys)
    query_input = sum(int(by_stage[key]["input_tokens"]) for key in query_stage_keys)
    query_output = sum(int(by_stage[key]["output_tokens"]) for key in query_stage_keys)
    query_total = sum(int(by_stage[key]["total_tokens"]) for key in query_stage_keys)
    query_cost_observations = sum(
        int(by_stage[key]["provider_cost_observations"])
        for key in query_stage_keys
    )
    query_cost = sum(
        float(by_stage[key]["provider_cost_credits"] or 0.0)
        for key in query_stage_keys
    )
    query_cache_observations = sum(
        int(by_stage[key]["cache_detail_observations"])
        for key in query_stage_keys
    )
    query_cached_tokens = sum(
        int(by_stage[key]["provider_cached_input_tokens"] or 0)
        for key in query_stage_keys
    )
    query_cache_write_tokens = sum(
        int(by_stage[key]["provider_cache_write_tokens"] or 0)
        for key in query_stage_keys
    )
    query_cache_observed_input = sum(
        int(by_stage_internal[key]["cache_observed_input_tokens"])
        for key in query_stage_keys
    )
    query_by_stage: dict[str, dict[str, Any]] = {}
    for key in query_stage_keys:
        row = dict(by_stage[key])
        row["cost_share_of_observed_query_cost"] = (
            float(row["provider_cost_credits"] or 0.0) / query_cost
            if row["provider_cost_observations"] and query_cost > 0
            else None
        )
        query_by_stage[key] = row

    query_stage_economics = {
        "calls": query_calls,
        "provider_input_tokens": query_input,
        "provider_output_tokens": query_output,
        "provider_total_tokens": query_total,
        "provider_cost_credits": (
            query_cost if query_cost_observations else None
        ),
        "provider_cost_observations": query_cost_observations,
        "provider_cost_coverage_fraction": (
            query_cost_observations / query_calls if query_calls else None
        ),
        "provider_cached_input_tokens": (
            query_cached_tokens if query_cache_observations else None
        ),
        "provider_cache_write_tokens": (
            query_cache_write_tokens if query_cache_observations else None
        ),
        "cache_detail_observations": query_cache_observations,
        "cache_detail_coverage_fraction": (
            query_cache_observations / query_calls if query_calls else None
        ),
        "cache_read_fraction_of_observed_input": (
            query_cached_tokens / query_cache_observed_input
            if query_cache_observed_input > 0
            else None
        ),
        "by_stage": query_by_stage,
    }

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
        "provider_cost_credits": (
            total_cost_credits if cost_observations else None
        ),
        "provider_cost_observations": cost_observations,
        "provider_cost_coverage_fraction": (
            cost_observations / len(rows) if rows else None
        ),
        "provider_cached_input_tokens": (
            total_cached_input_tokens if cache_detail_observations else None
        ),
        "provider_cache_write_tokens": (
            total_cache_write_tokens if cache_detail_observations else None
        ),
        "cache_detail_observations": cache_detail_observations,
        "cache_detail_coverage_fraction": (
            cache_detail_observations / len(rows) if rows else None
        ),
        "cache_read_fraction_of_observed_input": (
            total_cached_input_tokens / cache_observed_input_tokens
            if cache_observed_input_tokens > 0
            else None
        ),
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
        "query_stage_economics": query_stage_economics,
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
