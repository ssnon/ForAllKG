from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx


ALPHA4C5G1_ATTRIBUTION_ID = (
    "trend_candidate_rejection_attribution_v1_alpha4c5g1"
)
EXPECTED_5G_DIAGNOSTIC_ID = (
    "trend_yield_recall_diagnostic_v1_alpha4c5g"
)
EXPECTED_5G_SUMMARY_SEMANTIC_SHA256 = (
    "2a27f624347362a279f3d94948c89dd9fcc4b186401d0f39cfd6f69f639dc2be"
)
EXPECTED_TREND_IMPLEMENTATION_SHA256 = (
    "b3834b8daaeffbf537866d217e9957f987adb26749637429403174661e076770"
)

REQUIRED_TREND_HELPERS = (
    "_claim_control",
    "_claim_response",
    "_claim_direction_shape",
    "_control_key_from_name",
    "_measurement_control",
    "_lineage",
    "_methods_compatible",
)

CLAIM_ADMITTED_BASES = frozenset(
    {
        "reported_directional_claim",
        "reported_correlation",
    }
)
NUMERIC_ADMITTED_BASES = frozenset(
    {
        "controlled_numeric_pair",
        "controlled_numeric_series",
    }
)

PROVENANCE_KEYS = (
    "source_chunk_id",
    "chunk_id",
    "source_id",
    "source_document_id",
    "source_page",
    "page",
    "locator",
    "source_locator",
    "evidence_status",
    "requires_verification",
    "subject_id",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sample_score(*parts: object) -> str:
    raw = "\0".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected JSON object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def require_trend_helper_contract(
    trend_module: ModuleType,
) -> None:
    missing = [
        name
        for name in REQUIRED_TREND_HELPERS
        if not callable(getattr(trend_module, name, None))
    ]
    if missing:
        raise RuntimeError(
            "Current Trend implementation does not expose the "
            "helper contract required for exact rejection attribution: "
            + ", ".join(missing)
        )


def _serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_serializable(item) for item in value]
    return repr(value)


def _selected_provenance(
    attrs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: _serializable(attrs[key])
        for key in PROVENANCE_KEYS
        if key in attrs
        and str(attrs.get(key, "")).strip()
    }


def _claim_current_normalization(
    *,
    text: str,
    trend_module: ModuleType,
) -> tuple[Any, Any, Any]:
    control = trend_module._claim_control(text)
    if control is None:
        return None, None, None
    response = trend_module._claim_response(text)
    if response is None:
        return control, None, None
    control_key = (
        control[0]
        if isinstance(control, (tuple, list))
        and control
        else str(control)
    )
    direction_shape = trend_module._claim_direction_shape(
        control_key,
        text,
    )
    return control, response, direction_shape


def attribute_claim_miss(
    *,
    candidate: Mapping[str, Any],
    graph: nx.Graph,
    trend_module: ModuleType,
) -> dict[str, Any]:
    paper_id = str(candidate["paper_id"])
    claim_id = str(candidate["claim_id"])
    text = str(candidate.get("text", "")).strip()
    if bool(candidate.get("admitted_by_current_trend")):
        raise ValueError(
            "attribute_claim_miss received an admitted claim."
        )
    if claim_id not in graph:
        return {
            **dict(candidate),
            "attribution_kind": "claim",
            "primary_reason": "claim_node_missing_from_frozen_graph",
            "current_control": None,
            "current_response": None,
            "current_direction_shape": None,
            "provenance": {},
        }

    attrs = dict(graph.nodes[claim_id])
    control, response, direction_shape = (
        _claim_current_normalization(
            text=text,
            trend_module=trend_module,
        )
    )

    if control is None:
        reason = "claim_control_not_normalized"
    elif response is None:
        reason = "claim_response_not_normalized"
    elif direction_shape is None:
        reason = "claim_direction_not_normalized"
    else:
        # In the current SERS Trend adapter, the claim lane is admitted
        # after these normalizers plus metadata annotation. If the frozen
        # output still lacks the claim, preserve it as unexplained rather
        # than inventing a reason.
        reason = "claim_unexplained_post_normalization_miss"

    return {
        **dict(candidate),
        "attribution_kind": "claim",
        "primary_reason": reason,
        "current_control": _serializable(control),
        "current_response": _serializable(response),
        "current_direction_shape": _serializable(
            direction_shape
        ),
        "provenance": _selected_provenance(attrs),
        "requires_human_adjudication": True,
    }


def _identity_by_representative(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(
            row.get("representative_measurement_id", "")
        ).strip()
        if key:
            result[key] = row
    return result


def _methods_by_id(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("method_context_id", "")): row
        for row in rows
        if str(row.get("method_context_id", "")).strip()
    }


def _contexts_by_measurement(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        measurement_id = str(
            row.get("measurement_id", "")
        ).strip()
        if measurement_id:
            result[measurement_id].append(row)
    for measurement_id in result:
        result[measurement_id] = sorted(
            result[measurement_id],
            key=lambda row: (
                str(row.get("observable_key", "")),
                str(row.get("unit", "")),
                str(row.get("context_id", "")),
            ),
        )
    return dict(result)


def _choose_context(
    *,
    rows: Sequence[Mapping[str, Any]],
    observable_key: str,
    dependent_unit: str,
) -> tuple[Mapping[str, Any] | None, str]:
    exact = [
        row
        for row in rows
        if str(row.get("observable_key", "")).strip()
        == observable_key
        and str(row.get("unit", "")).strip()
        == dependent_unit
    ]
    if len(exact) == 1:
        return exact[0], ""
    if len(exact) > 1:
        return exact[0], "multiple_exact_comparison_contexts"

    observable = [
        row
        for row in rows
        if str(row.get("observable_key", "")).strip()
        == observable_key
    ]
    if len(observable) == 1:
        return observable[0], "dependent_unit_context_mismatch"
    if len(observable) > 1:
        return observable[0], "multiple_observable_contexts"
    return None, "comparison_context_missing"


def _current_method_compatible(
    trend_module: ModuleType,
    method_rows: list[Mapping[str, Any]],
) -> bool:
    # Call exactly as the frozen adapter's numeric lane traditionally
    # calls the helper: no diagnostic exemption is injected here.
    result = trend_module._methods_compatible(method_rows)
    return bool(result)


def _method_guard_attribution(
    *,
    current_compatible: bool,
    candidate: Mapping[str, Any],
) -> str:
    if current_compatible:
        return ""
    if (
        bool(
            candidate.get(
                "method_compatible_except_varied_dimension",
                False,
            )
        )
        and str(
            candidate.get("varied_method_dimension", "")
        ).strip()
    ):
        return "numeric_varied_dimension_blocked_by_method_guard"
    return "numeric_method_incompatible_other_dimension"


def attribute_numeric_miss(
    *,
    candidate: Mapping[str, Any],
    graph: nx.Graph,
    trend_module: ModuleType,
    identity_rows: list[Mapping[str, Any]],
    method_rows: list[Mapping[str, Any]],
    comparison_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if bool(candidate.get("admitted_by_current_trend")):
        raise ValueError(
            "attribute_numeric_miss received an admitted candidate."
        )

    observable_key = str(
        candidate.get("observable_key", "")
    ).strip()
    condition_name = str(
        candidate.get("condition_name", "")
    ).strip()
    dependent_unit = str(
        candidate.get("dependent_unit", "")
    ).strip()
    measurement_ids = [
        str(value)
        for value in candidate.get("measurement_ids", [])
        if str(value).strip()
    ]

    supported_responses = set(
        getattr(
            trend_module,
            "_NUMERIC_RESPONSE_KEYS",
            (),
        )
    )
    current_control_key = trend_module._control_key_from_name(
        condition_name
    )

    identities = _identity_by_representative(
        identity_rows
    )
    methods = _methods_by_id(method_rows)
    contexts = _contexts_by_measurement(
        comparison_rows
    )

    detail_rows: list[dict[str, Any]] = []
    source_binding_issue = ""
    current_controls: list[Any] = []
    current_lineages: list[Any] = []
    current_methods: list[Mapping[str, Any]] = []
    source_expressions: set[str] = set()

    for measurement_id in measurement_ids:
        identity_row = identities.get(measurement_id)
        context_row, context_issue = _choose_context(
            rows=contexts.get(measurement_id, []),
            observable_key=observable_key,
            dependent_unit=dependent_unit,
        )
        method_row = None
        if context_row is not None:
            method_id = str(
                context_row.get("method_context_id", "")
            ).strip()
            method_row = methods.get(method_id)
        else:
            method_id = ""

        if identity_row is None:
            issue = "measurement_identity_missing"
        elif context_row is None:
            issue = context_issue
        elif method_row is None:
            issue = "method_context_missing"
        else:
            issue = context_issue

        control = None
        lineage = None
        if (
            identity_row is not None
            and context_row is not None
            and method_row is not None
        ):
            control = trend_module._measurement_control(
                graph,
                measurement_id,
                identity_row,
            )
            lineage = trend_module._lineage(
                graph,
                measurement_id,
                identity_row,
                method_row,
                context_row,
            )
            current_methods.append(method_row)
            expression = str(
                context_row.get("source_expression", "")
            ).strip()
            if expression:
                source_expressions.add(expression)

        if control is not None:
            current_controls.append(control)
        if lineage is not None:
            current_lineages.append(lineage)

        if issue and not source_binding_issue:
            source_binding_issue = issue

        detail_rows.append(
            {
                "measurement_id": measurement_id,
                "context_id": (
                    str(
                        context_row.get("context_id", "")
                    )
                    if context_row is not None
                    else ""
                ),
                "method_context_id": method_id,
                "source_binding_issue": issue,
                "current_measurement_control": (
                    _serializable(control)
                ),
                "current_lineage": _serializable(lineage),
            }
        )

    current_control_keys: list[str] = []
    current_control_values: list[float] = []
    for control in current_controls:
        if isinstance(control, (tuple, list)) and control:
            current_control_keys.append(str(control[0]))
            if len(control) >= 3:
                try:
                    value = float(control[2])
                except (TypeError, ValueError):
                    value = math.nan
                if math.isfinite(value):
                    current_control_values.append(value)

    lineage_keys = {
        canonical_json(_serializable(value))
        for value in current_lineages
    }

    current_method_compatible = (
        _current_method_compatible(
            trend_module,
            current_methods,
        )
        if current_methods
        else False
    )
    method_reason = _method_guard_attribution(
        current_compatible=current_method_compatible,
        candidate=candidate,
    )

    if (
        supported_responses
        and observable_key not in supported_responses
    ):
        reason = "numeric_response_not_supported"
    elif not str(current_control_key).strip():
        reason = "numeric_control_name_not_normalized"
    elif source_binding_issue:
        reason = "numeric_source_binding_ambiguous_or_missing"
    elif len(current_controls) != len(measurement_ids):
        reason = "numeric_measurement_control_unresolved"
    elif (
        len(set(current_control_keys)) != 1
        or current_control_keys[0]
        != str(current_control_key)
    ):
        reason = "numeric_control_family_mismatch"
    elif len(set(current_control_values)) < 2:
        reason = "numeric_current_control_not_varied"
    elif len(current_lineages) != len(measurement_ids):
        reason = "numeric_current_lineage_unresolved"
    elif len(lineage_keys) != 1:
        reason = "numeric_current_lineage_split"
    elif method_reason:
        reason = method_reason
    else:
        reason = "numeric_unexplained_post_gate_miss"

    return {
        **dict(candidate),
        "attribution_kind": "numeric",
        "primary_reason": reason,
        "current_supported_numeric_response": (
            observable_key in supported_responses
            if supported_responses
            else None
        ),
        "current_control_key_from_name": str(
            current_control_key or ""
        ),
        "current_measurement_controls": _serializable(
            current_controls
        ),
        "current_control_distinct_value_count": len(
            set(current_control_values)
        ),
        "current_lineage_distinct_count": len(
            lineage_keys
        ),
        "current_method_compatible": (
            current_method_compatible
        ),
        "measurement_attribution": detail_rows,
        "source_expressions": sorted(source_expressions),
        "requires_human_adjudication": True,
    }


def build_stratified_sample(
    *,
    claim_rows: list[Mapping[str, Any]],
    numeric_rows: list[Mapping[str, Any]],
    per_bucket: int,
) -> list[dict[str, Any]]:
    if per_bucket <= 0:
        raise ValueError("per_bucket must be positive.")

    buckets: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    for row in claim_rows:
        buckets[
            ("claim", str(row["primary_reason"]))
        ].append(row)
    for row in numeric_rows:
        buckets[
            ("numeric", str(row["primary_reason"]))
        ].append(row)

    sample: list[dict[str, Any]] = []
    for (kind, reason), rows in sorted(
        buckets.items()
    ):
        ranked = sorted(
            rows,
            key=lambda row: stable_sample_score(
                ALPHA4C5G1_ATTRIBUTION_ID,
                kind,
                reason,
                row.get("paper_id", ""),
                row.get("claim_id", ""),
                row.get("condition_name", ""),
                row.get("observable_key", ""),
                ",".join(
                    str(value)
                    for value in row.get(
                        "measurement_ids",
                        [],
                    )
                ),
            ),
        )
        for rank, row in enumerate(
            ranked[:per_bucket],
            start=1,
        ):
            sample.append(
                {
                    "kind": kind,
                    "primary_reason": reason,
                    "sample_rank_within_reason": rank,
                    "paper_id": row.get("paper_id", ""),
                    "claim_id": row.get("claim_id", ""),
                    "condition_name": row.get(
                        "condition_name",
                        "",
                    ),
                    "observable_key": row.get(
                        "observable_key",
                        "",
                    ),
                    "measurement_ids": row.get(
                        "measurement_ids",
                        [],
                    ),
                    "text": row.get("text", ""),
                    "source_expressions": row.get(
                        "source_expressions",
                        [],
                    ),
                    "current_control": row.get(
                        "current_control",
                    ),
                    "current_response": row.get(
                        "current_response",
                    ),
                    "current_direction_shape": row.get(
                        "current_direction_shape",
                    ),
                    "current_control_key_from_name": row.get(
                        "current_control_key_from_name",
                        "",
                    ),
                    "compatibility_reasons": row.get(
                        "compatibility_reasons",
                        [],
                    ),
                    "provenance": row.get(
                        "provenance",
                        {},
                    ),
                    "human_adjudication": "",
                    "human_reason": "",
                    "recommended_action": "",
                }
            )
    return sample


def summarize_attribution(
    *,
    base_summary: Mapping[str, Any],
    claim_rows: list[Mapping[str, Any]],
    numeric_rows: list[Mapping[str, Any]],
    sample_rows: list[Mapping[str, Any]],
    trend_semantics_id: str,
    trend_implementation_sha256: str,
) -> dict[str, Any]:
    claim_counts = Counter(
        str(row["primary_reason"])
        for row in claim_rows
    )
    numeric_counts = Counter(
        str(row["primary_reason"])
        for row in numeric_rows
    )
    sample_counts = Counter(
        (
            str(row["kind"]),
            str(row["primary_reason"]),
        )
        for row in sample_rows
    )

    return {
        "attribution_id": ALPHA4C5G1_ATTRIBUTION_ID,
        "source_diagnostic_id": base_summary.get(
            "diagnostic_id"
        ),
        "source_diagnostic_semantic_sha256": (
            semantic_sha256(base_summary)
        ),
        "development_only": True,
        "paper_count": base_summary.get("paper_count"),
        "reserve_a_used": False,
        "reserve_b_used": False,
        "reserve_b_remains_sealed": True,
        "llm_calls": 0,
        "scientific_semantics_modified": False,
        "acceptance_semantics_modified": False,
        "trend_semantics_id": trend_semantics_id,
        "trend_implementation_sha256": (
            trend_implementation_sha256
        ),
        "claim_miss_count": len(claim_rows),
        "claim_reason_counts": dict(
            sorted(claim_counts.items())
        ),
        "numeric_miss_count": len(numeric_rows),
        "numeric_reason_counts": dict(
            sorted(numeric_counts.items())
        ),
        "sample_count": len(sample_rows),
        "sample_counts": {
            f"{kind}:{reason}": count
            for (kind, reason), count in sorted(
                sample_counts.items()
            )
        },
        "attribution_is_diagnostic_not_ground_truth": True,
        "human_adjudication_required_before_semantic_change": True,
        "notes": [
            (
                "Reason attribution follows the current frozen local "
                "Trend helper logic; it does not weaken or replace "
                "Trend admission."
            ),
            (
                "An unexplained_post_* bucket is intentionally "
                "preserved instead of guessing when all inspected "
                "normalization/gating helpers pass."
            ),
            (
                "Stratified samples are selected deterministically "
                "by SHA256 within each reason bucket."
            ),
        ],
    }
