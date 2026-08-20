from __future__ import annotations

from typing import Any

import networkx as nx


MEASUREMENT_MERGE_INVARIANT_ID = (
    "measurement_payload_isolation_v1_alpha4b4a"
)


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _text(value: Any) -> str:
    return str(value or "").strip()


def _numeric_equal(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return _text(left) == _text(right)


def measurement_mentions_conflict(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> bool:
    """Return True when two same-ID Measurement mentions are unsafe to merge.

    Chunk-level schemas enforce exactly one of value_numeric/value_text.
    Paper-level merging must preserve that invariant and must not silently
    combine distinct result payloads merely because an LLM reused a local ID.
    """
    if (
        _text(existing.get("type")) != "Measurement"
        or _text(incoming.get("type")) != "Measurement"
    ):
        return False

    for key in ("metric_id", "subject_id"):
        left = _text(existing.get(key))
        right = _text(incoming.get(key))
        if left and right and left != right:
            return True

    existing_numeric = not _blank(existing.get("value_numeric"))
    existing_text = not _blank(existing.get("value_text"))
    incoming_numeric = not _blank(incoming.get("value_numeric"))
    incoming_text = not _blank(incoming.get("value_text"))

    # An already-invalid mention must never be "repaired" by field-wise merge.
    if existing_numeric == existing_text:
        return True
    if incoming_numeric == incoming_text:
        return True

    # Numeric and textual representations are different scientific payload
    # kinds. Combining them would violate the strict Measurement XOR.
    if existing_numeric != incoming_numeric or existing_text != incoming_text:
        return True

    if existing_numeric:
        if not _numeric_equal(
            existing.get("value_numeric"),
            incoming.get("value_numeric"),
        ):
            return True
        left_unit = _text(existing.get("unit"))
        right_unit = _text(incoming.get("unit"))
        if left_unit and right_unit and left_unit != right_unit:
            return True
        return False

    # Textual measurements are qualitative/range/comparative payloads.
    # Different text is not assumed to be the same scalar result.
    return _text(existing.get("value_text")) != _text(
        incoming.get("value_text")
    )


def measurement_value_payload_issues(
    graph: nx.Graph,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for node_id, attrs in graph.nodes(data=True):
        if _text(attrs.get("type")) != "Measurement":
            continue
        numeric = not _blank(attrs.get("value_numeric"))
        text = not _blank(attrs.get("value_text"))
        if numeric != text:
            continue
        rows.append(
            {
                "id": str(node_id),
                "value_numeric": attrs.get("value_numeric", ""),
                "value_text": attrs.get("value_text", ""),
                "source_expression": attrs.get("source_expression", ""),
                "metric_id": attrs.get("metric_id", ""),
                "issue": (
                    "Measurement value payload violates numeric/text XOR"
                    if numeric
                    else "Measurement has neither numeric nor textual value"
                ),
                "measurement_merge_invariant_id": (
                    MEASUREMENT_MERGE_INVARIANT_ID
                ),
            }
        )
    return rows


def assert_measurement_value_xor(
    graph: nx.Graph,
    *,
    stage: str,
) -> None:
    issues = measurement_value_payload_issues(graph)
    if not issues:
        return
    examples = [
        {
            "id": row["id"],
            "metric_id": row["metric_id"],
            "value_numeric": row["value_numeric"],
            "value_text": row["value_text"],
        }
        for row in issues[:5]
    ]
    raise ValueError(
        "Measurement numeric/text XOR invariant failed at "
        f"{stage!r}: {len(issues)} issue(s). Examples: {examples!r}"
    )
