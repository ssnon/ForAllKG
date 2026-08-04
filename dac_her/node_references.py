from __future__ import annotations

import json
from typing import Any, Mapping


def _remap(value: str, id_map: Mapping[str, str]) -> str:
    return str(id_map.get(str(value), str(value)))


def _json_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def remap_node_reference_attributes(
    node_data: Mapping[str, Any],
    id_map: Mapping[str, str],
) -> dict[str, Any]:
    """Remap node IDs stored inside GraphML-compatible node attributes.

    Edge endpoints are remapped separately by NetworkX. This helper handles
    foreign-key-like values embedded inside nodes so they cannot become stale
    after collision namespacing or paper-level entity resolution.
    """
    remapped = dict(node_data)
    node_type = str(remapped.get("type", ""))

    if node_type == "Measurement":
        subject_id = str(remapped.get("subject_id", "")).strip()
        if subject_id:
            remapped["subject_id"] = _remap(subject_id, id_map)

        group_id = str(remapped.get("group_id", "")).strip()
        if group_id:
            remapped["group_id"] = _remap(group_id, id_map)

    if node_type == "MeasurementGroup":
        member_ids = _json_list(
            remapped.get("member_measurement_ids_json")
        )
        if member_ids:
            remapped["member_measurement_ids_json"] = json.dumps(
                [_remap(member_id, id_map) for member_id in member_ids],
                ensure_ascii=False,
            )

    return remapped
