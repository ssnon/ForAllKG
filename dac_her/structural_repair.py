from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Any


NODE_COLLECTIONS = (
    "entities",
    "experiments",
    "calculations",
    "measurements",
    "measurement_groups",
    "observation_claims",
    "mechanism_claims",
)


@dataclass(frozen=True)
class StructuralRepair:
    operation: str
    object_type: str
    object_id: str
    details: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _dedupe_exact_nodes(
    payload: dict[str, Any],
    repairs: list[StructuralRepair],
) -> None:
    seen: dict[str, tuple[str, dict[str, Any]]] = {}

    for collection in NODE_COLLECTIONS:
        items = payload.get(collection)
        if not isinstance(items, list):
            continue

        kept: list[Any] = []
        for item in items:
            if not isinstance(item, dict):
                kept.append(item)
                continue

            node_id = item.get("id")
            if not isinstance(node_id, str) or not node_id:
                kept.append(item)
                continue

            previous = seen.get(node_id)
            if previous is None:
                seen[node_id] = (collection, item)
                kept.append(item)
                continue

            previous_collection, previous_item = previous
            if previous_collection == collection and previous_item == item:
                repairs.append(
                    StructuralRepair(
                        operation="drop_exact_duplicate_node",
                        object_type=collection,
                        object_id=node_id,
                        details=(
                            "An identical node payload with the same ID "
                            "was already present."
                        ),
                    )
                )
                continue

            # A conflicting duplicate remains so the strict validator rejects it.
            kept.append(item)

        payload[collection] = kept


def _dedupe_exact_edges(
    payload: dict[str, Any],
    repairs: list[StructuralRepair],
) -> None:
    edges = payload.get("edges")
    if not isinstance(edges, list):
        return

    seen: set[str] = set()
    kept: list[Any] = []

    for edge in edges:
        if not isinstance(edge, dict):
            kept.append(edge)
            continue

        marker = json.dumps(
            edge,
            sort_keys=True,
            ensure_ascii=False,
        )
        if marker in seen:
            repairs.append(
                StructuralRepair(
                    operation="drop_exact_duplicate_edge",
                    object_type="edge",
                    object_id=(
                        f"{edge.get('source', '?')}|"
                        f"{edge.get('relation', '?')}|"
                        f"{edge.get('target', '?')}"
                    ),
                    details="An identical edge payload was already present.",
                )
            )
            continue

        seen.add(marker)
        kept.append(edge)

    payload["edges"] = kept


def _edge_template_for_node(
    edges: list[dict[str, Any]],
    node_id: str,
) -> dict[str, Any] | None:
    preferred = [
        edge
        for edge in edges
        if edge.get("target") == node_id
        and edge.get("relation") == "HAS_MEASUREMENT"
    ]
    candidates = preferred or [
        edge
        for edge in edges
        if edge.get("source") == node_id
        or edge.get("target") == node_id
    ]

    for edge in candidates:
        pointers = edge.get("evidence_pointers")
        if isinstance(pointers, list) and pointers:
            return edge

    return None


def _make_consistency_edge(
    *,
    template: dict[str, Any],
    source: str,
    relation: str,
    target: str,
    evidence_text: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "evidence_type": template["evidence_type"],
        "evidence_strength": template["evidence_strength"],
        "evidence_text": evidence_text[:200],
        "confidence": template["confidence"],
        "evidence_pointers": deepcopy(template["evidence_pointers"]),
        "subsection": template.get("subsection"),
    }


def _remove_singleton_groups(
    payload: dict[str, Any],
    repairs: list[StructuralRepair],
) -> None:
    groups = payload.get("measurement_groups")
    measurements = payload.get("measurements")
    edges = payload.get("edges")

    if not all(
        isinstance(value, list)
        for value in (groups, measurements, edges)
    ):
        return

    singleton_ids = {
        group.get("id")
        for group in groups
        if isinstance(group, dict)
        and isinstance(group.get("id"), str)
        and len(group.get("member_measurement_ids") or []) < 2
    }
    singleton_ids.discard(None)

    if not singleton_ids:
        return

    payload["measurement_groups"] = [
        group
        for group in groups
        if not (
            isinstance(group, dict)
            and group.get("id") in singleton_ids
        )
    ]

    for measurement in measurements:
        if (
            isinstance(measurement, dict)
            and measurement.get("group_id") in singleton_ids
        ):
            old_group = str(measurement.get("group_id"))
            measurement["group_id"] = None
            repairs.append(
                StructuralRepair(
                    operation="clear_singleton_group_reference",
                    object_type="measurement",
                    object_id=str(measurement.get("id", "")),
                    details=(
                        f"Cleared group_id={old_group!r} because the group "
                        "had fewer than two members."
                    ),
                )
            )

    payload["edges"] = [
        edge
        for edge in edges
        if not (
            isinstance(edge, dict)
            and edge.get("relation") == "IN_MEASUREMENT_GROUP"
            and edge.get("target") in singleton_ids
        )
    ]

    for group_id in sorted(singleton_ids):
        repairs.append(
            StructuralRepair(
                operation="drop_singleton_measurement_group",
                object_type="measurement_group",
                object_id=str(group_id),
                details="MeasurementGroup had fewer than two members.",
            )
        )


def _repair_measurement_edges(
    payload: dict[str, Any],
    repairs: list[StructuralRepair],
) -> None:
    entities = payload.get("entities")
    measurements = payload.get("measurements")
    groups = payload.get("measurement_groups")
    edges = payload.get("edges")

    if not all(
        isinstance(value, list)
        for value in (entities, measurements, groups, edges)
    ):
        return

    entity_ids = {
        item.get("id")
        for item in entities
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
    }
    group_by_id = {
        item.get("id"): item
        for item in groups
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
    }
    groups_by_member: dict[str, list[str]] = {}
    for group_id, group in group_by_id.items():
        for member_id in group.get("member_measurement_ids") or []:
            if isinstance(member_id, str):
                groups_by_member.setdefault(member_id, []).append(group_id)

    for measurement in measurements:
        if not isinstance(measurement, dict):
            continue

        measurement_id = measurement.get("id")
        if not isinstance(measurement_id, str) or not measurement_id:
            continue

        template = _edge_template_for_node(edges, measurement_id)
        source_expression = str(
            measurement.get("source_expression")
            or measurement.get("description")
            or "Measurement relation stated in the source."
        )

        subject_id = measurement.get("subject_id")
        measured_for_edges = [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and edge.get("source") == measurement_id
            and edge.get("relation") == "MEASURED_FOR"
        ]

        if (
            isinstance(subject_id, str)
            and subject_id in entity_ids
            and template is not None
        ):
            correct = [
                edge
                for edge in measured_for_edges
                if edge.get("target") == subject_id
            ]
            if len(measured_for_edges) != 1 or not correct:
                edges[:] = [
                    edge
                    for edge in edges
                    if not (
                        isinstance(edge, dict)
                        and edge.get("source") == measurement_id
                        and edge.get("relation") == "MEASURED_FOR"
                    )
                ]
                edges.append(
                    _make_consistency_edge(
                        template=correct[0] if correct else template,
                        source=measurement_id,
                        relation="MEASURED_FOR",
                        target=subject_id,
                        evidence_text=source_expression,
                    )
                )
                repairs.append(
                    StructuralRepair(
                        operation="normalize_measured_for_edge",
                        object_type="measurement",
                        object_id=measurement_id,
                        details=(
                            "Normalized MEASURED_FOR to the explicit "
                            f"subject_id={subject_id!r}."
                        ),
                    )
                )

        group_id = measurement.get("group_id")
        memberships = groups_by_member.get(measurement_id, [])

        if group_id is None and len(memberships) == 1:
            group_id = memberships[0]
            measurement["group_id"] = group_id
            repairs.append(
                StructuralRepair(
                    operation="backfill_measurement_group_id",
                    object_type="measurement",
                    object_id=measurement_id,
                    details=(
                        f"Backfilled group_id={group_id!r} from the "
                        "group member list."
                    ),
                )
            )

        if not isinstance(group_id, str):
            continue

        group = group_by_id.get(group_id)
        if group is None:
            continue
        if measurement_id not in (group.get("member_measurement_ids") or []):
            continue
        if template is None:
            continue

        membership_edges = [
            edge
            for edge in edges
            if isinstance(edge, dict)
            and edge.get("source") == measurement_id
            and edge.get("relation") == "IN_MEASUREMENT_GROUP"
        ]
        correct = [
            edge
            for edge in membership_edges
            if edge.get("target") == group_id
        ]

        if len(membership_edges) != 1 or not correct:
            edges[:] = [
                edge
                for edge in edges
                if not (
                    isinstance(edge, dict)
                    and edge.get("source") == measurement_id
                    and edge.get("relation") == "IN_MEASUREMENT_GROUP"
                )
            ]
            edges.append(
                _make_consistency_edge(
                    template=correct[0] if correct else template,
                    source=measurement_id,
                    relation="IN_MEASUREMENT_GROUP",
                    target=group_id,
                    evidence_text=source_expression,
                )
            )
            repairs.append(
                StructuralRepair(
                    operation="normalize_measurement_group_edge",
                    object_type="measurement",
                    object_id=measurement_id,
                    details=(
                        "Normalized IN_MEASUREMENT_GROUP to "
                        f"group_id={group_id!r}."
                    ),
                )
            )


def _drop_orphan_claims(
    payload: dict[str, Any],
    repairs: list[StructuralRepair],
) -> None:
    edges = payload.get("edges")
    if not isinstance(edges, list):
        return

    drop_ids: set[str] = set()

    for collection, mechanism in (
        ("observation_claims", False),
        ("mechanism_claims", True),
    ):
        claims = payload.get(collection)
        if not isinstance(claims, list):
            continue

        for claim in claims:
            if not isinstance(claim, dict):
                continue

            claim_id = claim.get("id")
            if not isinstance(claim_id, str):
                continue

            incoming_relations = {
                edge.get("relation")
                for edge in edges
                if isinstance(edge, dict)
                and edge.get("target") == claim_id
            }
            has_support = (
                "SUPPORTS_CLAIM" in incoming_relations
                or (mechanism and "INTERPRETED_AS" in incoming_relations)
            )
            has_target = any(
                isinstance(edge, dict)
                and edge.get("source") == claim_id
                and edge.get("relation") == "APPLIES_TO"
                for edge in edges
            )

            if has_support and has_target:
                continue

            drop_ids.add(claim_id)
            missing: list[str] = []
            if not has_support:
                missing.append("support")
            if not has_target:
                missing.append("application target")
            repairs.append(
                StructuralRepair(
                    operation="drop_orphan_claim",
                    object_type=collection,
                    object_id=claim_id,
                    details=(
                        "Dropped claim lacking "
                        + " and ".join(missing)
                        + "."
                    ),
                )
            )

    if not drop_ids:
        return

    for collection in ("observation_claims", "mechanism_claims"):
        claims = payload.get(collection)
        if isinstance(claims, list):
            payload[collection] = [
                claim
                for claim in claims
                if not (
                    isinstance(claim, dict)
                    and claim.get("id") in drop_ids
                )
            ]

    payload["edges"] = [
        edge
        for edge in edges
        if not (
            isinstance(edge, dict)
            and (
                edge.get("source") in drop_ids
                or edge.get("target") in drop_ids
            )
        )
    ]


def _drop_isolated_nonmeasurement_nodes(
    payload: dict[str, Any],
    repairs: list[StructuralRepair],
) -> None:
    edges = payload.get("edges")
    if not isinstance(edges, list):
        return

    connected = {
        endpoint
        for edge in edges
        if isinstance(edge, dict)
        for endpoint in (edge.get("source"), edge.get("target"))
        if isinstance(endpoint, str)
    }

    for collection in (
        "entities",
        "experiments",
        "calculations",
        "observation_claims",
        "mechanism_claims",
    ):
        items = payload.get(collection)
        if not isinstance(items, list):
            continue

        kept: list[Any] = []
        for item in items:
            if (
                isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"] not in connected
            ):
                repairs.append(
                    StructuralRepair(
                        operation="drop_isolated_node",
                        object_type=collection,
                        object_id=item["id"],
                        details=(
                            "Node had no incoming or outgoing edge after "
                            "conservative repairs."
                        ),
                    )
                )
                continue

            kept.append(item)

        payload[collection] = kept


def repair_knowledge_graph_payload(
    raw_payload: dict[str, Any],
    *,
    allow_lossy: bool = False,
) -> tuple[
    dict[str, Any],
    list[StructuralRepair],
]:
    payload = deepcopy(
        raw_payload
    )
    repairs: list[
        StructuralRepair
    ] = []

    # Non-lossy deterministic normalization
    _dedupe_exact_nodes(
        payload,
        repairs,
    )
    _dedupe_exact_edges(
        payload,
        repairs,
    )
    _remove_singleton_groups(
        payload,
        repairs,
    )
    _repair_measurement_edges(
        payload,
        repairs,
    )
    _dedupe_exact_edges(
        payload,
        repairs,
    )

    # Destructive fallback only
    if allow_lossy:
        _drop_orphan_claims(
            payload,
            repairs,
        )
        _drop_isolated_nonmeasurement_nodes(
            payload,
            repairs,
        )
        _dedupe_exact_edges(
            payload,
            repairs,
        )

    return payload, repairs