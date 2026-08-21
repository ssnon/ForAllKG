from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from pipeline_core.runtime.validation_issues import IssueCode, ValidationIssue


NODE_COLLECTIONS = (
    "entities",
    "experiments",
    "calculations",
    "measurements",
    "measurement_groups",
    "observation_claims",
    "mechanism_claims",
)

LOSSLESS_ISSUE_CODES = {
    IssueCode.DUPLICATE_NODE_ID,
    IssueCode.INVALID_MEASURED_FOR_COUNT,
    IssueCode.MEASURED_FOR_TARGET_MISMATCH,
    IssueCode.MISSING_MEASUREMENT_GROUP_EDGE,
    IssueCode.UNEXPECTED_MEASUREMENT_GROUP_EDGE,
    IssueCode.SINGLETON_MEASUREMENT_GROUP,
    IssueCode.DUPLICATE_MEASUREMENT_GROUP_MEMBER,
}


@dataclass(frozen=True)
class NormalizationOperation:
    operation: str
    object_type: str
    object_id: str
    details: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizationResult:
    payload: dict[str, Any]
    operations: list[NormalizationOperation]
    before_sha256: str
    after_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "operation_count": len(self.operations),
            "operations": [item.to_dict() for item in self.operations],
        }


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _dedupe_exact_nodes(
    payload: dict[str, Any],
    operations: list[NormalizationOperation],
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
                operations.append(
                    NormalizationOperation(
                        operation="drop_exact_duplicate_node",
                        object_type=collection,
                        object_id=node_id,
                        details="Removed an identical repeated node payload.",
                    )
                )
                continue

            # Conflicting duplicates are semantic failures. Keep them so the
            # structured validator can route the chunk to patch/rechunk.
            kept.append(item)

        payload[collection] = kept


def _dedupe_exact_edges(
    payload: dict[str, Any],
    operations: list[NormalizationOperation],
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

        marker = json.dumps(edge, ensure_ascii=False, sort_keys=True)
        if marker in seen:
            operations.append(
                NormalizationOperation(
                    operation="drop_exact_duplicate_edge",
                    object_type="edge",
                    object_id=(
                        f"{edge.get('source', '?')}|{edge.get('relation', '?')}|"
                        f"{edge.get('target', '?')}"
                    ),
                    details="Removed an identical repeated edge payload.",
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
        if edge.get("source") == node_id or edge.get("target") == node_id
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


def _dissolve_singleton_groups(
    payload: dict[str, Any],
    operations: list[NormalizationOperation],
) -> None:
    groups = payload.get("measurement_groups")
    measurements = payload.get("measurements")
    edges = payload.get("edges")
    if not all(isinstance(value, list) for value in (groups, measurements, edges)):
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
        if not (isinstance(group, dict) and group.get("id") in singleton_ids)
    ]

    for measurement in measurements:
        if not isinstance(measurement, dict):
            continue
        if measurement.get("group_id") in singleton_ids:
            old_group = str(measurement.get("group_id"))
            measurement["group_id"] = None
            operations.append(
                NormalizationOperation(
                    operation="clear_singleton_group_reference",
                    object_type="measurement",
                    object_id=str(measurement.get("id", "")),
                    details=(
                        f"Cleared group_id={old_group!r}; the group had fewer "
                        "than two scalar measurements."
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
        operations.append(
            NormalizationOperation(
                operation="drop_singleton_measurement_group",
                object_type="measurement_group",
                object_id=str(group_id),
                details="Removed a container that could not represent a group.",
            )
        )


def _normalize_measurement_edges(
    payload: dict[str, Any],
    operations: list[NormalizationOperation],
) -> None:
    entities = payload.get("entities")
    measurements = payload.get("measurements")
    groups = payload.get("measurement_groups")
    edges = payload.get("edges")
    if not all(isinstance(value, list) for value in (entities, measurements, groups, edges)):
        return

    entity_ids = {
        item.get("id")
        for item in entities
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    group_by_id = {
        item.get("id"): item
        for item in groups
        if isinstance(item, dict) and isinstance(item.get("id"), str)
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
        measured_for = [
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
            correct = [edge for edge in measured_for if edge.get("target") == subject_id]
            if len(measured_for) != 1 or not correct:
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
                operations.append(
                    NormalizationOperation(
                        operation="normalize_measured_for_edge",
                        object_type="measurement",
                        object_id=measurement_id,
                        details=(
                            "Materialized the relation already encoded by "
                            f"subject_id={subject_id!r}."
                        ),
                    )
                )

        group_id = measurement.get("group_id")
        memberships = groups_by_member.get(measurement_id, [])
        if group_id is None and len(memberships) == 1:
            group_id = memberships[0]
            measurement["group_id"] = group_id
            operations.append(
                NormalizationOperation(
                    operation="backfill_measurement_group_id",
                    object_type="measurement",
                    object_id=measurement_id,
                    details=(
                        f"Backfilled group_id={group_id!r} from the unique "
                        "MeasurementGroup member list."
                    ),
                )
            )

        if not isinstance(group_id, str):
            # Remove stale membership edges only when the scalar explicitly says
            # it has no group. This does not select a new scientific target.
            stale = [
                edge
                for edge in edges
                if isinstance(edge, dict)
                and edge.get("source") == measurement_id
                and edge.get("relation") == "IN_MEASUREMENT_GROUP"
            ]
            if stale:
                edges[:] = [edge for edge in edges if edge not in stale]
                operations.append(
                    NormalizationOperation(
                        operation="remove_stale_measurement_group_edges",
                        object_type="measurement",
                        object_id=measurement_id,
                        details="Removed membership edges while group_id was null.",
                    )
                )
            continue

        group = group_by_id.get(group_id)
        if group is None or measurement_id not in (group.get("member_measurement_ids") or []):
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
        correct = [edge for edge in membership_edges if edge.get("target") == group_id]
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
            operations.append(
                NormalizationOperation(
                    operation="normalize_measurement_group_edge",
                    object_type="measurement",
                    object_id=measurement_id,
                    details=(
                        "Materialized the membership already encoded by "
                        f"group_id={group_id!r} and the group member list."
                    ),
                )
            )


def normalize_knowledge_graph_payload(
    raw_payload: dict[str, Any],
    *,
    issues: list[ValidationIssue] | None = None,
) -> NormalizationResult:
    """Apply only deterministic, information-preserving bookkeeping edits."""

    if issues is not None:
        issue_codes = {item.code for item in issues}
        # The normalizer may still remove exact duplicates even when another
        # semantic family is present, but it never attempts semantic repairs.
        _ = issue_codes & LOSSLESS_ISSUE_CODES

    before = deepcopy(raw_payload)
    payload = deepcopy(raw_payload)
    operations: list[NormalizationOperation] = []

    _dedupe_exact_nodes(payload, operations)
    _dedupe_exact_edges(payload, operations)
    _dissolve_singleton_groups(payload, operations)
    _normalize_measurement_edges(payload, operations)
    _dedupe_exact_edges(payload, operations)

    return NormalizationResult(
        payload=payload,
        operations=operations,
        before_sha256=_sha256(before),
        after_sha256=_sha256(payload),
    )
