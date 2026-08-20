from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pipeline_core.draft_schema import KnowledgeGraphDraft
from dac_her.semantic_patch_schema import KnowledgeGraphPatch, PatchOperation
from pipeline_core.validation_issues import ValidationReport


class PatchRejected(ValueError):
    pass


@dataclass(frozen=True)
class PatchApplicationResult:
    draft: KnowledgeGraphDraft
    operation_count: int
    destructive_operation_count: int
    touched_issue_ids: tuple[str, ...]


def _required(value: Any, field: str, op: str) -> Any:
    if value is None:
        raise PatchRejected(f"Operation {op!r} requires {field!r}.")
    return value


def _all_node_ids(payload: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for collection in (
        "entities",
        "experiments",
        "calculations",
        "measurements",
        "measurement_groups",
        "observation_claims",
        "mechanism_claims",
    ):
        for item in payload.get(collection, []):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                result.add(item["id"])
    return result


def _validate_pointer_scope(
    edge_payload: dict[str, Any],
    *,
    document_id: str,
    document_role: str,
    page_ids: set[int],
    asset_ids: set[str],
) -> None:
    pointers = edge_payload.get("evidence_pointers")
    if not isinstance(pointers, list) or not pointers:
        raise PatchRejected("Added edge must contain at least one evidence pointer.")

    for pointer in pointers:
        if pointer.get("document_id") != document_id:
            raise PatchRejected(
                "Added edge pointer document_id is outside the chunk scope."
            )
        if pointer.get("document_role") != document_role:
            raise PatchRejected(
                "Added edge pointer document_role is outside the chunk scope."
            )
        page_id = pointer.get("page_id")
        if page_id is not None and page_ids and page_id not in page_ids:
            raise PatchRejected(f"Added edge uses unknown page_id={page_id!r}.")
        unknown_assets = set(pointer.get("asset_ids") or []) - asset_ids
        if unknown_assets:
            raise PatchRejected(
                f"Added edge uses unknown assets: {sorted(unknown_assets)!r}."
            )


def _find_node(
    payload: dict[str, Any],
    node_id: str,
) -> tuple[str, dict[str, Any]] | None:
    for collection in (
        "entities",
        "experiments",
        "calculations",
        "measurements",
        "measurement_groups",
        "observation_claims",
        "mechanism_claims",
    ):
        for item in payload.get(collection, []):
            if isinstance(item, dict) and item.get("id") == node_id:
                return collection, item
    return None


def _rename_references(payload: dict[str, Any], old_id: str, new_id: str) -> None:
    for edge in payload.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if edge.get("source") == old_id:
            edge["source"] = new_id
        if edge.get("target") == old_id:
            edge["target"] = new_id

    for measurement in payload.get("measurements", []):
        if not isinstance(measurement, dict):
            continue
        if measurement.get("subject_id") == old_id:
            measurement["subject_id"] = new_id
        if measurement.get("group_id") == old_id:
            measurement["group_id"] = new_id

    for group in payload.get("measurement_groups", []):
        if not isinstance(group, dict):
            continue
        members = group.get("member_measurement_ids")
        if isinstance(members, list):
            group["member_measurement_ids"] = [
                new_id if item == old_id else item for item in members
            ]


def apply_semantic_patch(
    *,
    draft: KnowledgeGraphDraft,
    patch: KnowledgeGraphPatch,
    report: ValidationReport,
    max_operations: int,
    allow_destructive: bool = False,
) -> PatchApplicationResult:
    if patch.paper_id != draft.paper_id or patch.chunk_id != draft.chunk_id:
        raise PatchRejected("Patch paper_id/chunk_id does not match the draft.")
    if len(patch.operations) > max_operations:
        raise PatchRejected(
            f"Patch contains {len(patch.operations)} operations; "
            f"maximum is {max_operations}."
        )

    known_issue_ids = {item.issue_id for item in report.issues}
    issues_by_id = {item.issue_id: item for item in report.issues}
    unknown_unresolved = set(patch.unresolved_issue_ids) - known_issue_ids
    if unknown_unresolved:
        raise PatchRejected(
            "Patch lists unknown unresolved issue IDs: "
            f"{sorted(unknown_unresolved)!r}."
        )

    payload = deepcopy(draft.model_dump())
    touched: set[str] = set()
    destructive_count = 0

    for operation in patch.operations:
        unknown_issue_ids = set(operation.issue_ids) - known_issue_ids
        if unknown_issue_ids:
            raise PatchRejected(
                "Patch operation references unknown issue IDs: "
                f"{sorted(unknown_issue_ids)!r}."
            )
        touched.update(operation.issue_ids)
        referenced_issues = [issues_by_id[item] for item in operation.issue_ids]

        if operation.op == "add_edge":
            edge = _required(operation.edge, "edge", operation.op)
            touched_nodes = {edge.source, edge.target}
            if not any(
                item.node_id in touched_nodes
                or item.source_id in touched_nodes
                or item.target_id in touched_nodes
                or item.relation == edge.relation
                for item in referenced_issues
            ):
                raise PatchRejected(
                    "add_edge is not scoped to any referenced validation issue."
                )
            edge_payload = edge.model_dump()
            node_ids = _all_node_ids(payload)
            if (
                edge_payload["source"] not in node_ids
                or edge_payload["target"] not in node_ids
            ):
                raise PatchRejected(
                    "add_edge endpoints must already exist in the draft."
                )
            _validate_pointer_scope(
                edge_payload,
                document_id=draft.document_id,
                document_role=draft.document_role,
                page_ids=set(draft.page_ids),
                asset_ids=set(draft.asset_ids),
            )
            marker = (
                edge_payload["source"],
                edge_payload["relation"],
                edge_payload["target"],
            )
            if any(
                isinstance(existing, dict)
                and (
                    existing.get("source"),
                    existing.get("relation"),
                    existing.get("target"),
                )
                == marker
                for existing in payload.get("edges", [])
            ):
                raise PatchRejected(f"add_edge would duplicate relation {marker!r}.")
            payload.setdefault("edges", []).append(edge_payload)

        elif operation.op == "remove_edge":
            edge_index = _required(operation.edge_index, "edge_index", operation.op)
            if not any(item.edge_index == edge_index for item in referenced_issues):
                raise PatchRejected(
                    "remove_edge is not scoped to the referenced edge issue."
                )
            destructive_count += 1
            if not allow_destructive:
                raise PatchRejected("remove_edge is disabled in strict recovery mode.")
            edges = payload.get("edges", [])
            if edge_index < 0 or edge_index >= len(edges):
                raise PatchRejected("remove_edge index is out of range.")
            edge = edges[edge_index]
            expected = (
                _required(operation.expected_source, "expected_source", operation.op),
                _required(operation.expected_relation, "expected_relation", operation.op),
                _required(operation.expected_target, "expected_target", operation.op),
            )
            actual = (edge.get("source"), edge.get("relation"), edge.get("target"))
            if actual != expected:
                raise PatchRejected(
                    f"remove_edge stale index: expected {expected!r}, found {actual!r}."
                )
            del edges[edge_index]

        elif operation.op == "replace_edge":
            edge_index = _required(
                operation.edge_index,
                "edge_index",
                operation.op,
            )

            if not any(
                item.edge_index == edge_index
                for item in referenced_issues
            ):
                raise PatchRejected(
                    "replace_edge is not scoped to "
                    "the referenced edge issue."
                )

            edges = payload.get("edges", [])

            if (
                edge_index < 0
                or edge_index >= len(edges)
            ):
                raise PatchRejected(
                    "replace_edge index is out of range."
                )

            old_edge = edges[edge_index]

            expected = (
                _required(
                    operation.expected_source,
                    "expected_source",
                    operation.op,
                ),
                _required(
                    operation.expected_relation,
                    "expected_relation",
                    operation.op,
                ),
                _required(
                    operation.expected_target,
                    "expected_target",
                    operation.op,
                ),
            )

            actual = (
                old_edge.get("source"),
                old_edge.get("relation"),
                old_edge.get("target"),
            )

            if actual != expected:
                raise PatchRejected(
                    "replace_edge stale index: "
                    f"expected {expected!r}, "
                    f"found {actual!r}."
                )

            replacement = _required(
                operation.edge,
                "edge",
                operation.op,
            )

            replacement_payload = (
                replacement.model_dump()
            )

            node_ids = _all_node_ids(payload)

            replacement_source = (
                replacement_payload.get("source")
            )
            replacement_target = (
                replacement_payload.get("target")
            )

            if replacement_source not in node_ids:
                raise PatchRejected(
                    "replace_edge replacement source "
                    f"is undefined: {replacement_source!r}."
                )

            if replacement_target not in node_ids:
                raise PatchRejected(
                    "replace_edge replacement target "
                    f"is undefined: {replacement_target!r}."
                )

            _validate_pointer_scope(
                replacement_payload,
                document_id=draft.document_id,
                document_role=draft.document_role,
                page_ids=set(draft.page_ids),
                asset_ids=set(draft.asset_ids),
            )

            edges[edge_index] = replacement_payload
            
        elif operation.op == "change_entity_type":
            node_id = _required(operation.node_id, "node_id", operation.op)
            old_type = _required(operation.old_type, "old_type", operation.op)
            new_type = _required(operation.new_type, "new_type", operation.op)
            if not any(
                item.node_id == node_id
                or item.source_id == node_id
                or item.target_id == node_id
                for item in referenced_issues
            ):
                raise PatchRejected(
                    "change_entity_type is not scoped to the referenced node issue."
                )
            found = _find_node(payload, node_id)
            if found is None or found[0] != "entities":
                raise PatchRejected(
                    "change_entity_type target must be an existing Entity."
                )
            entity = found[1]
            if entity.get("type") != old_type:
                raise PatchRejected(
                    f"change_entity_type expected {old_type!r}, "
                    f"found {entity.get('type')!r}."
                )
            entity["type"] = new_type

        elif operation.op == "replace_edge_endpoint":
            edge_index = _required(operation.edge_index, "edge_index", operation.op)
            endpoint = _required(operation.endpoint, "endpoint", operation.op)
            old_id = _required(operation.old_id, "old_id", operation.op)
            new_id = _required(operation.new_id, "new_id", operation.op)
            if not any(
                item.edge_index == edge_index
                or item.source_id == old_id
                or item.target_id == old_id
                for item in referenced_issues
            ):
                raise PatchRejected(
                    "replace_edge_endpoint is not scoped to the referenced issue."
                )
            edges = payload.get("edges", [])
            if edge_index < 0 or edge_index >= len(edges):
                raise PatchRejected("replace_edge_endpoint index is out of range.")
            if new_id not in _all_node_ids(payload):
                raise PatchRejected(
                    "replace_edge_endpoint new_id must already exist."
                )

            edge = edges[edge_index]
            expected = (
                _required(
                    operation.expected_source,
                    "expected_source",
                    operation.op,
                ),
                _required(
                    operation.expected_relation,
                    "expected_relation",
                    operation.op,
                ),
                _required(
                    operation.expected_target,
                    "expected_target",
                    operation.op,
                ),
            )
            actual = (
                edge.get("source"),
                edge.get("relation"),
                edge.get("target"),
            )
            if actual != expected:
                raise PatchRejected(
                    "replace_edge_endpoint stale index: "
                    f"expected {expected!r}, found {actual!r}."
                )

            expected_endpoint_id = (
                expected[0]
                if endpoint == "source"
                else expected[2]
            )
            if expected_endpoint_id != old_id:
                raise PatchRejected(
                    "replace_edge_endpoint old_id does not match the "
                    f"expected {endpoint} endpoint: "
                    f"{old_id!r} != {expected_endpoint_id!r}."
                )

            edge[endpoint] = new_id

        elif operation.op == "rename_node_id":
            old_id = _required(operation.old_id, "old_id", operation.op)
            new_id = _required(operation.new_id, "new_id", operation.op)
            if not any(
                item.node_id == old_id
                or item.source_id == old_id
                or item.target_id == old_id
                for item in referenced_issues
            ):
                raise PatchRejected(
                    "rename_node_id is not scoped to the referenced issue."
                )
            if new_id in _all_node_ids(payload):
                raise PatchRejected("rename_node_id new_id already exists.")
            found = _find_node(payload, old_id)
            if found is None:
                raise PatchRejected("rename_node_id old_id does not exist.")
            found[1]["id"] = new_id
            _rename_references(payload, old_id, new_id)

        else:  # pragma: no cover
            raise PatchRejected(f"Unsupported patch operation: {operation.op!r}")

    patched = KnowledgeGraphDraft.model_validate(payload)
    return PatchApplicationResult(
        draft=patched,
        operation_count=len(patch.operations),
        destructive_operation_count=destructive_count,
        touched_issue_ids=tuple(sorted(touched)),
    )
