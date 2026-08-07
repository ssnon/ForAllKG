from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from dac_her.schemas import EntityType, KGEdge


PatchOperationType = Literal[
    "add_edge",
    "remove_edge",
    "replace_edge",
    "change_entity_type",
    "replace_edge_endpoint",
    "rename_node_id",
]


class PatchOperation(BaseModel):
    """Provider-compatible flat patch operation.

    The previous discriminated union generated ``oneOf`` under
    ``operations.items``. The OpenAI structured-output endpoint used by this
    project rejects that schema shape. This model keeps one object shape for
    every operation. All operation-specific fields are required-but-nullable in
    the generated JSON Schema; irrelevant fields must be null.
    """

    model_config = ConfigDict(extra="forbid")

    op: PatchOperationType
    issue_ids: list[str]
    evidence_reason: str

    # add_edge / replace_edge
    edge: KGEdge | None

    # remove_edge / replace_edge_endpoint
    edge_index: int | None
    expected_source: str | None
    expected_relation: str | None
    expected_target: str | None

    # change_entity_type
    node_id: str | None
    old_type: EntityType | None
    new_type: EntityType | None

    # replace_edge_endpoint
    endpoint: Literal["source", "target"] | None

    # replace_edge_endpoint / rename_node_id
    old_id: str | None
    new_id: str | None

    @model_validator(mode="before")
    @classmethod
    def backfill_nullable_fields(cls, value: Any) -> Any:
        """Keep local callers/tests compatible while JSON Schema stays strict.

        Fields are declared without defaults so the response schema places all
        properties in ``required``. Local hand-written payloads from the first
        v2.3.5 bundle may omit irrelevant fields; fill those with null before
        ordinary validation.
        """
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for field in (
            "edge",
            "edge_index",
            "expected_source",
            "expected_relation",
            "expected_target",
            "node_id",
            "old_type",
            "new_type",
            "endpoint",
            "old_id",
            "new_id",
        ):
            result.setdefault(field, None)
        return result

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "PatchOperation":
        if not self.issue_ids:
            raise ValueError(
                "Every patch operation must reference at least one issue_id."
            )

        required_by_op: dict[str, tuple[str, ...]] = {
            "add_edge": ("edge",),

            "remove_edge": (
                "edge_index",
                "expected_source",
                "expected_relation",
                "expected_target",
            ),

            "replace_edge": (
                "edge",
                "edge_index",
                "expected_source",
                "expected_relation",
                "expected_target",
            ),

            "change_entity_type": (
                "node_id",
                "old_type",
                "new_type",
            ),

            "replace_edge_endpoint": (
                "edge_index",
                "expected_source",
                "expected_relation",
                "expected_target",
                "endpoint",
                "old_id",
                "new_id",
            ),

            "rename_node_id": (
                "old_id",
                "new_id",
            ),
        }
        missing = [
            field
            for field in required_by_op[self.op]
            if getattr(self, field) is None
        ]
        if missing:
            raise ValueError(
                f"Operation {self.op!r} requires non-null fields: {missing}."
            )

        allowed_non_null: dict[str, set[str]] = {
            "add_edge": {
                "edge",
            },

            "remove_edge": {
                "edge_index",
                "expected_source",
                "expected_relation",
                "expected_target",
            },

            "replace_edge": {
                "edge",
                "edge_index",
                "expected_source",
                "expected_relation",
                "expected_target",
            },

            "change_entity_type": {
                "node_id",
                "old_type",
                "new_type",
            },

            "replace_edge_endpoint": {
                "edge_index",
                "expected_source",
                "expected_relation",
                "expected_target",
                "endpoint",
                "old_id",
                "new_id",
            },

            "rename_node_id": {
                "old_id",
                "new_id",
            },
        }
        operation_fields = {
            "edge",
            "edge_index",
            "expected_source",
            "expected_relation",
            "expected_target",
            "node_id",
            "old_type",
            "new_type",
            "endpoint",
            "old_id",
            "new_id",
        }
        unexpected = sorted(
            field
            for field in operation_fields - allowed_non_null[self.op]
            if getattr(self, field) is not None
        )
        if unexpected:
            raise ValueError(
                f"Operation {self.op!r} must set unrelated fields to null: "
                f"{unexpected}."
            )
        return self


class KnowledgeGraphPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    chunk_id: str
    operations: list[PatchOperation]
    unresolved_issue_ids: list[str]
    summary: str

    @model_validator(mode="after")
    def validate_issue_references(self) -> "KnowledgeGraphPatch":
        for operation in self.operations:
            if not operation.issue_ids:
                raise ValueError(
                    "Every patch operation must reference at least one issue_id."
                )
        return self


def forbidden_one_of_paths() -> list[str]:
    """Return JSON paths containing a provider-incompatible ``oneOf``."""
    schema = KnowledgeGraphPatch.model_json_schema()
    paths: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if key == "oneOf":
                    paths.append(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(schema, "")
    return paths
