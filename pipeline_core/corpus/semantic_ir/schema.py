from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SemanticObjectKind = Literal[
    "paper",
    "chunk",
    "entity",
    "experiment",
    "calculation",
    "measurement",
    "measurement_group",
    "observation_claim",
    "mechanism_claim",
    "edge",
    "other",
]


class SemanticObjectRef(StrictModel):
    """Stable reference to an existing source or canonical semantic object."""

    schema_version: Literal[
        "semantic-object-ref-v1"
    ] = "semantic-object-ref-v1"

    paper_id: str = Field(min_length=1)
    chunk_id: str | None = None
    object_kind: SemanticObjectKind
    object_id: str = Field(min_length=1)
    source_path: str | None = None

    @model_validator(mode="after")
    def validate_chunk_identity(self) -> "SemanticObjectRef":
        if self.object_kind == "chunk" and not self.chunk_id:
            raise ValueError("chunk references require chunk_id")
        return self

    def identity_key(self) -> tuple[str, str | None, str, str]:
        return (
            self.paper_id,
            self.chunk_id,
            self.object_kind,
            self.object_id,
        )


class SemanticIRRecord(StrictModel):
    """
    Lossless wrapper around an existing canonical extraction object.

    This layer does not reinterpret the payload or create new evidence
    authority. It only makes existing extraction artifacts addressable by a
    stable reference plus their source chunk.
    """

    schema_version: Literal[
        "semantic-ir-record-v1"
    ] = "semantic-ir-record-v1"

    ref: SemanticObjectRef
    payload: dict[str, Any]
    source_chunk_ref: SemanticObjectRef | None = None

    canonical_graph_mutated: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_source_chunk_ref(self) -> "SemanticIRRecord":
        source = self.source_chunk_ref
        if source is None:
            return self

        if source.object_kind != "chunk":
            raise ValueError("source_chunk_ref must reference a chunk")
        if source.paper_id != self.ref.paper_id:
            raise ValueError(
                "source_chunk_ref and semantic object must share paper_id"
            )
        if (
            self.ref.chunk_id is not None
            and source.chunk_id != self.ref.chunk_id
        ):
            raise ValueError(
                "source_chunk_ref must match the semantic object's chunk_id"
            )
        return self


class SemanticIRBundle(StrictModel):
    """Paper-local reusable semantic IR assembled without new LLM calls."""

    schema_version: Literal[
        "semantic-ir-bundle-v1"
    ] = "semantic-ir-bundle-v1"

    bundle_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    source_run_id: str | None = None
    source_attempt_directory: str | None = None
    records: list[SemanticIRRecord] = Field(default_factory=list)
    source_chunks: list[SemanticObjectRef] = Field(default_factory=list)

    llm_calls_performed: Literal[0] = 0
    canonical_graph_mutated: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False

    @model_validator(mode="after")
    def validate_paper_locality(self) -> "SemanticIRBundle":
        for record in self.records:
            if record.ref.paper_id != self.paper_id:
                raise ValueError(
                    "SemanticIRBundle records must be paper-local"
                )
        for source in self.source_chunks:
            if source.object_kind != "chunk":
                raise ValueError("source_chunks may contain only chunk refs")
            if source.paper_id != self.paper_id:
                raise ValueError(
                    "SemanticIRBundle source chunks must be paper-local"
                )
        return self
