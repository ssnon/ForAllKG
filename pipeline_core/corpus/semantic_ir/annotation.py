from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.corpus.semantic_ir.schema import SemanticObjectRef


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


AnnotationEpistemicStatus = Literal[
    "source_interpretation",
    "deterministic_derivation",
]


class SemanticAnnotation(StrictModel):
    """
    Append-only semantic enrichment attached to an existing semantic object.

    An annotation is a sidecar. Merely creating one must never mutate the
    canonical evidence graph or create premise, novelty, selection, or
    rejection authority.
    """

    schema_version: Literal[
        "semantic-annotation-v1"
    ] = "semantic-annotation-v1"

    annotation_id: str = Field(min_length=1)
    capability: str = Field(min_length=1)
    subject_ref: SemanticObjectRef
    value: dict[str, Any]
    source_refs: list[SemanticObjectRef] = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    epistemic_status: AnnotationEpistemicStatus

    requires_verification: Literal[True] = True
    append_only: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    selection_authority: Literal[False] = False
    rejection_authority: Literal[False] = False
