from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# Shared evidence / provenance vocabulary
# ============================================================

EvidenceType = Literal[
    "bibliographic_metadata",
    "synthesis_procedure",
    "experimental_setup",
    "experimental_observation",
    "structural_characterization",
    "computational_method",
    "computational_result",
    "author_interpretation",
]


EvidenceStrength = Literal[
    "direct",
    "indirect",
    "interpretive",
]


ConfidenceLevel = Literal[
    "high",
    "medium",
    "low",
]


DocumentRole = Literal[
    "main",
    "supporting_information",
    "other",
]


RelationType = str

# ============================================================
# Shared evidence / provenance graph models
# ============================================================

class EvidencePointer(BaseModel):
    """Locator from a graph edge back to a source document or asset."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        ...,
        description="Document identifier supplied in the prompt.",
    )
    document_role: DocumentRole
    page_id: int | None = Field(
        ...,
        description=(
            "Marker page identifier when available. Use null when the "
            "source block has no reliable page locator."
        ),
    )
    asset_ids: list[str] = Field(
        ...,
        description=(
            "Subset of the chunk-level asset IDs that directly support "
            "this edge. Use an empty list for text-only evidence."
        ),
    )
    locator_text: str | None = Field(
        ...,
        description=(
            "Figure/table label, subsection locator, or concise source "
            "locator. Use null when unavailable."
        ),
    )


class KGEdge(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    source: str = Field(
        ...,
        description=(
            "ID of an existing source node."
        ),
    )

    relation: RelationType

    target: str = Field(
        ...,
        description=(
            "ID of an existing target node."
        ),
    )

    evidence_type: EvidenceType

    evidence_strength: EvidenceStrength = Field(
        ...,
        description=(
            "direct for directly reported observations "
            "or calculations; indirect for evidence-based "
            "support; interpretive for author explanations."
        ),
    )

    evidence_text: str = Field(
        ...,
        description=(
            "Short source-supported evidence span or "
            "faithful paraphrase."
        ),
    )

    confidence: ConfidenceLevel

    evidence_pointers: list[EvidencePointer] = Field(
        ...,
        description=(
            "One or more source locators. Every edge must retain at least "
            "one text/document pointer; asset_ids may be empty."
        ),
    )

    subsection: str | None = Field(
        ...,
        description=(
            "More specific subsection title when the "
            "chunk contains multiple subsections. "
            "Use null when unavailable."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def backfill_legacy_pointer(cls, value):
        # Old chunk caches predate document/asset provenance. Keep them
        # readable while requiring the fields in new structured outputs.
        if isinstance(value, dict) and "evidence_pointers" not in value:
            value = dict(value)
            value["evidence_pointers"] = [{
                "document_id": "main",
                "document_role": "main",
                "page_id": None,
                "asset_ids": [],
                "locator_text": value.get("subsection"),
            }]
        return value
