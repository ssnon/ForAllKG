from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dac_her.bridge_schemas import (
    BridgeAnchorRelation,
    BridgeConceptType,
    BridgeEvidenceScope,
    BridgePatternRelation,
    BridgePatternSupportMode,
    BridgeRelationStrength,
    BridgeRetentionLane,
)
from dac_her.schemas import (
    ConfidenceLevel,
    DocumentRole,
    EvidencePointer,
    EvidenceStrength,
    EvidenceType,
)


class BridgeQualifierDraft(BaseModel):
    """Provider-friendly draft version of BridgeQualifier."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, max_length=80)
    value: str = Field(..., min_length=1, max_length=240)


class BridgeComparisonItemDraft(BaseModel):
    """Provider-friendly draft version of BridgeComparisonItem."""

    model_config = ConfigDict(extra="forbid")

    subject_value: str = Field(..., min_length=1, max_length=180)
    object_value: str = Field(..., min_length=1, max_length=220)
    source_phrase: str = Field(..., min_length=2, max_length=520)


class BridgeConceptDraft(BaseModel):
    """
    LLM-output draft for one bridge concept.

    This intentionally omits cross-field validators. Cross-field scientific and
    provenance constraints are applied after deterministic normalization by the
    Bridge recovery layer. Keeping these constraints out of the provider schema
    prevents one local concept error from invalidating the entire chunk before
    recovery has a chance to run.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=120)
    concept_type: BridgeConceptType
    label: str = Field(..., min_length=2, max_length=180)
    source_phrase: str = Field(..., min_length=2, max_length=520)
    description: str | None = Field(..., max_length=320)
    retention_lane: BridgeRetentionLane
    evidence_scope: BridgeEvidenceScope

    pattern_subject: str | None = Field(..., max_length=180)
    pattern_relation: BridgePatternRelation | None
    pattern_object: str | None = Field(..., max_length=180)
    relation_strength: BridgeRelationStrength | None
    qualifiers: list[BridgeQualifierDraft] = Field(..., max_length=12)

    pattern_support_mode: BridgePatternSupportMode | None
    supporting_phrases: list[str] = Field(..., max_length=8)
    subject_evidence_phrase: str | None = Field(..., max_length=240)
    relation_evidence_phrase: str | None = Field(..., max_length=240)
    object_evidence_phrase: str | None = Field(..., max_length=240)
    comparison_items: list[BridgeComparisonItemDraft] = Field(..., max_length=8)


class BridgeLinkDraft(BaseModel):
    """Provider-friendly draft version of BridgeLink."""

    model_config = ConfigDict(extra="forbid")

    anchor_id: str = Field(..., min_length=1)
    relation: BridgeAnchorRelation
    concept_id: str = Field(..., min_length=1)
    evidence_type: EvidenceType
    evidence_strength: EvidenceStrength
    evidence_text: str = Field(..., min_length=2, max_length=520)
    confidence: ConfidenceLevel
    evidence_pointers: list[EvidencePointer] = Field(..., min_length=1)
    subsection: str | None = Field(..., max_length=180)


class BridgeChunkDraft(BaseModel):
    """
    Recoverable Bridge chunk draft.

    Only field-level shape constraints are applied here. Graph/lane/source/anchor
    invariants are intentionally deferred to bridge_recovery.py.
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    chunk_id: str
    section: str
    document_id: str
    document_role: DocumentRole
    page_ids: list[int]
    asset_ids: list[str]
    concepts: list[BridgeConceptDraft] = Field(..., max_length=8)
    links: list[BridgeLinkDraft] = Field(..., max_length=16)


class BridgeCandidateRepair(BaseModel):
    """
    One constrained local repair result.

    Nullable concept is deliberate: the model may explicitly say the original
    candidate cannot be repaired without inventing unsupported science. Existing
    Bridge schemas already use nullable fields, keeping this schema compatible
    with the current structured-output provider path.
    """

    model_config = ConfigDict(extra="forbid")

    repairable: bool
    reason: str = Field(..., min_length=1, max_length=500)
    concept: BridgeConceptDraft | None
    links: list[BridgeLinkDraft] = Field(..., max_length=8)
