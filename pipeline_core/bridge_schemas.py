from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.schemas import (
    ConfidenceLevel,
    DocumentRole,
    EvidencePointer,
    EvidenceStrength,
    EvidenceType,
)


BridgeConceptType = Literal[
    "RelationPattern",
    "Phenomenon",
    "ElectronicEffect",
    "InterfacialEffect",
    "TransportEffect",
    "DynamicState",
    "DesignPrinciple",
    "FailureMode",
    "MechanisticAnalogy",
    "OpenQuestion",
]

BridgeRetentionLane = Literal[
    "accepted_pattern",
    "paper_local_frontier",
]

BridgePatternRelation = Literal[
    "CORRELATES_WITH",
    "VARIES_WITH",
    "COMPETES_WITH",
    "COMPETES_FOR",
    "SELECTS",
    "CONTRASTS_WITH",
    "MODULATES",
    "MEDIATES",
    "PROMOTES",
    "SUPPRESSES",
    "SUGGESTS_DESIGN_RULE",
    "IMPOSES_TRADEOFF",
    "IDENTIFIES_FAILURE_MODE",
]

BridgeRelationStrength = Literal[
    "descriptive",
    "correlational",
    "causal_interpretive",
]

BridgeEvidenceScope = Literal[
    "paper_result",
    "author_interpretation",
    "background",
]

BridgePatternSupportMode = Literal[
    "explicit_single_span",
    "derived_multi_span",
]

BridgeAnchorRelation = Literal[
    "EXPRESSES_PATTERN",
    "INVOLVES_PHENOMENON",
    "DESCRIBES_INTERFACE",
    "EXHIBITS_DYNAMIC_STATE",
    "SUGGESTS_DESIGN_PRINCIPLE",
    "HAS_FAILURE_MODE",
    "USES_MECHANISTIC_ANALOGY",
    "RAISES_OPEN_QUESTION",
]


class BridgeQualifier(BaseModel):
    """One structured qualifier narrowing a reusable bridge pattern."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1, max_length=80)
    value: str = Field(..., min_length=1, max_length=240)


class BridgeComparisonItem(BaseModel):
    """One explicit row/item supporting a relation derived across items."""

    model_config = ConfigDict(extra="forbid")

    subject_value: str = Field(..., min_length=1, max_length=180)
    object_value: str = Field(..., min_length=1, max_length=220)
    source_phrase: str = Field(..., min_length=2, max_length=520)


class BridgeConcept(BaseModel):
    """A reified relation pattern or rare source-explicit frontier concept."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=120)
    concept_type: BridgeConceptType
    label: str = Field(..., min_length=2, max_length=180)
    source_phrase: str = Field(..., min_length=2, max_length=520)
    description: str | None = Field(..., max_length=320)
    retention_lane: BridgeRetentionLane
    evidence_scope: BridgeEvidenceScope

    # Pattern fields. Frontier concepts set these to null/empty values.
    pattern_subject: str | None = Field(..., max_length=180)
    pattern_relation: BridgePatternRelation | None
    pattern_object: str | None = Field(..., max_length=180)
    relation_strength: BridgeRelationStrength | None
    qualifiers: list[BridgeQualifier] = Field(..., max_length=12)

    # Pattern grounding fields. They make the relation support auditable instead
    # of treating the mere presence of source_phrase as relation entailment.
    pattern_support_mode: BridgePatternSupportMode | None
    supporting_phrases: list[str] = Field(..., max_length=8)
    subject_evidence_phrase: str | None = Field(..., max_length=240)
    relation_evidence_phrase: str | None = Field(..., max_length=240)
    object_evidence_phrase: str | None = Field(..., max_length=240)
    comparison_items: list[BridgeComparisonItem] = Field(..., max_length=8)

    @model_validator(mode="before")
    @classmethod
    def backfill_legacy_concept_fields(cls, value):
        # Legacy v1/v2 files remain parseable enough to invalidate their cache
        # deterministically. New accepted patterns fail lane validation until
        # regenerated with the grounding fields below.
        if isinstance(value, dict):
            value = dict(value)
            if "retention_lane" not in value:
                value.setdefault("retention_lane", "paper_local_frontier")
                value.setdefault("evidence_scope", "paper_result")
                value.setdefault("pattern_subject", None)
                value.setdefault("pattern_relation", None)
                value.setdefault("pattern_object", None)
                value.setdefault("relation_strength", None)
                value.setdefault("qualifiers", [])
            value.setdefault("pattern_support_mode", None)
            value.setdefault("supporting_phrases", [])
            value.setdefault("subject_evidence_phrase", None)
            value.setdefault("relation_evidence_phrase", None)
            value.setdefault("object_evidence_phrase", None)
            value.setdefault("comparison_items", [])
        return value

    @model_validator(mode="after")
    def validate_lane_shape(self) -> "BridgeConcept":
        pattern_fields = (
            self.pattern_subject,
            self.pattern_relation,
            self.pattern_object,
            self.relation_strength,
        )
        evidence_fields = (
            self.subject_evidence_phrase,
            self.relation_evidence_phrase,
            self.object_evidence_phrase,
        )

        if self.retention_lane == "accepted_pattern":
            if self.concept_type != "RelationPattern":
                raise ValueError(
                    "accepted_pattern concepts must use concept_type=RelationPattern."
                )
            if any(value is None for value in pattern_fields):
                raise ValueError(
                    "accepted_pattern requires subject, relation, object, and "
                    "relation_strength."
                )
            if self.pattern_support_mode is None:
                raise ValueError(
                    "accepted_pattern requires pattern_support_mode."
                )
            if not self.supporting_phrases:
                raise ValueError(
                    "accepted_pattern requires at least one supporting phrase."
                )
            if self.source_phrase not in self.supporting_phrases:
                raise ValueError(
                    "source_phrase must be one of supporting_phrases."
                )

            if self.pattern_support_mode == "explicit_single_span":
                if len(self.supporting_phrases) != 1:
                    raise ValueError(
                        "explicit_single_span requires exactly one supporting phrase."
                    )
                if any(value is None for value in evidence_fields):
                    raise ValueError(
                        "explicit_single_span requires subject, relation, and object "
                        "evidence phrases."
                    )
                if self.comparison_items:
                    raise ValueError(
                        "explicit_single_span must use an empty comparison_items list."
                    )
            else:
                if len(self.supporting_phrases) < 2:
                    raise ValueError(
                        "derived_multi_span requires at least two supporting phrases."
                    )
                if len(self.comparison_items) < 2:
                    raise ValueError(
                        "derived_multi_span requires at least two comparison items."
                    )
                if any(value is not None for value in evidence_fields):
                    raise ValueError(
                        "derived_multi_span must set the three evidence phrase "
                        "fields to null; the comparison items provide grounding."
                    )
                if self.pattern_relation not in {
                    "CORRELATES_WITH",
                    "VARIES_WITH",
                    "CONTRASTS_WITH",
                }:
                    raise ValueError(
                        "derived_multi_span is restricted to CORRELATES_WITH, "
                        "VARIES_WITH, or CONTRASTS_WITH."
                    )
                comparison_phrases = {
                    item.source_phrase for item in self.comparison_items
                }
                if not comparison_phrases.issubset(set(self.supporting_phrases)):
                    raise ValueError(
                        "Every comparison item source_phrase must also appear in "
                        "supporting_phrases."
                    )
        else:
            if self.concept_type == "RelationPattern":
                raise ValueError(
                    "paper_local_frontier cannot use concept_type=RelationPattern."
                )
            if any(value is not None for value in pattern_fields):
                raise ValueError(
                    "paper_local_frontier must set all pattern fields to null."
                )
            if self.qualifiers:
                raise ValueError(
                    "paper_local_frontier must use an empty qualifiers list."
                )
            if self.pattern_support_mode is not None:
                raise ValueError(
                    "paper_local_frontier must set pattern_support_mode to null."
                )
            if self.supporting_phrases:
                raise ValueError(
                    "paper_local_frontier must use an empty supporting_phrases list."
                )
            if any(value is not None for value in evidence_fields):
                raise ValueError(
                    "paper_local_frontier must set pattern evidence phrases to null."
                )
            if self.comparison_items:
                raise ValueError(
                    "paper_local_frontier must use an empty comparison_items list."
                )
        return self


class BridgeLink(BaseModel):
    """A source-explicit grounding link from a strict node to a bridge node."""

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


class BridgeChunkGraph(BaseModel):
    """Bridge v2 output for one already-validated strict extraction chunk."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    chunk_id: str
    section: str
    document_id: str
    document_role: DocumentRole
    page_ids: list[int]
    asset_ids: list[str]
    concepts: list[BridgeConcept] = Field(..., max_length=8)
    links: list[BridgeLink] = Field(..., max_length=16)

    @model_validator(mode="after")
    def validate_graph_shape(self) -> "BridgeChunkGraph":
        concept_ids = [concept.id for concept in self.concepts]
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("Bridge concept IDs must be unique within a chunk.")

        by_id = {concept.id: concept for concept in self.concepts}
        linked_concepts: set[str] = set()
        seen_links: set[tuple[str, str, str]] = set()

        for link in self.links:
            concept = by_id.get(link.concept_id)
            if concept is None:
                raise ValueError(
                    f"Bridge link targets unknown concept: {link.concept_id!r}."
                )
            signature = (link.anchor_id, link.relation, link.concept_id)
            if signature in seen_links:
                raise ValueError(f"Duplicate bridge link: {signature!r}.")
            seen_links.add(signature)
            linked_concepts.add(link.concept_id)

            if concept.retention_lane == "accepted_pattern":
                if link.relation != "EXPRESSES_PATTERN":
                    raise ValueError(
                        "accepted_pattern concepts must be grounded with "
                        "EXPRESSES_PATTERN."
                    )
            elif link.relation == "EXPRESSES_PATTERN":
                raise ValueError(
                    "paper_local_frontier concepts cannot use EXPRESSES_PATTERN."
                )

        missing = set(concept_ids) - linked_concepts
        if missing:
            raise ValueError(
                "Every bridge concept must be linked to at least one strict-graph "
                f"anchor. Unlinked: {sorted(missing)!r}."
            )

        if not self.concepts and self.links:
            raise ValueError("A bridge graph with no concepts cannot contain links.")
        return self
