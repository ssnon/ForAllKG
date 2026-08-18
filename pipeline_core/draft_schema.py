from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.schemas import ExperimentNode
from pipeline_core.evidence_schema import (
    DocumentRole,
    KGEdge,
)
from pipeline_core.measurement_schema import (
    MeasurementGroupType,
    MeasurementNode,
)
from pipeline_core.scientific_node_schema import (
    CalculationNode,
    EntityNode,
    MechanismClaimNode,
    ObservationClaimNode,
)


class MeasurementGroupDraft(BaseModel):
    """Lenient group shape used before cross-graph validation.

    The strict MeasurementGroupNode rejects singleton groups immediately, which
    prevents the lossless normalizer from dissolving them. The draft preserves
    the payload and defers the cardinality rule to graph_validation.py.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    group_type: MeasurementGroupType
    label: str
    member_measurement_ids: list[str]
    description: str | None


class KnowledgeGraphDraft(BaseModel):
    """Schema-valid graph payload before cross-node graph validation.

    Node-local field validation remains strict. Cross-node constraints such as
    endpoint existence, relation endpoint types, isolated nodes, claim support,
    and measurement bookkeeping are collected as structured ValidationIssue
    objects instead of raising one opaque ValueError at a time.
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    chunk_id: str
    section: str

    document_id: str
    document_role: DocumentRole
    page_ids: list[int]
    asset_ids: list[str]

    entities: list[EntityNode]
    experiments: list[ExperimentNode]
    calculations: list[CalculationNode]
    measurements: list[MeasurementNode]
    measurement_groups: list[MeasurementGroupDraft]
    observation_claims: list[ObservationClaimNode]
    mechanism_claims: list[MechanismClaimNode]
    edges: list[KGEdge]

    @model_validator(mode="before")
    @classmethod
    def backfill_legacy_document_provenance(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value.setdefault("document_id", "main")
            value.setdefault("document_role", "main")
            value.setdefault("page_ids", [])
            value.setdefault("asset_ids", [])
            value.setdefault("measurement_groups", [])
        return value

    def all_node_ids(self) -> set[str]:
        return {
            node.id
            for group in (
                self.entities,
                self.experiments,
                self.calculations,
                self.measurements,
                self.measurement_groups,
                self.observation_claims,
                self.mechanism_claims,
            )
            for node in group
        }

    def node_collections(self) -> dict[str, list[BaseModel]]:
        return {
            "entities": list(self.entities),
            "experiments": list(self.experiments),
            "calculations": list(self.calculations),
            "measurements": list(self.measurements),
            "measurement_groups": list(self.measurement_groups),
            "observation_claims": list(self.observation_claims),
            "mechanism_claims": list(self.mechanism_claims),
        }
