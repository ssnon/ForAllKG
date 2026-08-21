from __future__ import annotations


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
)

from pipeline_core.corpus.graph.knowledge_graph_compat_validation import (
    validate_graph_integrity_compat,
)

from pipeline_core.corpus.graph.knowledge_graph_validation_context import (
    relation_semantics_already_validated,
)

from pipeline_core.corpus.extraction.evidence_schema import (
    DocumentRole,
    KGEdge,
)
from pipeline_core.corpus.extraction.experiment_schema import ExperimentNode
from pipeline_core.corpus.extraction.measurement_schema import (
    MeasurementGroupNode,
    MeasurementNode,
)
from pipeline_core.corpus.extraction.scientific_node_schema import (
    CalculationNode,
    EntityNode,
    MechanismClaimNode,
    ObservationClaimNode,
)


# ============================================================
# Shared KnowledgeGraph wire container
#
# IMPORTANT:
# This model still preserves the historical mixed validation
# behavior, including legacy DAC relation semantics.
# M1f.1 changes ownership only. Semantic decomposition is deferred.
# ============================================================

class KnowledgeGraph(BaseModel):
    """
    Provenance-preserving graph extracted from one chunk.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    paper_id: str = Field(
        ...,
        description=(
            "Paper identifier supplied in the prompt."
        ),
    )

    chunk_id: str = Field(
        ...,
        description=(
            "Chunk identifier supplied in the prompt."
        ),
    )

    section: str = Field(
        ...,
        description=(
            "Parent section supplied in the prompt."
        ),
    )

    document_id: str = Field(
        ...,
        description="Source document identifier supplied in the prompt.",
    )

    document_role: DocumentRole

    page_ids: list[int] = Field(
        ...,
        description="Marker page identifiers associated with the core chunk.",
    )

    asset_ids: list[str] = Field(
        ...,
        description="Figure/table asset IDs linked to the core chunk.",
    )

    entities: list[EntityNode] = Field(
        ...,
        description=(
            "Scientific entity nodes."
        ),
    )

    experiments: list[ExperimentNode] = Field(
        ...,
        description=(
            "Experimental and characterization setups."
        ),
    )

    calculations: list[CalculationNode] = Field(
        ...,
        description=(
            "Computational procedures."
        ),
    )

    measurements: list[MeasurementNode] = Field(
        ...,
        description=(
            "Individual scalar experimental or computational results."
        ),
    )

    measurement_groups: list[MeasurementGroupNode] = Field(
        ...,
        description=(
            "Comparison/series containers whose members remain separate "
            "scalar Measurement nodes."
        ),
    )

    observation_claims: list[ObservationClaimNode] = Field(
        ...,
        description=(
            "Direct evidence-supported observational "
            "or comparative conclusions."
        ),
    )

    mechanism_claims: list[MechanismClaimNode] = Field(
        ...,
        description=(
            "Author-proposed mechanistic explanations."
        ),
    )

    edges: list[KGEdge] = Field(
        ...,
        description=(
            "Directed relationships between all nodes."
        ),
    )

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

    @model_validator(mode="after")
    def validate_graph_integrity(
        self,
        info: ValidationInfo,
    ) -> "KnowledgeGraph":
        if relation_semantics_already_validated(
            info.context
        ):
            return validate_graph_integrity_compat(
                self,
                validate_legacy_relations=False,
            )

        return validate_graph_integrity_compat(self)
