from __future__ import annotations

from pydantic import Field, model_validator

from pipeline_core.draft_schema import KnowledgeGraphDraft


BROAD_COMPACT_SCHEMA_ID = "broad-mechanism-graph-draft-v1"


class BroadMechanismGraphDraft(KnowledgeGraphDraft):
    """Compact initial-generation schema for the broad abstract corpus."""

    # The broad abstract policy already forbids measurement plumbing. Keep the
    # two top-level keys so the JSON shape remains familiar to the prompt, but
    # replace the large Measurement/MeasurementGroup definitions with trivial
    # empty-array surfaces.
    measurements: list[str] = Field(
        ...,
        description=(
            "Broad abstract mode forbids Measurement nodes. "
            "Return an empty array."
        ),
    )
    measurement_groups: list[str] = Field(
        ...,
        description=(
            "Broad abstract mode forbids MeasurementGroup nodes. "
            "Return an empty array."
        ),
    )

    @model_validator(mode="after")
    def enforce_disabled_measurement_collections(
        self,
    ) -> "BroadMechanismGraphDraft":
        if self.measurements:
            raise ValueError(
                "Broad compact schema requires measurements=[]"
            )
        if self.measurement_groups:
            raise ValueError(
                "Broad compact schema requires measurement_groups=[]"
            )
        return self

    def to_knowledge_graph_draft(self) -> KnowledgeGraphDraft:
        """Expand back to the canonical draft before deterministic validation."""

        payload = self.model_dump(mode="python")
        payload["measurements"] = []
        payload["measurement_groups"] = []
        return KnowledgeGraphDraft.model_validate(payload)
