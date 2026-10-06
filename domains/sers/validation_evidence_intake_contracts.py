from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.sers.validation_orchestration_contracts import (
    SERSRouteEvidenceReadiness,
    SERSRouteEvidenceRelation,
    SERSRouteEvidenceRole,
)
from domains.sers.validation_routing_contracts import SERSValidationRouteKind


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SERSRouteEvidenceSubmission(StrictModel):
    """Human/validator-facing intake record resolved onto a validation route.

    The submission deliberately names hypothesis + route kind rather than forcing
    callers to know an opaque route_id.  When more than one route of the same kind
    exists, route_id becomes mandatory so intake fails closed instead of guessing.
    """

    schema_version: Literal["sers-route-evidence-submission-v0"] = (
        "sers-route-evidence-submission-v0"
    )

    submission_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    route_kind: SERSValidationRouteKind
    route_id: str | None = None

    evidence_role: SERSRouteEvidenceRole
    review_readiness: SERSRouteEvidenceReadiness = "raw"
    relation_to_claim: SERSRouteEvidenceRelation = "not_assessed"

    target_observables: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    source_overlap_with_generation: bool | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> "SERSRouteEvidenceSubmission":
        if self.route_kind in {"classical_em", "unresolved"}:
            raise ValueError(
                "generic route-evidence intake does not accept classical_em or unresolved routes"
            )
        if self.evidence_role == "independent_validation":
            if self.source_overlap_with_generation is not False:
                raise ValueError(
                    "independent_validation requires explicit "
                    "source_overlap_with_generation=false"
                )
            if not self.source_ids and not self.source_paper_ids:
                raise ValueError(
                    "independent_validation requires at least one concrete source identifier"
                )
        if self.review_readiness == "review_ready" and self.relation_to_claim == "not_assessed":
            raise ValueError(
                "review_ready evidence requires an assessed relation_to_claim"
            )
        for values, label in (
            (self.target_observables, "target_observables"),
            (self.source_ids, "source_ids"),
            (self.source_paper_ids, "source_paper_ids"),
            (self.limitations, "limitations"),
            (self.provenance_notes, "provenance_notes"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SERSRouteEvidenceSubmissionBundle(StrictModel):
    schema_version: Literal["sers-route-evidence-submission-bundle-v0"] = (
        "sers-route-evidence-submission-bundle-v0"
    )
    bundle_id: str = Field(min_length=1)
    submissions: list[SERSRouteEvidenceSubmission] = Field(default_factory=list)
    submission_count: int

    @model_validator(mode="after")
    def _counts(self) -> "SERSRouteEvidenceSubmissionBundle":
        if self.submission_count != len(self.submissions):
            raise ValueError("submission_count does not match submissions")
        ids = [row.submission_id for row in self.submissions]
        if len(ids) != len(set(ids)):
            raise ValueError("submission_id values must be unique")
        return self
