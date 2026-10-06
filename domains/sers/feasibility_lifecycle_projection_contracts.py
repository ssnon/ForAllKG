from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SERSCanonicalLifecycleProjection(StrictModel):
    schema_version: Literal["sers-canonical-lifecycle-projection-v0"] = (
        "sers-canonical-lifecycle-projection-v0"
    )

    projection_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_candidate_decision_id: str = Field(min_length=1)
    source_scientific_review_id: str = Field(min_length=1)
    source_experiment_plan_id: str | None = None
    source_followup_id: str | None = None

    canonical_final_disposition: str = Field(min_length=1)
    sers_research_decision: str = Field(min_length=1)
    pre_experiment_mechanism_verdict: str = Field(min_length=1)
    outcome_verdict: str = Field(min_length=1)
    experiment_review_state: str = Field(min_length=1)

    experiment_plan_status: str | None = None
    experiment_information_priority: str | None = None

    preserved_uncertainties: list[str] = Field(default_factory=list)
    claim_scoped_feedback_actions: list[str] = Field(default_factory=list)
    followup_feedback_actions: list[str] = Field(default_factory=list)
    unresolved_route_kinds: list[str] = Field(default_factory=list)

    interpretation: str = Field(min_length=1)
    authority_conflict_detected: bool = False

    canonical_candidate_rejection_override_permitted: Literal[False] = False
    canonical_candidate_promotion_override_permitted: Literal[False] = False
    canonical_feedback_permitted: Literal[False] = False
    hypothesis_rewrite_permitted: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    shadow_only: Literal[True] = True

    @model_validator(mode="after")
    def _unique_lists(self) -> "SERSCanonicalLifecycleProjection":
        for values, label in (
            (self.preserved_uncertainties, "preserved_uncertainties"),
            (self.claim_scoped_feedback_actions, "claim_scoped_feedback_actions"),
            (self.followup_feedback_actions, "followup_feedback_actions"),
            (self.unresolved_route_kinds, "unresolved_route_kinds"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SERSCanonicalLifecycleProjectionBundle(StrictModel):
    schema_version: Literal["sers-canonical-lifecycle-projection-bundle-v0"] = (
        "sers-canonical-lifecycle-projection-bundle-v0"
    )

    bundle_id: str = Field(min_length=1)
    source_candidate_decision_portfolio_id: str = Field(min_length=1)
    source_scientific_review_bundle_id: str = Field(min_length=1)
    source_experiment_plan_bundle_id: str | None = None
    source_followup_bundle_id: str | None = None
    projections: list[SERSCanonicalLifecycleProjection] = Field(default_factory=list)
    projection_count: int

    canonical_authority_changed: Literal[False] = False
    canonical_feedback_permitted: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    shadow_only: Literal[True] = True

    @model_validator(mode="after")
    def _counts(self) -> "SERSCanonicalLifecycleProjectionBundle":
        if self.projection_count != len(self.projections):
            raise ValueError("projection_count does not match projections")
        ids = [row.projection_id for row in self.projections]
        if len(ids) != len(set(ids)):
            raise ValueError("projection_id values must be unique")
        return self
