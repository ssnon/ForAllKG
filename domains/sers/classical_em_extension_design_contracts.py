from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.sers.classical_em_backend_gap_contracts import (
    SERSEMBackendModelAxis,
)
from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlBlocker,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSEMJointExtensionFrontierStatus = Literal[
    "pareto_frontier",
]

SERSEMJointExtensionDesignDecision = Literal[
    "review_joint_extension_frontier",
    "no_joint_extension_candidate",
]


class SERSEMJointExtensionCandidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    blockers_addressed: list[SERSEMPositiveControlBlocker] = Field(
        default_factory=list
    )
    model_axes: list[SERSEMBackendModelAxis] = Field(default_factory=list)
    extension_targets: list[str] = Field(default_factory=list)
    axis_count: int

    unlocked_control_ids: list[str] = Field(default_factory=list)
    unlocked_source_locators: list[str] = Field(default_factory=list)
    unlocked_control_count: int
    unlocked_independent_source_count: int

    remaining_blocked_control_ids: list[str] = Field(default_factory=list)
    remaining_blocked_control_count: int

    unlocks_cross_source_controls: bool
    unlocks_all_model_extension_controls: bool
    frontier_status: SERSEMJointExtensionFrontierStatus = "pareto_frontier"

    backend_extension_authorized: Literal[False] = False
    model_validation_permitted: Literal[False] = False

    @model_validator(mode="after")
    def _consistent(self) -> "SERSEMJointExtensionCandidate":
        if self.axis_count != len(self.model_axes):
            raise ValueError("axis_count does not match model_axes")
        if self.axis_count != len(self.blockers_addressed):
            raise ValueError("v0 requires one backend blocker per model axis")
        if len(self.blockers_addressed) != len(set(self.blockers_addressed)):
            raise ValueError("blockers_addressed must be unique")
        if len(self.model_axes) != len(set(self.model_axes)):
            raise ValueError("model_axes must be unique")
        if len(self.extension_targets) != len(set(self.extension_targets)):
            raise ValueError("extension_targets must be unique")
        if len(self.extension_targets) != self.axis_count:
            raise ValueError("extension_targets must align one-to-one with axes")
        if self.unlocked_control_count != len(self.unlocked_control_ids):
            raise ValueError("unlocked_control_count does not match ids")
        if self.unlocked_independent_source_count != len(
            self.unlocked_source_locators
        ):
            raise ValueError("unlocked source count does not match locators")
        if self.remaining_blocked_control_count != len(
            self.remaining_blocked_control_ids
        ):
            raise ValueError("remaining blocked count does not match ids")
        if set(self.unlocked_control_ids) & set(self.remaining_blocked_control_ids):
            raise ValueError("unlocked and remaining blocked controls must be disjoint")
        if self.unlocks_cross_source_controls != (
            self.unlocked_independent_source_count >= 2
        ):
            raise ValueError("cross-source flag/count mismatch")
        return self


class SERSEMJointExtensionDesignBundle(StrictModel):
    schema_version: Literal[
        "sers-em-joint-extension-design-bundle-v0"
    ] = "sers-em-joint-extension-design-bundle-v0"

    bundle_id: str = Field(min_length=1)
    source_qualification_bundle_id: str = Field(min_length=1)
    source_backend_gap_bundle_id: str = Field(min_length=1)
    backend_profile: str = Field(min_length=1)

    model_extension_control_count: int
    independent_source_count: int
    capability_blocker_count: int

    candidates: list[SERSEMJointExtensionCandidate] = Field(default_factory=list)
    candidate_count: int

    minimum_axis_count_for_any_unlock: int | None = None
    minimum_axis_count_for_cross_source_unlock: int | None = None
    minimum_axis_count_for_full_coverage: int | None = None

    design_decision: SERSEMJointExtensionDesignDecision
    recommended_next_actions: list[str] = Field(default_factory=list)

    backend_extension_authorized: Literal[False] = False
    positive_control_execution_authorized: Literal[False] = False
    model_validation_permitted: Literal[False] = False
    target_hypothesis_verdict_permitted: Literal[False] = False

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSEMJointExtensionDesignBundle":
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count does not match candidates")
        if self.model_extension_control_count < 0 or self.independent_source_count < 0:
            raise ValueError("counts must be non-negative")
        ids = [row.candidate_id for row in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")
        if self.design_decision == "no_joint_extension_candidate" and self.candidates:
            raise ValueError("no candidate decision must not contain candidates")
        return self
