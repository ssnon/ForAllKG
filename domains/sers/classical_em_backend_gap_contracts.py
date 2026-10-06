from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlBlocker,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSEMBackendGapKind = Literal[
    "backend_capability_gap",
    "reference_detail_gap",
]

SERSEMBackendModelAxis = Literal[
    "particle_shape",
    "shell_geometry",
    "environment",
    "illumination",
    "polarization",
    "observable",
    "reference_detail",
]

SERSEMBackendGapPriority = Literal[
    "insufficient_independent_source_support",
    "cross_source_gap_candidate",
    "reference_curation_required",
]

SERSEMBackendExtensionDecision = Literal[
    "collect_more_independent_controls",
    "targeted_extension_candidate_available",
    "cross_source_gaps_require_joint_design",
    "no_backend_extension_indicated",
]


class SERSEMBackendGapAssessment(StrictModel):
    blocker: SERSEMPositiveControlBlocker
    gap_kind: SERSEMBackendGapKind
    model_axis: SERSEMBackendModelAxis
    extension_target: str = Field(min_length=1)

    affected_control_ids: list[str] = Field(default_factory=list)
    affected_source_locators: list[str] = Field(default_factory=list)
    affected_control_count: int
    independent_source_count: int

    single_axis_unlock_control_ids: list[str] = Field(default_factory=list)
    single_axis_unlock_count: int

    priority_status: SERSEMBackendGapPriority
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _counts(self) -> "SERSEMBackendGapAssessment":
        if self.affected_control_count != len(self.affected_control_ids):
            raise ValueError("affected_control_count does not match ids")
        if self.independent_source_count != len(self.affected_source_locators):
            raise ValueError("independent_source_count does not match source locators")
        if self.single_axis_unlock_count != len(self.single_axis_unlock_control_ids):
            raise ValueError("single_axis_unlock_count does not match ids")
        if len(self.affected_control_ids) != len(set(self.affected_control_ids)):
            raise ValueError("affected control ids must be unique")
        if len(self.affected_source_locators) != len(set(self.affected_source_locators)):
            raise ValueError("affected source locators must be unique")
        if len(self.single_axis_unlock_control_ids) != len(set(self.single_axis_unlock_control_ids)):
            raise ValueError("single-axis unlock control ids must be unique")
        if not set(self.single_axis_unlock_control_ids) <= set(self.affected_control_ids):
            raise ValueError("single-axis unlock controls must be affected controls")
        if self.gap_kind == "reference_detail_gap":
            if self.priority_status != "reference_curation_required":
                raise ValueError("reference-detail gap requires curation priority")
        return self


class SERSEMBackendGapAssessmentBundle(StrictModel):
    schema_version: Literal[
        "sers-em-backend-gap-assessment-bundle-v0"
    ] = "sers-em-backend-gap-assessment-bundle-v0"

    bundle_id: str = Field(min_length=1)
    source_qualification_bundle_id: str = Field(min_length=1)
    backend_profile: str = Field(min_length=1)

    control_count: int
    independent_source_count: int
    qualified_control_count: int
    model_extension_control_count: int
    insufficient_detail_control_count: int

    gap_assessments: list[SERSEMBackendGapAssessment] = Field(default_factory=list)
    shared_blockers_across_model_extension_controls: list[
        SERSEMPositiveControlBlocker
    ] = Field(default_factory=list)

    extension_decision: SERSEMBackendExtensionDecision
    recommended_next_actions: list[str] = Field(default_factory=list)

    backend_extension_authorized: Literal[False] = False
    model_validation_permitted: Literal[False] = False
    target_hypothesis_verdict_permitted: Literal[False] = False

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _consistent(self) -> "SERSEMBackendGapAssessmentBundle":
        if self.control_count < 0 or self.independent_source_count < 0:
            raise ValueError("counts must be non-negative")
        subtotal = (
            self.qualified_control_count
            + self.model_extension_control_count
            + self.insufficient_detail_control_count
        )
        if subtotal != self.control_count:
            raise ValueError("qualification status counts do not sum to control_count")
        blockers = [row.blocker for row in self.gap_assessments]
        if len(blockers) != len(set(blockers)):
            raise ValueError("gap assessments must have unique blockers")
        if len(self.shared_blockers_across_model_extension_controls) != len(
            set(self.shared_blockers_across_model_extension_controls)
        ):
            raise ValueError("shared blockers must be unique")
        return self
