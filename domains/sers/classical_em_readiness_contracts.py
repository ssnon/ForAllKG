from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSEMReadinessState = Literal[
    "numerical_repair_required",
    "model_validation_required",
    "evidence_scope_limited",
    "ready_for_component_review",
]

SERSEMNumericalReadiness = Literal[
    "repair_required",
    "review_candidate",
]

SERSEMModelFormStatus = Literal[
    "operator_concretized_unvalidated",
]

SERSEMPositiveControlStatus = Literal[
    "not_assessed",
]

SERSEMEvidenceScopeStatus = Literal[
    "route_targets_partially_observed",
    "route_targets_observed",
]

SERSEMReadinessBlocker = Literal[
    "numerical_evidence_not_ready",
    "operator_concretized_model_unvalidated",
    "positive_control_not_assessed",
    "route_target_observable_unobserved",
]

SERSEMResearchAction = Literal[
    "resolve_spectral_boundary",
    "complete_polarization_pair",
    "add_resolution_point",
    "establish_convergence_acceptance_policy",
    "establish_positive_control_model_validation",
    "review_operator_concretized_model",
    "expand_classical_em_observables",
    "defer_high_cost_spatial_fidelity_escalation",
    "ready_for_component_review",
]


class SERSEMModelFormSnapshot(StrictModel):
    validation_design_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    operator_note: str | None = None

    geometry_candidate_id: str = Field(min_length=1)
    geometry_candidate_label: str = Field(min_length=1)
    geometry_candidate_rationale: str = Field(min_length=1)
    nanorod_length_nm: float
    nanorod_diameter_nm: float
    shell_thickness_nm: float
    surrounding_medium: str = Field(min_length=1)

    geometry_source_role: Literal[
        "validation_design_assumption"
    ] = "validation_design_assumption"
    surrounding_medium_source_role: Literal[
        "validation_design_assumption"
    ] = "validation_design_assumption"
    wavelength_grid_source_role: Literal[
        "validation_design_assumption"
    ] = "validation_design_assumption"
    polarization_source_role: Literal[
        "validation_design_axis"
    ] = "validation_design_axis"

    source_baseline_wavelength_nm: float | None = None
    source_reported_resonance_wavelengths_nm: list[float] = Field(default_factory=list)

    model_form_status: SERSEMModelFormStatus = "operator_concretized_unvalidated"
    positive_control_status: SERSEMPositiveControlStatus = "not_assessed"

    @model_validator(mode="after")
    def _physical_values(self) -> "SERSEMModelFormSnapshot":
        if self.nanorod_length_nm <= 0 or self.nanorod_diameter_nm <= 0:
            raise ValueError("nanorod dimensions must be > 0")
        if self.shell_thickness_nm <= 0:
            raise ValueError("shell_thickness_nm must be > 0")
        if self.nanorod_length_nm <= self.nanorod_diameter_nm:
            raise ValueError("nanorod length must exceed diameter")
        if self.source_baseline_wavelength_nm is not None and self.source_baseline_wavelength_nm <= 0:
            raise ValueError("source baseline wavelength must be > 0")
        if any(value <= 0 for value in self.source_reported_resonance_wavelengths_nm):
            raise ValueError("source resonance wavelengths must be > 0")
        return self


class SERSClassicalEMReadinessAssessment(StrictModel):
    schema_version: Literal[
        "sers-classical-em-readiness-assessment-v0"
    ] = "sers-classical-em-readiness-assessment-v0"

    assessment_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_evidence_id: str = Field(min_length=1)
    source_validation_plan_id: str = Field(min_length=1)
    source_validation_route_id: str = Field(min_length=1)
    source_validation_design_id: str = Field(min_length=1)
    source_simulation_spec_id: str = Field(min_length=1)

    route_target_observables: list[str] = Field(default_factory=list)
    observed_route_target_observables: list[str] = Field(default_factory=list)
    unobserved_route_target_observables: list[str] = Field(default_factory=list)
    evidence_scope_status: SERSEMEvidenceScopeStatus

    numerical_evidence_state: str = Field(min_length=1)
    numerical_blockers: list[str] = Field(default_factory=list)
    numerical_readiness: SERSEMNumericalReadiness

    model_form: SERSEMModelFormSnapshot

    readiness_blockers: list[SERSEMReadinessBlocker] = Field(default_factory=list)
    immediate_numerical_repairs: list[SERSEMResearchAction] = Field(default_factory=list)
    deferred_high_cost_actions: list[SERSEMResearchAction] = Field(default_factory=list)
    next_research_actions: list[SERSEMResearchAction] = Field(default_factory=list)
    overall_readiness: SERSEMReadinessState

    component_review_permitted: Literal[False] = False
    mechanism_support_verdict_permitted: Literal[False] = False
    whole_hypothesis_verdict_permitted: Literal[False] = False
    experiment_promotion_permitted: Literal[False] = False
    self_feedback_generation_permitted: Literal[False] = False
    high_cost_numerical_escalation_permitted: Literal[False] = False

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _consistency(self) -> "SERSClassicalEMReadinessAssessment":
        for values, label in (
            (self.route_target_observables, "route_target_observables"),
            (self.observed_route_target_observables, "observed_route_target_observables"),
            (self.unobserved_route_target_observables, "unobserved_route_target_observables"),
            (self.readiness_blockers, "readiness_blockers"),
            (self.immediate_numerical_repairs, "immediate_numerical_repairs"),
            (self.deferred_high_cost_actions, "deferred_high_cost_actions"),
            (self.next_research_actions, "next_research_actions"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if set(self.observed_route_target_observables) & set(self.unobserved_route_target_observables):
            raise ValueError("route target observable cannot be both observed and unobserved")
        if set(self.observed_route_target_observables) | set(self.unobserved_route_target_observables) != set(self.route_target_observables):
            raise ValueError("observed/unobserved route targets must partition route_target_observables")
        return self


class SERSClassicalEMReadinessBundle(StrictModel):
    schema_version: Literal[
        "sers-classical-em-readiness-bundle-v0"
    ] = "sers-classical-em-readiness-bundle-v0"

    bundle_id: str = Field(min_length=1)
    source_evidence_bundle_id: str = Field(min_length=1)
    source_validation_plan_bundle_id: str = Field(min_length=1)
    source_validation_design_bundle_id: str = Field(min_length=1)
    assessments: list[SERSClassicalEMReadinessAssessment] = Field(default_factory=list)
    assessment_count: int

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSClassicalEMReadinessBundle":
        if self.assessment_count != len(self.assessments):
            raise ValueError("assessment_count does not match assessments")
        ids = [row.assessment_id for row in self.assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("assessment_id values must be unique")
        return self
