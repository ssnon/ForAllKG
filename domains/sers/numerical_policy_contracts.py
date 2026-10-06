from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from domains.sers.simulation_contracts import StrictModel


SERSFDTDIntendedUse = Literal[
    "execution_sanity",
    "coarse_resonance_calibration",
    "quantitative_physics",
]

SERSFDTDResolutionStatus = Literal[
    "underresolved",
    "coarse",
    "target_resolved",
]

SERSFDTDEvidenceEligibility = Literal[
    "execution_sanity_only",
    "screening_only",
    "calibration_candidate",
    "quantitative_candidate",
]

SERSFDTDResourceDecision = Literal[
    "resource_unknown",
    "run_local_candidate",
    "resource_deferred",
]


class SERSFDTDNumericalPolicyRequirement(StrictModel):
    field: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    unit: str | None = None
    allowed_values: list[str] = Field(default_factory=list)


class SERSFDTDNumericalPolicyRequest(StrictModel):
    schema_version: Literal[
        "sers-fdtd-numerical-policy-request-v0"
    ] = "sers-fdtd-numerical-policy-request-v0"
    request_id: str = Field(min_length=1)
    source_solver_job_bundle_id: str = Field(min_length=1)
    solver_job_count: int
    minimum_shell_thickness_nm: float
    maximum_outer_length_nm: float
    maximum_outer_diameter_nm: float
    wavelength_min_nm: float
    wavelength_max_nm: float
    default_target_cells_per_min_feature: float = 8.0
    target_resolution_for_minimum_shell_px_per_um: float
    requirements: list[SERSFDTDNumericalPolicyRequirement] = Field(default_factory=list)
    numerical_risk_notes: list[str] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _valid_request(self) -> "SERSFDTDNumericalPolicyRequest":
        if self.solver_job_count <= 0:
            raise ValueError("solver_job_count must be > 0")
        for value in (
            self.minimum_shell_thickness_nm,
            self.maximum_outer_length_nm,
            self.maximum_outer_diameter_nm,
            self.wavelength_min_nm,
            self.wavelength_max_nm,
            self.default_target_cells_per_min_feature,
            self.target_resolution_for_minimum_shell_px_per_um,
        ):
            if value <= 0:
                raise ValueError("numerical request values must be > 0")
        if self.wavelength_min_nm >= self.wavelength_max_nm:
            raise ValueError("wavelength_min_nm must be below wavelength_max_nm")
        return self


class SERSFDTDResourceBenchmark(StrictModel):
    """Optional empirical benchmark for rough local resource scaling.

    The estimate is explicitly heuristic. Memory is scaled with voxel count;
    wall time is scaled with voxel count times the resolution ratio, matching
    the usual fixed-physical-domain FDTD trend where both spatial unknowns and
    timestep count grow with resolution.
    """

    reference_resolution_px_per_um: float
    reference_cell_x_um: float
    reference_cell_y_um: float
    reference_cell_z_um: float
    reference_wall_seconds: float
    reference_peak_rss_mb: float
    local_memory_limit_mb: float | None = None
    local_wall_seconds_limit_per_job: float | None = None

    @model_validator(mode="after")
    def _positive(self) -> "SERSFDTDResourceBenchmark":
        values = [
            self.reference_resolution_px_per_um,
            self.reference_cell_x_um,
            self.reference_cell_y_um,
            self.reference_cell_z_um,
            self.reference_wall_seconds,
            self.reference_peak_rss_mb,
        ]
        if any(value <= 0 for value in values):
            raise ValueError("resource benchmark values must be > 0")
        if self.local_memory_limit_mb is not None and self.local_memory_limit_mb <= 0:
            raise ValueError("local_memory_limit_mb must be > 0")
        if (
            self.local_wall_seconds_limit_per_job is not None
            and self.local_wall_seconds_limit_per_job <= 0
        ):
            raise ValueError("local_wall_seconds_limit_per_job must be > 0")
        return self


class SERSFDTDNumericalPolicyOverride(StrictModel):
    source_solver_job_bundle_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    intended_use: SERSFDTDIntendedUse

    # v0 supports the physically faithful 3-D branch only. Lower-dimensional
    # approximations, if introduced later, must be separate model classes.
    dimension: Literal["3d"] = "3d"
    incident_propagation: Literal[
        "normal_to_nanorod_axis"
    ] = "normal_to_nanorod_axis"

    resolution_px_per_um: float
    cells_per_min_feature_target: float = 8.0
    pml_thickness_um: float
    padding_um: float
    spectrum_frequency_points: int
    spectral_guard_band_nm: float = 0.0
    resource_benchmark: SERSFDTDResourceBenchmark | None = None
    operator_note: str | None = None

    @model_validator(mode="after")
    def _valid_policy(self) -> "SERSFDTDNumericalPolicyOverride":
        if self.resolution_px_per_um <= 0:
            raise ValueError("resolution_px_per_um must be > 0")
        if self.cells_per_min_feature_target < 2:
            raise ValueError("cells_per_min_feature_target must be >= 2")
        if self.pml_thickness_um <= 0:
            raise ValueError("pml_thickness_um must be > 0")
        if self.padding_um <= 0:
            raise ValueError("padding_um must be > 0")
        if self.spectrum_frequency_points < 3:
            raise ValueError("spectrum_frequency_points must be >= 3")
        if self.spectral_guard_band_nm < 0:
            raise ValueError("spectral_guard_band_nm must be >= 0")
        return self


class SERSFDTDNumericalPolicyOverrideBundle(StrictModel):
    schema_version: Literal[
        "sers-fdtd-numerical-policy-override-bundle-v0"
    ] = "sers-fdtd-numerical-policy-override-bundle-v0"
    policies: list[SERSFDTDNumericalPolicyOverride] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_policy_ids(self) -> "SERSFDTDNumericalPolicyOverrideBundle":
        ids = [row.policy_id for row in self.policies]
        if len(ids) != len(set(ids)):
            raise ValueError("policy_id values must be unique")
        return self


class SERSFDTDResourceEstimate(StrictModel):
    estimation_method: Literal[
        "empirical_fixed_domain_fdtd_scaling_v0"
    ] = "empirical_fixed_domain_fdtd_scaling_v0"
    estimated_voxel_count: float
    voxel_ratio_to_reference: float
    estimated_peak_rss_mb: float
    estimated_wall_seconds: float
    decision: SERSFDTDResourceDecision
    caveat: str = Field(min_length=1)


class SERSFDTDNumericalJobPlan(StrictModel):
    schema_version: Literal[
        "sers-fdtd-numerical-job-plan-v0"
    ] = "sers-fdtd-numerical-job-plan-v0"
    numerical_plan_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    source_solver_job_id: str = Field(min_length=1)
    source_solver_job_bundle_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    geometry_candidate_id: str = Field(min_length=1)

    intended_use: SERSFDTDIntendedUse
    dimension: Literal["3d"] = "3d"
    incident_propagation: Literal[
        "normal_to_nanorod_axis"
    ] = "normal_to_nanorod_axis"

    outer_length_nm: float
    outer_diameter_nm: float
    minimum_geometric_feature_nm: float
    resolution_px_per_um: float
    grid_spacing_nm: float
    minimum_feature_cells: float
    cells_per_min_feature_target: float
    target_resolution_px_per_um: float
    resolution_status: SERSFDTDResolutionStatus
    evidence_eligibility: SERSFDTDEvidenceEligibility

    pml_thickness_um: float
    padding_um: float
    cell_x_um: float
    cell_y_um: float
    cell_z_um: float

    wavelength_min_nm: float
    wavelength_max_nm: float
    spectrum_frequency_points: int
    scientific_wavelength_min_nm: float | None = None
    scientific_wavelength_max_nm: float | None = None
    spectral_guard_band_nm: float = 0.0
    calibration_wavelength_min_nm: float | None = None
    calibration_wavelength_max_nm: float | None = None
    spectral_observable: Literal[
        "scattering_spectrum"
    ] = "scattering_spectrum"

    resource_estimate: SERSFDTDResourceEstimate | None = None
    execution_status: Literal[
        "numerically_assessed_not_executed"
    ] = "numerically_assessed_not_executed"

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSFDTDNumericalPlanBundle(StrictModel):
    schema_version: Literal[
        "sers-fdtd-numerical-plan-bundle-v0"
    ] = "sers-fdtd-numerical-plan-bundle-v0"
    bundle_id: str = Field(min_length=1)
    source_solver_job_bundle_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    plans: list[SERSFDTDNumericalJobPlan] = Field(default_factory=list)
    solver_job_count: int

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _count_match(self) -> "SERSFDTDNumericalPlanBundle":
        if self.solver_job_count != len(self.plans):
            raise ValueError("solver_job_count does not match plans")
        return self
