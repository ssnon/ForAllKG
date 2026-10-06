from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from domains.sers.simulation_contracts import (
    SERSPolarization,
    SERSSurroundingMedium,
    StrictModel,
)


class SERSMeepWavelengthBinding(StrictModel):
    wavelength_nm: float
    source_case_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _positive(self) -> "SERSMeepWavelengthBinding":
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be > 0")
        return self


class SERSMeepExecutionRequest(StrictModel):
    schema_version: Literal[
        "sers-meep-execution-request-v0"
    ] = "sers-meep-execution-request-v0"
    request_id: str = Field(min_length=1)
    source_numerical_plan_id: str = Field(min_length=1)
    source_numerical_plan_bundle_id: str = Field(min_length=1)
    source_solver_job_id: str = Field(min_length=1)
    source_solver_job_bundle_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    geometry_candidate_id: str = Field(min_length=1)

    intended_use: Literal[
        "execution_sanity",
        "coarse_resonance_calibration",
        "quantitative_physics",
    ]
    resolution_status: Literal[
        "underresolved",
        "coarse",
        "target_resolved",
    ]
    evidence_eligibility: Literal[
        "execution_sanity_only",
        "screening_only",
        "calibration_candidate",
        "quantitative_candidate",
    ]
    resource_decision: Literal[
        "resource_unknown",
        "run_local_candidate",
        "resource_deferred",
    ]

    geometry_family: Literal["core_shell_nanorod"]
    core_material: Literal["Au", "Ag"]
    shell_material: Literal["Au", "Ag"]
    nanorod_length_nm: float
    nanorod_diameter_nm: float
    shell_thickness_nm: float
    polarization: SERSPolarization
    surrounding_medium: SERSSurroundingMedium
    background_index: float
    background_model: Literal[
        "air_nondispersive_n1",
        "water_nondispersive_n1p33",
    ]

    wavelength_case_bindings: list[SERSMeepWavelengthBinding]
    baseline_wavelength_nm: float | None = None
    reported_resonance_wavelengths_nm: list[float] = Field(default_factory=list)

    resolution_px_per_um: float
    pml_thickness_um: float
    padding_um: float
    cell_x_um: float
    cell_y_um: float
    cell_z_um: float
    spectrum_frequency_points: int
    wavelength_min_nm: float
    wavelength_max_nm: float
    scientific_wavelength_min_nm: float | None = None
    scientific_wavelength_max_nm: float | None = None
    spectral_guard_band_nm: float = 0.0

    incident_propagation: Literal[
        "normal_to_nanorod_axis"
    ] = "normal_to_nanorod_axis"
    coordinate_convention: Literal[
        "nanorod_x_incident_z"
    ] = "nanorod_x_incident_z"
    field_component: Literal["Ex", "Ey"]
    source_profile: Literal[
        "integrated_planewave_gaussian_pulse_v0"
    ] = "integrated_planewave_gaussian_pulse_v0"
    spectral_observable: Literal[
        "scattering_cross_section_um2"
    ] = "scattering_cross_section_um2"
    material_model: Literal[
        "meep_materials_library"
    ] = "meep_materials_library"
    courant: float = 0.25
    run_until_after_sources: float = 100.0

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_request(self) -> "SERSMeepExecutionRequest":
        positive_values = [
            self.nanorod_length_nm,
            self.nanorod_diameter_nm,
            self.shell_thickness_nm,
            self.background_index,
            self.resolution_px_per_um,
            self.pml_thickness_um,
            self.padding_um,
            self.cell_x_um,
            self.cell_y_um,
            self.cell_z_um,
            self.wavelength_min_nm,
            self.wavelength_max_nm,
            self.courant,
            self.run_until_after_sources,
        ]
        if any(value <= 0 for value in positive_values):
            raise ValueError("Meep execution request positive quantities must be > 0")
        if self.nanorod_length_nm <= self.nanorod_diameter_nm:
            raise ValueError("nanorod core length must exceed core diameter")
        if self.wavelength_min_nm >= self.wavelength_max_nm:
            raise ValueError("wavelength_min_nm must be below wavelength_max_nm")
        if self.spectrum_frequency_points < 3:
            raise ValueError("spectrum_frequency_points must be >= 3")
        if self.spectral_guard_band_nm < 0:
            raise ValueError("spectral_guard_band_nm must be >= 0")
        if (self.scientific_wavelength_min_nm is None) != (self.scientific_wavelength_max_nm is None):
            raise ValueError("scientific wavelength bounds must be supplied together")
        if self.scientific_wavelength_min_nm is not None:
            if self.scientific_wavelength_min_nm <= 0:
                raise ValueError("scientific_wavelength_min_nm must be > 0")
            if self.scientific_wavelength_min_nm >= self.scientific_wavelength_max_nm:
                raise ValueError("scientific wavelength min must be below max")
            if self.wavelength_min_nm > self.scientific_wavelength_min_nm + 1e-9:
                raise ValueError("solver wavelength window does not cover scientific minimum")
            if self.wavelength_max_nm < self.scientific_wavelength_max_nm - 1e-9:
                raise ValueError("solver wavelength window does not cover scientific maximum")
        if not 0 < self.courant <= 0.5:
            raise ValueError("courant must be in (0, 0.5]")
        if not self.wavelength_case_bindings:
            raise ValueError("at least one wavelength binding is required")
        wavelengths = [row.wavelength_nm for row in self.wavelength_case_bindings]
        if wavelengths != sorted(wavelengths):
            raise ValueError("wavelength bindings must be sorted")
        if len(wavelengths) != len(set(wavelengths)):
            raise ValueError("wavelength bindings must be unique")
        if min(wavelengths) < self.wavelength_min_nm - 1e-9:
            raise ValueError("binding below wavelength_min_nm")
        if max(wavelengths) > self.wavelength_max_nm + 1e-9:
            raise ValueError("binding above wavelength_max_nm")
        return self


class SERSMeepExecutionRequestBundle(StrictModel):
    schema_version: Literal[
        "sers-meep-execution-request-bundle-v0"
    ] = "sers-meep-execution-request-bundle-v0"
    bundle_id: str = Field(min_length=1)
    source_numerical_plan_bundle_id: str = Field(min_length=1)
    requests: list[SERSMeepExecutionRequest] = Field(default_factory=list)
    request_count: int

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _count(self) -> "SERSMeepExecutionRequestBundle":
        if self.request_count != len(self.requests):
            raise ValueError("request_count does not match requests")
        return self


class SERSMeepExecutionRecord(StrictModel):
    schema_version: Literal[
        "sers-meep-execution-record-v0"
    ] = "sers-meep-execution-record-v0"
    request_id: str = Field(min_length=1)
    source_numerical_plan_id: str = Field(min_length=1)
    source_solver_job_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    geometry_candidate_id: str = Field(min_length=1)
    execution_status: Literal[
        "planned_not_executed",
        "completed",
        "failed",
        "timed_out",
    ]
    command: list[str] = Field(default_factory=list)
    return_code: int | None = None
    wall_seconds: float | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    worker_result_path: str | None = None
    worker_result_sha256: str | None = None
    error_message: str | None = None

    numerical_resolution_status: Literal[
        "underresolved",
        "coarse",
        "target_resolved",
    ]
    evidence_eligibility: Literal[
        "execution_sanity_only",
        "screening_only",
        "calibration_candidate",
        "quantitative_candidate",
    ]

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSMeepExecutionRecordBundle(StrictModel):
    schema_version: Literal[
        "sers-meep-execution-record-bundle-v0"
    ] = "sers-meep-execution-record-bundle-v0"
    source_request_bundle_id: str = Field(min_length=1)
    records: list[SERSMeepExecutionRecord] = Field(default_factory=list)
    record_count: int

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _count(self) -> "SERSMeepExecutionRecordBundle":
        if self.record_count != len(self.records):
            raise ValueError("record_count does not match records")
        return self
