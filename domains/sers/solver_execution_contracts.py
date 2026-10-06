from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from domains.sers.simulation_contracts import (
    SERSPolarization,
    SERSSurroundingMedium,
    StrictModel,
)


class SERSFDTDWavelengthCaseBinding(StrictModel):
    wavelength_nm: float
    source_case_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _positive_wavelength(self) -> "SERSFDTDWavelengthCaseBinding":
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be > 0")
        return self


class SERSFDTDSolverJob(StrictModel):
    """A solver-level grouping of scientific simulation cases.

    S1.3 expands the scientific design along wavelength because each requested
    wavelength is a distinct evidence point. Meep does not necessarily need a
    separate time-domain run per wavelength: cases sharing geometry,
    polarization, and medium can be coalesced into one broadband job and mapped
    back to their original case ids after execution.

    v0 intentionally stops before assigning numerical execution settings.
    """

    schema_version: Literal[
        "sers-fdtd-solver-job-v0"
    ] = "sers-fdtd-solver-job-v0"
    job_id: str = Field(min_length=1)
    design_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_spec_id: str = Field(min_length=1)
    geometry_candidate_id: str = Field(min_length=1)

    geometry_family: Literal["core_shell_nanorod"]
    core_material: Literal["Au", "Ag"]
    shell_material: Literal["Au", "Ag"]
    nanorod_length_nm: float
    nanorod_diameter_nm: float
    shell_thickness_nm: float
    polarization: SERSPolarization
    surrounding_medium: SERSSurroundingMedium

    wavelength_case_bindings: list[SERSFDTDWavelengthCaseBinding]
    baseline_wavelength_nm: float | None = None
    reported_resonance_wavelengths_nm: list[float] = Field(default_factory=list)

    execution_strategy: Literal[
        "broadband_frequency_sweep"
    ] = "broadband_frequency_sweep"
    solver_backend: Literal["meep_subprocess"] = "meep_subprocess"
    numerical_policy_status: Literal[
        "requires_numerical_policy"
    ] = "requires_numerical_policy"
    execution_status: Literal[
        "planned_not_executable"
    ] = "planned_not_executable"

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _validate_bindings(self) -> "SERSFDTDSolverJob":
        if self.nanorod_length_nm <= 0 or self.nanorod_diameter_nm <= 0:
            raise ValueError("nanorod dimensions must be > 0")
        if self.shell_thickness_nm <= 0:
            raise ValueError("shell_thickness_nm must be > 0")
        if self.nanorod_length_nm <= self.nanorod_diameter_nm:
            raise ValueError("nanorod core length must exceed core diameter")
        wavelengths = [row.wavelength_nm for row in self.wavelength_case_bindings]
        case_ids = [row.source_case_id for row in self.wavelength_case_bindings]
        if not wavelengths:
            raise ValueError("solver job requires at least one wavelength binding")
        if len(wavelengths) != len(set(wavelengths)):
            raise ValueError("duplicate wavelength binding in solver job")
        if wavelengths != sorted(wavelengths):
            raise ValueError("wavelength bindings must be sorted")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate source_case_id in solver job")
        if any(value <= 0 for value in self.reported_resonance_wavelengths_nm):
            raise ValueError("reported resonance wavelengths must be > 0")
        return self


class SERSFDTDSolverJobBundle(StrictModel):
    schema_version: Literal[
        "sers-fdtd-solver-job-bundle-v0"
    ] = "sers-fdtd-solver-job-bundle-v0"
    bundle_id: str = Field(min_length=1)
    source_simulation_case_bundle_id: str = Field(min_length=1)
    jobs: list[SERSFDTDSolverJob] = Field(default_factory=list)
    scientific_case_count: int
    solver_job_count: int
    coalesced_scientific_case_count: int

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts_match(self) -> "SERSFDTDSolverJobBundle":
        if self.solver_job_count != len(self.jobs):
            raise ValueError("solver_job_count does not match jobs")
        if self.scientific_case_count < self.solver_job_count:
            raise ValueError("scientific_case_count cannot be below solver_job_count")
        expected = self.scientific_case_count - self.solver_job_count
        if self.coalesced_scientific_case_count != expected:
            raise ValueError("coalesced_scientific_case_count mismatch")
        return self
