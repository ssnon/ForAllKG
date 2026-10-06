from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.sers.simulation_contracts import (
    SERSPolarization,
    SERSSurroundingMedium,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSValidationDesignRequestStatus = Literal[
    "ready_for_design_input",
    "not_required",
    "blocked",
]


class SERSValidationDesignRequirement(StrictModel):
    field: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_fields: list[str] = Field(default_factory=list)
    unit: str | None = None
    allowed_values: list[str] = Field(default_factory=list)


class SERSSourceSpectralConstraint(StrictModel):
    baseline_wavelength_nm: float | None = None
    reported_resonance_wavelengths_nm: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _positive_unique(self) -> "SERSSourceSpectralConstraint":
        values = self.reported_resonance_wavelengths_nm
        if any(value <= 0 for value in values):
            raise ValueError("reported resonance wavelengths must be > 0")
        if len(values) != len(set(values)):
            raise ValueError("reported resonance wavelengths must be unique")
        if self.baseline_wavelength_nm is not None and self.baseline_wavelength_nm <= 0:
            raise ValueError("baseline_wavelength_nm must be > 0")
        return self


class SERSValidationDesignRequest(StrictModel):
    schema_version: Literal[
        "sers-validation-design-request-v0"
    ] = "sers-validation-design-request-v0"
    request_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_spec_id: str = Field(min_length=1)
    source_validation_report_id: str = Field(min_length=1)
    status: SERSValidationDesignRequestStatus
    geometry_family: str | None = None
    study_kind: str = Field(min_length=1)
    focal_axes: list[str] = Field(default_factory=list)
    uncertainty_axes: list[str] = Field(default_factory=list)
    source_constraints: SERSSourceSpectralConstraint = Field(
        default_factory=SERSSourceSpectralConstraint
    )
    requirements: list[SERSValidationDesignRequirement] = Field(
        default_factory=list
    )
    blocked_reason: str | None = None

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _status_consistency(self) -> "SERSValidationDesignRequest":
        if self.status == "ready_for_design_input" and not self.requirements:
            raise ValueError("ready_for_design_input requires requirements")
        if self.status == "not_required" and self.requirements:
            raise ValueError("not_required must not contain requirements")
        if self.status == "blocked" and not (self.blocked_reason or "").strip():
            raise ValueError("blocked requires blocked_reason")
        return self


class SERSValidationDesignRequestBundle(StrictModel):
    schema_version: Literal[
        "sers-validation-design-request-bundle-v0"
    ] = "sers-validation-design-request-bundle-v0"
    bundle_id: str = Field(min_length=1)
    source_compilation_bundle_id: str = Field(min_length=1)
    source_validation_bundle_id: str = Field(min_length=1)
    requests: list[SERSValidationDesignRequest] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSGeometryCandidate(StrictModel):
    candidate_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    nanorod_length_nm: float
    nanorod_diameter_nm: float
    shell_thickness_nm: float
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_geometry(self) -> "SERSGeometryCandidate":
        if self.nanorod_length_nm <= 0:
            raise ValueError("nanorod_length_nm must be > 0")
        if self.nanorod_diameter_nm <= 0:
            raise ValueError("nanorod_diameter_nm must be > 0")
        if self.shell_thickness_nm <= 0:
            raise ValueError("shell_thickness_nm must be > 0")
        if self.nanorod_length_nm <= self.nanorod_diameter_nm:
            raise ValueError(
                "core-shell nanorod core length must exceed core diameter"
            )
        return self


class SERSValidationDesignOverride(StrictModel):
    """Operator-authored *design*, not source-derived scientific facts.

    Geometry candidates are correlated tuples on purpose. This prevents an
    accidental Cartesian product of independently chosen length/diameter/shell
    values that could create physically nonsensical structures.
    """

    hypothesis_id: str = Field(min_length=1)
    source_spec_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    operator_note: str | None = None
    geometry_candidates: list[SERSGeometryCandidate] = Field(default_factory=list)
    wavelength_sweep_nm: list[float] = Field(default_factory=list)
    polarizations: list[SERSPolarization] = Field(default_factory=list)
    surrounding_media: list[SERSSurroundingMedium] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_axes(self) -> "SERSValidationDesignOverride":
        candidate_ids = [row.candidate_id for row in self.geometry_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("geometry candidate_id values must be unique")
        if len(self.wavelength_sweep_nm) != len(set(self.wavelength_sweep_nm)):
            raise ValueError("wavelength_sweep_nm values must be unique")
        if len(self.polarizations) != len(set(self.polarizations)):
            raise ValueError("polarizations must be unique")
        if len(self.surrounding_media) != len(set(self.surrounding_media)):
            raise ValueError("surrounding_media must be unique")
        if any(value <= 0 for value in self.wavelength_sweep_nm):
            raise ValueError("wavelength_sweep_nm values must be > 0")
        return self


class SERSValidationDesignOverrideBundle(StrictModel):
    schema_version: Literal[
        "sers-validation-design-override-bundle-v0"
    ] = "sers-validation-design-override-bundle-v0"
    overrides: list[SERSValidationDesignOverride] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_targets(self) -> "SERSValidationDesignOverrideBundle":
        targets = [
            (row.hypothesis_id, row.source_spec_id)
            for row in self.overrides
        ]
        if len(targets) != len(set(targets)):
            raise ValueError("duplicate validation-design override target")
        return self


class SERSValidationDesign(StrictModel):
    schema_version: Literal[
        "sers-validation-design-v0"
    ] = "sers-validation-design-v0"
    design_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_spec_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    operator_note: str | None = None

    geometry_family: Literal["core_shell_nanorod"]
    core_material: Literal["Au", "Ag"]
    shell_material: Literal["Au", "Ag"]

    focal_axes: list[Literal["wavelength"]] = Field(default_factory=list)
    uncertainty_axes: list[Literal[
        "geometry_candidate",
        "polarization",
        "surrounding_medium",
    ]] = Field(default_factory=list)

    source_constraints: SERSSourceSpectralConstraint
    geometry_candidates: list[SERSGeometryCandidate]
    wavelength_sweep_nm: list[float]
    polarizations: list[SERSPolarization]
    surrounding_media: list[SERSSurroundingMedium]

    # Source resonance wavelengths are calibration landmarks, not a criterion
    # with an invented tolerance. Later physics-evidence code can report the
    # spectral distance without declaring pass/fail until a criterion exists.
    calibration_mode: Literal[
        "report_distance_to_source_resonances"
    ] = "report_distance_to_source_resonances"

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSValidationDesignBundle(StrictModel):
    schema_version: Literal[
        "sers-validation-design-bundle-v0"
    ] = "sers-validation-design-bundle-v0"
    bundle_id: str = Field(min_length=1)
    source_request_bundle_id: str = Field(min_length=1)
    designs: list[SERSValidationDesign] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSFDTDSimulationCase(StrictModel):
    schema_version: Literal[
        "sers-fdtd-simulation-case-v0"
    ] = "sers-fdtd-simulation-case-v0"
    case_id: str = Field(min_length=1)
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
    excitation_wavelength_nm: float
    polarization: SERSPolarization
    surrounding_medium: SERSSurroundingMedium

    baseline_wavelength_nm: float | None = None
    reported_resonance_wavelengths_nm: list[float] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSFDTDSimulationCaseBundle(StrictModel):
    schema_version: Literal[
        "sers-fdtd-simulation-case-bundle-v0"
    ] = "sers-fdtd-simulation-case-bundle-v0"
    bundle_id: str = Field(min_length=1)
    source_validation_design_bundle_id: str = Field(min_length=1)
    cases: list[SERSFDTDSimulationCase] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
