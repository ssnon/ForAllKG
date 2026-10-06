from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSEMPositiveControlParticleShape = Literal[
    "spherocylinder_like",
    "cuboid_or_faceted",
    "other_or_uncertain",
]

SERSEMPositiveControlShellGeometry = Literal[
    "uniform_thickness",
    "anisotropic_or_faceted",
    "unknown",
]

SERSEMPositiveControlEnvironment = Literal[
    "homogeneous_water",
    "homogeneous_air",
    "substrate_with_water",
    "substrate_with_air",
    "other_or_uncertain",
]

SERSEMPositiveControlIllumination = Literal[
    "normal_incidence_plane_wave",
    "dark_field_high_na",
    "unpolarized_ensemble",
    "other_or_uncertain",
]

SERSEMPositiveControlPolarization = Literal[
    "parallel_to_nanorod_long_axis",
    "perpendicular_to_nanorod_long_axis",
    "unpolarized",
    "not_reported",
]

SERSEMPositiveControlObservable = Literal[
    "single_particle_scattering_spectrum",
    "ensemble_scattering_spectrum",
    "single_particle_extinction_spectrum",
    "ensemble_extinction_spectrum",
    "other_or_uncertain",
]

SERSEMPositiveControlQualificationStatus = Literal[
    "qualified_for_current_backend",
    "requires_model_extension",
    "insufficient_reference_detail",
]

SERSEMPositiveControlBlocker = Literal[
    "particle_shape_mismatch",
    "shell_geometry_mismatch",
    "environment_mismatch",
    "illumination_mismatch",
    "polarization_mismatch",
    "observable_mismatch",
    "reference_peak_missing",
]


class SERSEMPositiveControlReference(StrictModel):
    """Source-authored facts for one physical-model positive control.

    The reference is intentionally stricter than a generic literature citation.
    A paper is not usable merely because it contains an Au/Ag nanorod and an
    optical spectrum: its geometry, environment, illumination, polarization,
    and measured observable must be representable by the current backend.

    v0 requires explicit human/source extraction. Nothing in this contract
    silently infers a uniform shell from outer/core dimensions or equates
    extinction with scattering.
    """

    schema_version: Literal[
        "sers-em-positive-control-reference-v0"
    ] = "sers-em-positive-control-reference-v0"

    control_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    source_note: str | None = None

    geometry_family: Literal["core_shell_nanorod"] = "core_shell_nanorod"
    core_material: Literal["Au", "Ag"]
    shell_material: Literal["Au", "Ag"]
    particle_shape: SERSEMPositiveControlParticleShape
    shell_geometry: SERSEMPositiveControlShellGeometry

    core_length_nm: float | None = None
    core_diameter_nm: float | None = None
    uniform_shell_thickness_nm: float | None = None

    environment: SERSEMPositiveControlEnvironment
    illumination: SERSEMPositiveControlIllumination
    polarization: SERSEMPositiveControlPolarization
    observable: SERSEMPositiveControlObservable

    reference_peak_wavelength_nm: float | None = None
    reference_peak_label: str | None = None

    independent_reference: bool = True

    @model_validator(mode="after")
    def _physical_values(self) -> "SERSEMPositiveControlReference":
        for value, label in (
            (self.core_length_nm, "core_length_nm"),
            (self.core_diameter_nm, "core_diameter_nm"),
            (self.uniform_shell_thickness_nm, "uniform_shell_thickness_nm"),
            (self.reference_peak_wavelength_nm, "reference_peak_wavelength_nm"),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be > 0 when supplied")
        if (
            self.core_length_nm is not None
            and self.core_diameter_nm is not None
            and self.core_length_nm <= self.core_diameter_nm
        ):
            raise ValueError("core_length_nm must exceed core_diameter_nm")
        if (
            self.shell_geometry != "uniform_thickness"
            and self.uniform_shell_thickness_nm is not None
        ):
            raise ValueError(
                "uniform_shell_thickness_nm is only valid for uniform_thickness shell_geometry"
            )
        return self


class SERSEMPositiveControlReferenceBundle(StrictModel):
    schema_version: Literal[
        "sers-em-positive-control-reference-bundle-v0"
    ] = "sers-em-positive-control-reference-bundle-v0"
    references: list[SERSEMPositiveControlReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "SERSEMPositiveControlReferenceBundle":
        ids = [row.control_id for row in self.references]
        if len(ids) != len(set(ids)):
            raise ValueError("positive-control control_id values must be unique")
        return self


class SERSEMPositiveControlQualificationIssue(StrictModel):
    blocker: SERSEMPositiveControlBlocker
    source_field: str = Field(min_length=1)
    message: str = Field(min_length=1)


class SERSEMPositiveControlQualification(StrictModel):
    schema_version: Literal[
        "sers-em-positive-control-qualification-v0"
    ] = "sers-em-positive-control-qualification-v0"

    qualification_id: str = Field(min_length=1)
    control_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    backend_profile: Literal[
        "meep_core_shell_nanorod_scattering_v0"
    ] = "meep_core_shell_nanorod_scattering_v0"

    status: SERSEMPositiveControlQualificationStatus
    blockers: list[SERSEMPositiveControlBlocker] = Field(default_factory=list)
    issues: list[SERSEMPositiveControlQualificationIssue] = Field(default_factory=list)

    backend_geometry_representable: bool
    backend_environment_representable: bool
    backend_illumination_representable: bool
    backend_polarization_representable: bool
    backend_observable_representable: bool
    spectral_reference_available: bool

    # Qualification is descriptive. Even a qualified reference requires an
    # explicit numerical design, execution, and comparison artifact later.
    positive_control_execution_candidate: bool
    model_validation_permitted: Literal[False] = False
    target_hypothesis_verdict_permitted: Literal[False] = False

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _consistent(self) -> "SERSEMPositiveControlQualification":
        if len(self.blockers) != len(set(self.blockers)):
            raise ValueError("positive-control blockers must be unique")
        issue_blockers = [row.blocker for row in self.issues]
        if set(issue_blockers) != set(self.blockers):
            raise ValueError("issues must cover exactly the declared blockers")
        if self.status == "qualified_for_current_backend":
            if self.blockers:
                raise ValueError("qualified reference must not contain blockers")
            if not self.positive_control_execution_candidate:
                raise ValueError("qualified reference must be an execution candidate")
        else:
            if not self.blockers:
                raise ValueError("non-qualified reference requires blockers")
            if self.positive_control_execution_candidate:
                raise ValueError("blocked reference cannot be an execution candidate")
        return self


class SERSEMPositiveControlQualificationBundle(StrictModel):
    schema_version: Literal[
        "sers-em-positive-control-qualification-bundle-v0"
    ] = "sers-em-positive-control-qualification-bundle-v0"

    bundle_id: str = Field(min_length=1)
    backend_profile: Literal[
        "meep_core_shell_nanorod_scattering_v0"
    ] = "meep_core_shell_nanorod_scattering_v0"
    qualifications: list[SERSEMPositiveControlQualification] = Field(default_factory=list)
    qualification_count: int

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSEMPositiveControlQualificationBundle":
        if self.qualification_count != len(self.qualifications):
            raise ValueError("qualification_count does not match qualifications")
        ids = [row.qualification_id for row in self.qualifications]
        if len(ids) != len(set(ids)):
            raise ValueError("qualification_id values must be unique")
        return self
