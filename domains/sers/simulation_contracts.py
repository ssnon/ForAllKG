from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSGeometryFamily = Literal[
    "sphere_dimer",
    "core_shell_nanorod",
]

SERSStudyKind = Literal[
    "single_candidate",
    "parameter_sweep",
    "wavelength_sweep",
]

SERSMaterial = Literal[
    "Au",
    "Ag",
]

SERSPolarization = Literal[
    "parallel_to_dimer_axis",
    "perpendicular_to_dimer_axis",
    "parallel_to_nanorod_long_axis",
    "perpendicular_to_nanorod_long_axis",
]

SERSSurroundingMedium = Literal[
    "air",
    "water",
]

SERSParameterSourceKind = Literal[
    "hypothesis_text",
    "derived_hypothesis_text",
    "applicability_report",
    "operator_override",
    "policy_default",
]

SERSValidationDisposition = Literal[
    "runnable_shadow",
    "requires_concretization",
    "unsupported",
    "not_applicable",
    "requires_interpretation",
    "invalid",
]

SERSValidationSeverity = Literal[
    "error",
    "warning",
]


class SERSParameterProvenance(StrictModel):
    parameter: str = Field(min_length=1)
    source_kind: SERSParameterSourceKind
    source_field: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    derivation: str | None = None


class SERSSweepDefinition(StrictModel):
    parameter: Literal[
        "gap_nm",
        "excitation_wavelength_nm",
    ]
    values: list[float] = Field(default_factory=list)


class SERSSimulationSpec(StrictModel):
    """Machine-readable SERS simulation intent compiled from one hypothesis.

    v0 remains shadow-only and fail-closed. The spec now records FDTD
    applicability explicitly so a partial hypothesis can expose only its
    classical electromagnetic subclaim without treating the unmodelled portion
    as negative evidence.
    """

    schema_version: Literal[
        "sers-simulation-spec-v0",
        "sers-simulation-spec-v0.1",
    ] = "sers-simulation-spec-v0.1"
    spec_id: str = Field(min_length=1)
    compiler_version: Literal[
        "sers-simulation-spec-compiler-v0",
        "sers-simulation-spec-compiler-v0.1",
    ] = "sers-simulation-spec-compiler-v0.1"

    source_portfolio_id: str = Field(min_length=1)
    source_applicability_report_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)
    source_hypothesis_text_sha256: str = Field(min_length=64, max_length=64)

    fdtd_applicability: Literal[
        "direct",
        "partial",
        "not_applicable",
        "requires_interpretation",
    ]
    fdtd_subclaim_ids: list[str] = Field(default_factory=list)
    non_fdtd_subclaim_ids: list[str] = Field(default_factory=list)
    study_axes: list[Literal[
        "wavelength",
        "gap",
        "geometry",
        "polarization",
    ]] = Field(default_factory=list)

    study_kind: SERSStudyKind
    geometry_family: SERSGeometryFamily | None = None

    # Dimer representation.
    left_material: SERSMaterial | None = None
    right_material: SERSMaterial | None = None
    particle_radius_nm: float | None = None
    gap_nm: float | None = None

    # Core-shell nanorod representation. Au@Ag means Au core / Ag shell.
    core_material: SERSMaterial | None = None
    shell_material: SERSMaterial | None = None
    nanorod_length_nm: float | None = None
    nanorod_diameter_nm: float | None = None
    shell_thickness_nm: float | None = None

    # A baseline/reference wavelength is not automatically a fixed simulation
    # wavelength. For wavelength-shift hypotheses it anchors the comparison.
    baseline_wavelength_nm: float | None = None
    reported_resonance_wavelengths_nm: list[float] = Field(default_factory=list)
    excitation_wavelength_nm: float | None = None
    polarization: SERSPolarization | None = None
    surrounding_medium: SERSSurroundingMedium | None = None

    sweep: SERSSweepDefinition | None = None

    parameter_provenance: list[SERSParameterProvenance] = Field(
        default_factory=list
    )
    unsupported_features: list[str] = Field(default_factory=list)
    compiler_notes: list[str] = Field(default_factory=list)

    # Authority contract: this artifact is descriptive only in v0.x.
    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _unique_parameter_provenance(
        self,
    ) -> "SERSSimulationSpec":
        keys = [
            (
                row.parameter,
                row.source_kind,
                row.source_field,
                row.source_text,
                row.derivation,
            )
            for row in self.parameter_provenance
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate simulation parameter provenance")
        resonances = self.reported_resonance_wavelengths_nm
        if any(value <= 0 for value in resonances):
            raise ValueError("reported resonance wavelengths must be > 0")
        if len(resonances) != len(set(resonances)):
            raise ValueError("reported resonance wavelengths must be unique")
        return self


class SERSSimulationCompilationBundle(StrictModel):
    schema_version: Literal[
        "sers-simulation-compilation-bundle-v0",
        "sers-simulation-compilation-bundle-v0.1",
    ] = "sers-simulation-compilation-bundle-v0.1"
    bundle_id: str = Field(min_length=1)
    compiler_version: Literal[
        "sers-simulation-spec-compiler-v0",
        "sers-simulation-spec-compiler-v0.1",
    ] = "sers-simulation-spec-compiler-v0.1"
    source_portfolio_id: str = Field(min_length=1)
    source_applicability_bundle_id: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)
    specs: list[SERSSimulationSpec] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSSimulationValidationIssue(StrictModel):
    severity: SERSValidationSeverity
    code: str = Field(min_length=1)
    field: str = Field(min_length=1)
    message: str = Field(min_length=1)


class SERSSimulationValidationReport(StrictModel):
    schema_version: Literal[
        "sers-simulation-validation-report-v0",
        "sers-simulation-validation-report-v0.1",
    ] = "sers-simulation-validation-report-v0.1"
    report_id: str = Field(min_length=1)
    source_spec_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    disposition: SERSValidationDisposition
    issues: list[SERSSimulationValidationIssue] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSSimulationValidationBundle(StrictModel):
    schema_version: Literal[
        "sers-simulation-validation-bundle-v0",
        "sers-simulation-validation-bundle-v0.1",
    ] = "sers-simulation-validation-bundle-v0.1"
    bundle_id: str = Field(min_length=1)
    source_compilation_bundle_id: str = Field(min_length=1)
    reports: list[SERSSimulationValidationReport] = Field(
        default_factory=list
    )

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
