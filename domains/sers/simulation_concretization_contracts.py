from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.sers.simulation_contracts import (
    SERSPolarization,
    SERSSimulationSpec,
    SERSSurroundingMedium,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSConcretizationRequestStatus = Literal[
    "ready_for_operator_input",
    "not_required",
    "blocked_nonconcretizable",
]


class SERSConcretizationRequirement(StrictModel):
    field: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    unit: str | None = None
    allowed_values: list[str] = Field(default_factory=list)


class SERSConcretizationRequest(StrictModel):
    schema_version: Literal[
        "sers-simulation-concretization-request-v0"
    ] = "sers-simulation-concretization-request-v0"
    request_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_spec_id: str = Field(min_length=1)
    source_validation_report_id: str = Field(min_length=1)
    status: SERSConcretizationRequestStatus
    geometry_family: str | None = None
    study_kind: str = Field(min_length=1)
    baseline_wavelength_nm: float | None = None
    requirements: list[SERSConcretizationRequirement] = Field(
        default_factory=list
    )
    blocked_reason: str | None = None

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _status_consistency(self) -> "SERSConcretizationRequest":
        if self.status == "ready_for_operator_input" and not self.requirements:
            raise ValueError(
                "ready_for_operator_input requires at least one requirement"
            )
        if self.status == "not_required" and self.requirements:
            raise ValueError("not_required must not contain requirements")
        if self.status == "blocked_nonconcretizable" and not (
            self.blocked_reason or ""
        ).strip():
            raise ValueError(
                "blocked_nonconcretizable requires blocked_reason"
            )
        return self


class SERSConcretizationRequestBundle(StrictModel):
    schema_version: Literal[
        "sers-simulation-concretization-request-bundle-v0"
    ] = "sers-simulation-concretization-request-bundle-v0"
    bundle_id: str = Field(min_length=1)
    source_compilation_bundle_id: str = Field(min_length=1)
    source_validation_bundle_id: str = Field(min_length=1)
    requests: list[SERSConcretizationRequest] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSConcretizationOverride(StrictModel):
    """Explicit operator-supplied scientific simulation parameters.

    v0 permits only filling parameters that are absent from the compiled spec.
    It does not permit rewriting hypothesis-derived values.
    """

    hypothesis_id: str = Field(min_length=1)
    source_spec_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    operator_note: str | None = None

    particle_radius_nm: float | None = None
    gap_nm: float | None = None

    # For core_shell_nanorod these are the *core* end-to-end length and
    # core diameter. shell_thickness_nm is the uniform radial shell thickness.
    nanorod_length_nm: float | None = None
    nanorod_diameter_nm: float | None = None
    shell_thickness_nm: float | None = None

    excitation_wavelength_nm: float | None = None
    polarization: SERSPolarization | None = None
    surrounding_medium: SERSSurroundingMedium | None = None
    sweep_values: list[float] | None = None


class SERSConcretizationOverrideBundle(StrictModel):
    schema_version: Literal[
        "sers-simulation-concretization-override-bundle-v0"
    ] = "sers-simulation-concretization-override-bundle-v0"
    overrides: list[SERSConcretizationOverride] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_targets(self) -> "SERSConcretizationOverrideBundle":
        keys = [
            (row.hypothesis_id, row.source_spec_id)
            for row in self.overrides
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate concretization override target")
        return self


class SERSConcretizationRecord(StrictModel):
    schema_version: Literal[
        "sers-simulation-concretization-record-v0"
    ] = "sers-simulation-concretization-record-v0"
    record_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_spec_id: str = Field(min_length=1)
    output_spec_id: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    applied_fields: list[str] = Field(default_factory=list)
    operator_note: str | None = None

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSSimulationConcretizationBundle(StrictModel):
    schema_version: Literal[
        "sers-simulation-concretization-bundle-v0"
    ] = "sers-simulation-concretization-bundle-v0"
    bundle_id: str = Field(min_length=1)
    source_request_bundle_id: str = Field(min_length=1)
    records: list[SERSConcretizationRecord] = Field(default_factory=list)
    specs: list[SERSSimulationSpec] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
