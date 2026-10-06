from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSFDTDApplicability = Literal[
    "direct",
    "partial",
    "not_applicable",
    "requires_interpretation",
]

SERSFDTDSubclaimKind = Literal[
    "fdtd_computable",
    "non_fdtd",
]

SERSFDTDStudyAxis = Literal[
    "wavelength",
    "gap",
    "geometry",
    "polarization",
]


class SERSFDTDSubclaim(StrictModel):
    subclaim_id: str = Field(min_length=1)
    kind: SERSFDTDSubclaimKind
    statement: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    target_observables: list[str] = Field(default_factory=list)
    source_fields: list[str] = Field(default_factory=list)


class SERSFDTDApplicabilityReport(StrictModel):
    schema_version: Literal[
        "sers-fdtd-applicability-report-v0"
    ] = "sers-fdtd-applicability-report-v0"

    report_id: str = Field(min_length=1)
    analyzer_version: Literal[
        "sers-fdtd-applicability-analyzer-v0"
    ] = "sers-fdtd-applicability-analyzer-v0"

    source_portfolio_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)

    applicability: SERSFDTDApplicability
    rationale: str = Field(min_length=1)

    fdtd_subclaims: list[SERSFDTDSubclaim] = Field(default_factory=list)
    non_fdtd_subclaims: list[SERSFDTDSubclaim] = Field(default_factory=list)
    study_axes: list[SERSFDTDStudyAxis] = Field(default_factory=list)

    baseline_wavelength_nm: float | None = None
    detected_analytes: list[str] = Field(default_factory=list)
    detected_geometry_terms: list[str] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSFDTDApplicabilityBundle(StrictModel):
    schema_version: Literal[
        "sers-fdtd-applicability-bundle-v0"
    ] = "sers-fdtd-applicability-bundle-v0"

    bundle_id: str = Field(min_length=1)
    analyzer_version: Literal[
        "sers-fdtd-applicability-analyzer-v0"
    ] = "sers-fdtd-applicability-analyzer-v0"

    source_portfolio_id: str = Field(min_length=1)
    domain_profile_id: str = Field(min_length=1)
    reports: list[SERSFDTDApplicabilityReport] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
