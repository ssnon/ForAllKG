from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SERSEMEvidenceState = Literal[
    "incomplete_observation",
    "boundary_limited_observation",
    "underresolved_observation",
    "insufficient_resolution_series",
    "coarse_screening_observation",
    "drift_report_only_no_threshold",
    "calibration_candidate_review",
]

SERSEvidenceBlocker = Literal[
    "incomplete_polarization_pair",
    "spectral_boundary_censoring",
    "underresolved_geometry",
    "coarse_only",
    "missing_resolution_series",
    "no_convergence_threshold",
]

SERSEvidenceNextAction = Literal[
    "complete_polarization_pair",
    "resolve_spectral_boundary",
    "improve_spatial_resolution_or_solver_fidelity",
    "add_resolution_point",
    "establish_convergence_acceptance_policy",
    "review_calibration_candidate",
]


class SERSEMSpectrumSnapshot(StrictModel):
    source_calibration_observation_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    source_solver_job_id: str = Field(min_length=1)
    polarization: Literal[
        "parallel_to_nanorod_long_axis",
        "perpendicular_to_nanorod_long_axis",
    ]
    field_component: Literal["Ex", "Ey"]
    resolution_px_per_um: float
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
    peak_wavelength_nm: float
    peak_scattering_cross_section_um2: float
    peak_location_status: Literal[
        "interior_peak",
        "lower_boundary_censored",
        "upper_boundary_censored",
    ]
    spectrum_wavelength_min_nm: float
    spectrum_wavelength_max_nm: float
    parent_quality_gate: Literal[
        "observation_only_underresolved",
        "screening_only_coarse",
        "calibration_candidate_review",
        "incomplete_polarization_pair",
    ]


class SERSSourceResonanceSnapshot(StrictModel):
    source_calibration_observation_id: str = Field(min_length=1)
    source_resonance_wavelength_nm: float
    nearest_observed_field_component: Literal["Ex", "Ey"]
    nearest_observed_peak_wavelength_nm: float
    absolute_distance_nm: float
    relative_distance_fraction: float


class SERSEMConvergenceSummary(StrictModel):
    source_convergence_observation_id: str = Field(min_length=1)
    overall_status: Literal[
        "insufficient_resolution_series",
        "drift_report_only_no_threshold",
        "boundary_limited",
    ]
    resolution_count: int
    series_status_by_field: dict[str, Literal[
        "insufficient_resolution_series",
        "interior_drift_report_only",
        "boundary_limited",
    ]] = Field(default_factory=dict)
    boundary_limited_fields: list[Literal["Ex", "Ey"]] = Field(default_factory=list)
    maximum_reported_interior_peak_shift_nm: float | None = None
    convergence_claim_permitted: Literal[False] = False


class SERSClassicalEMPhysicsEvidence(StrictModel):
    schema_version: Literal[
        "sers-classical-em-physics-evidence-v0"
    ] = "sers-classical-em-physics-evidence-v0"

    evidence_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    source_classical_em_handoff_id: str = Field(min_length=1)
    source_validation_plan_id: str = Field(min_length=1)
    source_validation_route_id: str = Field(min_length=1)
    source_fdtd_subclaim_ids: list[str] = Field(min_length=1)
    source_simulation_spec_id: str = Field(min_length=1)
    source_solver_job_ids: list[str] = Field(min_length=1)
    source_calibration_observation_ids: list[str] = Field(min_length=1)
    source_convergence_observation_id: str | None = None

    geometry_candidate_id: str = Field(min_length=1)
    surrounding_medium: str = Field(min_length=1)
    backend_family: Literal["classical_em"] = "classical_em"
    selected_backend: Literal["fdtd"] = "fdtd"
    evidence_role: Literal[
        "em_observation_for_mechanism_assessment"
    ] = "em_observation_for_mechanism_assessment"
    semantic_scope: Literal[
        "electromagnetic_subclaim_only"
    ] = "electromagnetic_subclaim_only"

    required_for_outcome_assessment: bool = False
    required_for_mechanism_assessment: bool = False

    spectrum_snapshots: list[SERSEMSpectrumSnapshot] = Field(default_factory=list)
    source_resonance_snapshots: list[SERSSourceResonanceSnapshot] = Field(
        default_factory=list
    )
    convergence_summary: SERSEMConvergenceSummary | None = None

    numerical_evidence_state: SERSEMEvidenceState
    numerical_blockers: list[SERSEvidenceBlocker] = Field(default_factory=list)
    next_required_actions: list[SERSEvidenceNextAction] = Field(default_factory=list)

    scientific_interpretation_permitted: Literal[False] = False
    mechanism_support_verdict_permitted: Literal[False] = False
    whole_hypothesis_verdict_permitted: Literal[False] = False
    integrated_sers_outcome_verdict_permitted: Literal[False] = False
    experiment_promotion_permitted: Literal[False] = False
    self_feedback_generation_permitted: Literal[False] = False

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _unique_lineage(self) -> "SERSClassicalEMPhysicsEvidence":
        for values, label in (
            (self.source_fdtd_subclaim_ids, "source_fdtd_subclaim_ids"),
            (self.source_solver_job_ids, "source_solver_job_ids"),
            (
                self.source_calibration_observation_ids,
                "source_calibration_observation_ids",
            ),
            (self.numerical_blockers, "numerical_blockers"),
            (self.next_required_actions, "next_required_actions"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if not self.spectrum_snapshots:
            raise ValueError("classical-EM evidence requires at least one spectrum snapshot")
        return self


class SERSClassicalEMPhysicsEvidenceBundle(StrictModel):
    schema_version: Literal[
        "sers-classical-em-physics-evidence-bundle-v0"
    ] = "sers-classical-em-physics-evidence-bundle-v0"

    bundle_id: str = Field(min_length=1)
    source_handoff_bundle_id: str = Field(min_length=1)
    source_solver_job_bundle_id: str = Field(min_length=1)
    source_calibration_bundle_ids: list[str] = Field(min_length=1)
    source_convergence_bundle_ids: list[str] = Field(default_factory=list)
    evidence: list[SERSClassicalEMPhysicsEvidence] = Field(default_factory=list)
    evidence_count: int
    unobserved_handoff_ids: list[str] = Field(default_factory=list)

    shadow_only: Literal[True] = True
    evidence_authority_created: Literal[False] = False
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSClassicalEMPhysicsEvidenceBundle":
        if self.evidence_count != len(self.evidence):
            raise ValueError("evidence_count does not match evidence")
        ids = [row.evidence_id for row in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence_id values must be unique")
        return self
