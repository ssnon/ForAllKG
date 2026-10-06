from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from domains.sers.simulation_contracts import StrictModel


SERSPeakLocationStatus = Literal[
    "interior_peak",
    "lower_boundary_censored",
    "upper_boundary_censored",
]


class SERSResolutionConvergencePoint(StrictModel):
    resolution_px_per_um: float
    grid_spacing_nm: float
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
    peak_location_status: SERSPeakLocationStatus
    spectrum_wavelength_min_nm: float
    spectrum_wavelength_max_nm: float
    wall_seconds: float | None = None
    peak_rss_mb: float | None = None

    @model_validator(mode="after")
    def _positive(self) -> "SERSResolutionConvergencePoint":
        values = [
            self.resolution_px_per_um,
            self.grid_spacing_nm,
            self.peak_wavelength_nm,
            self.spectrum_wavelength_min_nm,
            self.spectrum_wavelength_max_nm,
        ]
        if any(value <= 0 for value in values):
            raise ValueError("resolution convergence point values must be > 0")
        if self.peak_scattering_cross_section_um2 < 0:
            raise ValueError("peak scattering cross section must be >= 0")
        if self.spectrum_wavelength_min_nm >= self.spectrum_wavelength_max_nm:
            raise ValueError("spectrum wavelength min must be below max")
        return self


class SERSResolutionDriftStep(StrictModel):
    from_resolution_px_per_um: float
    to_resolution_px_per_um: float
    from_peak_location_status: SERSPeakLocationStatus
    to_peak_location_status: SERSPeakLocationStatus
    peak_shift_status: Literal[
        "estimated_interior_to_interior",
        "not_estimable_boundary_censored",
    ]
    signed_peak_shift_nm: float | None = None
    absolute_peak_shift_nm: float | None = None
    relative_peak_shift_fraction: float | None = None
    peak_cross_section_relative_change: float | None = None
    wall_time_ratio: float | None = None
    peak_rss_ratio: float | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "SERSResolutionDriftStep":
        if self.to_resolution_px_per_um <= self.from_resolution_px_per_um:
            raise ValueError("resolution drift steps must be ascending")
        if self.peak_shift_status == "estimated_interior_to_interior":
            if (
                self.signed_peak_shift_nm is None
                or self.absolute_peak_shift_nm is None
                or self.relative_peak_shift_fraction is None
            ):
                raise ValueError("interior drift step requires peak-shift values")
        else:
            if any(
                value is not None
                for value in (
                    self.signed_peak_shift_nm,
                    self.absolute_peak_shift_nm,
                    self.relative_peak_shift_fraction,
                )
            ):
                raise ValueError("boundary-censored drift must not report numeric peak shift")
        return self


class SERSPolarizationResolutionSeries(StrictModel):
    polarization: Literal[
        "parallel_to_nanorod_long_axis",
        "perpendicular_to_nanorod_long_axis",
    ]
    field_component: Literal["Ex", "Ey"]
    points: list[SERSResolutionConvergencePoint] = Field(default_factory=list)
    drift_steps: list[SERSResolutionDriftStep] = Field(default_factory=list)
    series_status: Literal[
        "insufficient_resolution_series",
        "interior_drift_report_only",
        "boundary_limited",
    ]

    @model_validator(mode="after")
    def _counts(self) -> "SERSPolarizationResolutionSeries":
        if not self.points:
            raise ValueError("resolution series requires at least one point")
        if len(self.drift_steps) != max(0, len(self.points) - 1):
            raise ValueError("drift step count must equal point count minus one")
        return self


class SERSResolutionConvergenceObservation(StrictModel):
    schema_version: Literal[
        "sers-resolution-convergence-observation-v0"
    ] = "sers-resolution-convergence-observation-v0"
    observation_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    geometry_candidate_id: str = Field(min_length=1)
    surrounding_medium: str = Field(min_length=1)
    baseline_wavelength_nm: float | None = None
    reported_resonance_wavelengths_nm: list[float] = Field(default_factory=list)
    series: list[SERSPolarizationResolutionSeries] = Field(default_factory=list)
    resolution_count: int
    overall_status: Literal[
        "insufficient_resolution_series",
        "drift_report_only_no_threshold",
        "boundary_limited",
    ]
    assessment_scope: Literal[
        "drift_report_only_no_convergence_threshold"
    ] = "drift_report_only_no_convergence_threshold"
    convergence_claim_permitted: Literal[False] = False
    geometry_selection_permitted: Literal[False] = False
    scientific_interpretation_permitted: Literal[False] = False
    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSResolutionConvergenceBundle(StrictModel):
    schema_version: Literal[
        "sers-resolution-convergence-observation-bundle-v0"
    ] = "sers-resolution-convergence-observation-bundle-v0"
    bundle_id: str = Field(min_length=1)
    observations: list[SERSResolutionConvergenceObservation] = Field(default_factory=list)
    observation_count: int
    source_calibration_observation_count: int
    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSResolutionConvergenceBundle":
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count does not match observations")
        return self
