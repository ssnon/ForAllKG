from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from domains.sers.simulation_contracts import StrictModel


SERSPolarizationLabel = Literal[
    "parallel_to_nanorod_long_axis",
    "perpendicular_to_nanorod_long_axis",
]


class SERSObservedResonancePeak(StrictModel):
    wavelength_nm: float
    scattering_cross_section_um2: float
    relative_to_global_peak: float
    peak_kind: Literal["sampled_local_maximum", "global_peak_fallback"]

    @model_validator(mode="after")
    def _validate_peak(self) -> "SERSObservedResonancePeak":
        if self.wavelength_nm <= 0:
            raise ValueError("peak wavelength must be > 0")
        if self.scattering_cross_section_um2 < 0:
            raise ValueError("scattering cross section must be >= 0")
        if not 0 <= self.relative_to_global_peak <= 1 + 1e-9:
            raise ValueError("relative_to_global_peak must be in [0, 1]")
        return self


class SERSPolarizationSpectrumObservation(StrictModel):
    request_id: str = Field(min_length=1)
    source_solver_job_id: str = Field(min_length=1)
    polarization: SERSPolarizationLabel
    field_component: Literal["Ex", "Ey"]
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
    resolution_px_per_um: float
    grid_spacing_nm: float
    wall_seconds: float | None = None
    peak_rss_mb: float | None = None
    spectrum_wavelength_min_nm: float
    spectrum_wavelength_max_nm: float
    peak_location_status: Literal[
        "interior_peak",
        "lower_boundary_censored",
        "upper_boundary_censored",
    ]
    global_peak: SERSObservedResonancePeak
    local_peaks: list[SERSObservedResonancePeak] = Field(default_factory=list)


class SERSPerPolarizationResonanceDistance(StrictModel):
    polarization: SERSPolarizationLabel
    field_component: Literal["Ex", "Ey"]
    source_resonance_wavelength_nm: float
    observed_peak_wavelength_nm: float
    absolute_distance_nm: float
    relative_distance_fraction: float
    observed_peak_relative_to_global: float
    observed_peak_kind: Literal["sampled_local_maximum", "global_peak_fallback"]


class SERSSourceResonanceDistanceObservation(StrictModel):
    source_resonance_wavelength_nm: float
    nearest_observed_polarization: SERSPolarizationLabel
    nearest_observed_field_component: Literal["Ex", "Ey"]
    nearest_observed_peak_wavelength_nm: float
    absolute_distance_nm: float
    relative_distance_fraction: float
    nearest_peak_relative_to_global: float
    nearest_peak_kind: Literal["sampled_local_maximum", "global_peak_fallback"]
    per_polarization: list[SERSPerPolarizationResonanceDistance] = Field(default_factory=list)


class SERSResonanceCalibrationObservation(StrictModel):
    schema_version: Literal[
        "sers-resonance-calibration-observation-v0"
    ] = "sers-resonance-calibration-observation-v0"
    observation_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    geometry_candidate_id: str = Field(min_length=1)
    surrounding_medium: str = Field(min_length=1)
    baseline_wavelength_nm: float | None = None
    reported_resonance_wavelengths_nm: list[float] = Field(default_factory=list)
    spectra: list[SERSPolarizationSpectrumObservation] = Field(default_factory=list)
    paired_polarization_complete: bool
    source_resonance_distances: list[SERSSourceResonanceDistanceObservation] = Field(
        default_factory=list
    )
    quality_gate: Literal[
        "observation_only_underresolved",
        "screening_only_coarse",
        "calibration_candidate_review",
        "incomplete_polarization_pair",
    ]
    calibration_scope: Literal[
        "distance_report_only_no_acceptance_threshold"
    ] = "distance_report_only_no_acceptance_threshold"
    geometry_selection_permitted: Literal[False] = False
    scientific_interpretation_permitted: Literal[False] = False
    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class SERSResonanceCalibrationBundle(StrictModel):
    schema_version: Literal[
        "sers-resonance-calibration-observation-bundle-v0"
    ] = "sers-resonance-calibration-observation-bundle-v0"
    bundle_id: str = Field(min_length=1)
    observations: list[SERSResonanceCalibrationObservation] = Field(default_factory=list)
    observation_count: int
    execution_result_count: int
    shadow_only: Literal[True] = True
    physics_authority_created: Literal[False] = False
    hypothesis_rejection_authority: Literal[False] = False
    feedback_generation_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def _counts(self) -> "SERSResonanceCalibrationBundle":
        if self.observation_count != len(self.observations):
            raise ValueError("observation_count does not match observations")
        return self
