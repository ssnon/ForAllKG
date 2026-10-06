from __future__ import annotations

import pytest

from domains.sers.classical_em_evidence import SERSClassicalEMEvidenceAssembler
from domains.sers.classical_em_handoff_contracts import (
    SERSClassicalEMHandoff,
    SERSClassicalEMHandoffBundle,
)
from domains.sers.resonance_calibration_contracts import (
    SERSObservedResonancePeak,
    SERSPolarizationSpectrumObservation,
    SERSResonanceCalibrationBundle,
    SERSResonanceCalibrationObservation,
    SERSSourceResonanceDistanceObservation,
)
from domains.sers.resolution_convergence_contracts import (
    SERSPolarizationResolutionSeries,
    SERSResolutionConvergenceBundle,
    SERSResolutionConvergenceObservation,
    SERSResolutionConvergencePoint,
    SERSResolutionDriftStep,
)
from domains.sers.solver_execution_contracts import (
    SERSFDTDSolverJob,
    SERSFDTDSolverJobBundle,
    SERSFDTDWavelengthCaseBinding,
)


HID = "hypothesis:test"
SPEC = "sers_simulation_spec:test"
GEO = "geometry:test"
MEDIUM = "water"


def _handoffs(spec_id: str = SPEC):
    row = SERSClassicalEMHandoff(
        handoff_id="handoff:test",
        hypothesis_id=HID,
        source_validation_plan_id="plan:test",
        source_validation_route_id="route:test",
        source_fdtd_applicability_report_id="app:test",
        source_fdtd_subclaim_ids=["subclaim:em"],
        route_scoped_source_fields=["hypothesis_statement"],
        source_simulation_spec_id=spec_id,
        source_simulation_validation_report_id="simval:test",
        simulation_disposition="requires_concretization",
        handoff_status="compiled_requires_concretization",
        required_for_outcome_assessment=False,
        required_for_mechanism_assessment=True,
    )
    return SERSClassicalEMHandoffBundle(
        bundle_id="handoff_bundle:test",
        source_portfolio_id="portfolio:test",
        source_validation_plan_bundle_id="plan_bundle:test",
        source_fdtd_applicability_bundle_id="app_bundle:test",
        source_simulation_compilation_bundle_id="sim_bundle:test",
        source_simulation_validation_bundle_id="simval_bundle:test",
        handoffs=[row],
        routed_hypothesis_count=1,
    )


def _job(job_id: str, polarization: str, spec_id: str = SPEC):
    return SERSFDTDSolverJob(
        job_id=job_id,
        design_id="design:test",
        hypothesis_id=HID,
        source_spec_id=spec_id,
        geometry_candidate_id=GEO,
        geometry_family="core_shell_nanorod",
        core_material="Au",
        shell_material="Ag",
        nanorod_length_nm=75.0,
        nanorod_diameter_nm=15.0,
        shell_thickness_nm=7.5,
        polarization=polarization,
        surrounding_medium=MEDIUM,
        wavelength_case_bindings=[
            SERSFDTDWavelengthCaseBinding(
                wavelength_nm=512.0,
                source_case_id=f"case:{job_id}:512",
            ),
            SERSFDTDWavelengthCaseBinding(
                wavelength_nm=772.0,
                source_case_id=f"case:{job_id}:772",
            ),
        ],
        baseline_wavelength_nm=785.0,
        reported_resonance_wavelengths_nm=[512.0, 772.0],
    )


def _solver_bundle(spec_id: str = SPEC):
    jobs = [
        _job("job:ex", "parallel_to_nanorod_long_axis", spec_id),
        _job("job:ey", "perpendicular_to_nanorod_long_axis", spec_id),
    ]
    return SERSFDTDSolverJobBundle(
        bundle_id="solver_bundle:test",
        source_simulation_case_bundle_id="cases:test",
        jobs=jobs,
        scientific_case_count=4,
        solver_job_count=2,
        coalesced_scientific_case_count=2,
    )


def _spectrum(*, resolution: float, field: str, peak: float, location: str):
    if field == "Ex":
        polarization = "parallel_to_nanorod_long_axis"
        job_id = "job:ex"
    else:
        polarization = "perpendicular_to_nanorod_long_axis"
        job_id = "job:ey"
    return SERSPolarizationSpectrumObservation(
        request_id=f"request:{field}:{resolution:g}",
        source_solver_job_id=job_id,
        polarization=polarization,
        field_component=field,
        numerical_resolution_status="underresolved",
        evidence_eligibility=(
            "execution_sanity_only" if resolution == 100.0 else "screening_only"
        ),
        resolution_px_per_um=resolution,
        grid_spacing_nm=1000.0 / resolution,
        wall_seconds=1000.0,
        peak_rss_mb=500.0,
        spectrum_wavelength_min_nm=450.0,
        spectrum_wavelength_max_nm=850.0,
        peak_location_status=location,
        global_peak=SERSObservedResonancePeak(
            wavelength_nm=peak,
            scattering_cross_section_um2=0.02,
            relative_to_global_peak=1.0,
            peak_kind=(
                "sampled_local_maximum"
                if location == "interior_peak"
                else "global_peak_fallback"
            ),
        ),
    )


def _calibration(*, observation_id: str, resolution: float, ey_peak: float, ey_location: str):
    ex = _spectrum(
        resolution=resolution,
        field="Ex",
        peak=735.0 if resolution == 100.0 else 683.0,
        location="interior_peak",
    )
    ey = _spectrum(
        resolution=resolution,
        field="Ey",
        peak=ey_peak,
        location=ey_location,
    )
    return SERSResonanceCalibrationObservation(
        observation_id=observation_id,
        hypothesis_id=HID,
        geometry_candidate_id=GEO,
        surrounding_medium=MEDIUM,
        baseline_wavelength_nm=785.0,
        reported_resonance_wavelengths_nm=[512.0, 772.0],
        spectra=[ex, ey],
        paired_polarization_complete=True,
        source_resonance_distances=[
            SERSSourceResonanceDistanceObservation(
                source_resonance_wavelength_nm=772.0,
                nearest_observed_polarization="parallel_to_nanorod_long_axis",
                nearest_observed_field_component="Ex",
                nearest_observed_peak_wavelength_nm=ex.global_peak.wavelength_nm,
                absolute_distance_nm=abs(772.0 - ex.global_peak.wavelength_nm),
                relative_distance_fraction=(
                    abs(772.0 - ex.global_peak.wavelength_nm) / 772.0
                ),
                nearest_peak_relative_to_global=1.0,
                nearest_peak_kind=ex.global_peak.peak_kind,
            )
        ],
        quality_gate="observation_only_underresolved",
    )


def _calibration_bundle(bundle_id: str, row):
    return SERSResonanceCalibrationBundle(
        bundle_id=bundle_id,
        observations=[row],
        observation_count=1,
        execution_result_count=2,
    )


def _point(spectrum):
    return SERSResolutionConvergencePoint(
        resolution_px_per_um=spectrum.resolution_px_per_um,
        grid_spacing_nm=spectrum.grid_spacing_nm,
        numerical_resolution_status=spectrum.numerical_resolution_status,
        evidence_eligibility=spectrum.evidence_eligibility,
        peak_wavelength_nm=spectrum.global_peak.wavelength_nm,
        peak_scattering_cross_section_um2=(
            spectrum.global_peak.scattering_cross_section_um2
        ),
        peak_location_status=spectrum.peak_location_status,
        spectrum_wavelength_min_nm=spectrum.spectrum_wavelength_min_nm,
        spectrum_wavelength_max_nm=spectrum.spectrum_wavelength_max_nm,
        wall_seconds=spectrum.wall_seconds,
        peak_rss_mb=spectrum.peak_rss_mb,
    )


def _convergence(cal100, cal150, *, mutate_peak: bool = False):
    ex100, ey100 = cal100.spectra
    ex150, ey150 = cal150.spectra
    ex_points = [_point(ex100), _point(ex150)]
    ey_points = [_point(ey100), _point(ey150)]
    if mutate_peak:
        ey_points[1] = ey_points[1].model_copy(
            update={"peak_wavelength_nm": ey_points[1].peak_wavelength_nm + 1.0}
        )

    ex_shift = ex_points[1].peak_wavelength_nm - ex_points[0].peak_wavelength_nm
    ex_series = SERSPolarizationResolutionSeries(
        polarization="parallel_to_nanorod_long_axis",
        field_component="Ex",
        points=ex_points,
        drift_steps=[SERSResolutionDriftStep(
            from_resolution_px_per_um=100.0,
            to_resolution_px_per_um=150.0,
            from_peak_location_status="interior_peak",
            to_peak_location_status="interior_peak",
            peak_shift_status="estimated_interior_to_interior",
            signed_peak_shift_nm=ex_shift,
            absolute_peak_shift_nm=abs(ex_shift),
            relative_peak_shift_fraction=abs(ex_shift) / ex_points[0].peak_wavelength_nm,
            peak_cross_section_relative_change=0.0,
            wall_time_ratio=1.0,
            peak_rss_ratio=1.0,
        )],
        series_status="interior_drift_report_only",
    )
    ey_series = SERSPolarizationResolutionSeries(
        polarization="perpendicular_to_nanorod_long_axis",
        field_component="Ey",
        points=ey_points,
        drift_steps=[SERSResolutionDriftStep(
            from_resolution_px_per_um=100.0,
            to_resolution_px_per_um=150.0,
            from_peak_location_status="interior_peak",
            to_peak_location_status="lower_boundary_censored",
            peak_shift_status="not_estimable_boundary_censored",
            peak_cross_section_relative_change=0.0,
            wall_time_ratio=1.0,
            peak_rss_ratio=1.0,
        )],
        series_status="boundary_limited",
    )
    obs = SERSResolutionConvergenceObservation(
        observation_id="convergence:test",
        hypothesis_id=HID,
        geometry_candidate_id=GEO,
        surrounding_medium=MEDIUM,
        baseline_wavelength_nm=785.0,
        reported_resonance_wavelengths_nm=[512.0, 772.0],
        series=[ex_series, ey_series],
        resolution_count=2,
        overall_status="boundary_limited",
    )
    return SERSResolutionConvergenceBundle(
        bundle_id="convergence_bundle:test",
        observations=[obs],
        observation_count=1,
        source_calibration_observation_count=2,
    )


def test_boundary_limited_underresolved_evidence_preserves_no_verdict_authority():
    cal100 = _calibration(
        observation_id="cal:100",
        resolution=100.0,
        ey_peak=471.0,
        ey_location="interior_peak",
    )
    cal150 = _calibration(
        observation_id="cal:150",
        resolution=150.0,
        ey_peak=450.0,
        ey_location="lower_boundary_censored",
    )
    result = SERSClassicalEMEvidenceAssembler().assemble(
        _handoffs(),
        _solver_bundle(),
        [
            _calibration_bundle("cal_bundle:100", cal100),
            _calibration_bundle("cal_bundle:150", cal150),
        ],
        [_convergence(cal100, cal150)],
    )

    assert result.evidence_count == 1
    row = result.evidence[0]
    assert row.numerical_evidence_state == "boundary_limited_observation"
    assert row.numerical_blockers == [
        "spectral_boundary_censoring",
        "underresolved_geometry",
    ]
    assert row.next_required_actions == [
        "resolve_spectral_boundary",
        "improve_spatial_resolution_or_solver_fidelity",
    ]
    assert row.required_for_outcome_assessment is False
    assert row.required_for_mechanism_assessment is True
    assert row.scientific_interpretation_permitted is False
    assert row.mechanism_support_verdict_permitted is False
    assert row.experiment_promotion_permitted is False
    assert row.self_feedback_generation_permitted is False
    assert row.convergence_summary is not None
    assert row.convergence_summary.boundary_limited_fields == ["Ey"]
    assert row.convergence_summary.maximum_reported_interior_peak_shift_nm == 52.0


def test_solver_job_must_descend_from_routed_simulation_spec():
    cal100 = _calibration(
        observation_id="cal:100",
        resolution=100.0,
        ey_peak=471.0,
        ey_location="interior_peak",
    )
    with pytest.raises(ValueError, match="does not descend from routed simulation spec"):
        SERSClassicalEMEvidenceAssembler().assemble(
            _handoffs(spec_id="spec:routed"),
            _solver_bundle(spec_id="spec:old"),
            [_calibration_bundle("cal_bundle:100", cal100)],
        )


def test_convergence_points_must_be_reproducible_from_supplied_calibrations():
    cal100 = _calibration(
        observation_id="cal:100",
        resolution=100.0,
        ey_peak=471.0,
        ey_location="interior_peak",
    )
    cal150 = _calibration(
        observation_id="cal:150",
        resolution=150.0,
        ey_peak=450.0,
        ey_location="lower_boundary_censored",
    )
    with pytest.raises(ValueError, match="not reproducible from supplied calibration"):
        SERSClassicalEMEvidenceAssembler().assemble(
            _handoffs(),
            _solver_bundle(),
            [
                _calibration_bundle("cal_bundle:100", cal100),
                _calibration_bundle("cal_bundle:150", cal150),
            ],
            [_convergence(cal100, cal150, mutate_peak=True)],
        )


def test_missing_convergence_is_reported_as_numerical_blocker_not_scientific_failure():
    cal100 = _calibration(
        observation_id="cal:100",
        resolution=100.0,
        ey_peak=471.0,
        ey_location="interior_peak",
    )
    result = SERSClassicalEMEvidenceAssembler().assemble(
        _handoffs(),
        _solver_bundle(),
        [_calibration_bundle("cal_bundle:100", cal100)],
    )
    row = result.evidence[0]
    assert row.numerical_evidence_state == "underresolved_observation"
    assert "missing_resolution_series" in row.numerical_blockers
    assert "add_resolution_point" in row.next_required_actions
    assert row.hypothesis_rejection_authority is False


def test_unobserved_handoff_is_retained_without_inventing_evidence():
    extra = _handoffs().handoffs[0].model_copy(update={
        "handoff_id": "handoff:unobserved",
        "hypothesis_id": "hypothesis:unobserved",
        "source_simulation_spec_id": "spec:unobserved",
    })
    handoffs = _handoffs().model_copy(update={
        "handoffs": [_handoffs().handoffs[0], extra],
        "routed_hypothesis_count": 2,
    })
    cal100 = _calibration(
        observation_id="cal:100",
        resolution=100.0,
        ey_peak=471.0,
        ey_location="interior_peak",
    )
    result = SERSClassicalEMEvidenceAssembler().assemble(
        handoffs,
        _solver_bundle(),
        [_calibration_bundle("cal_bundle:100", cal100)],
    )
    assert result.evidence_count == 1
    assert result.unobserved_handoff_ids == ["handoff:unobserved"]
