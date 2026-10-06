from __future__ import annotations

from domains.sers.resonance_calibration import (
    SERSCompletedMeepRun,
    SERSResonanceCalibrationEvaluator,
)


def _run(field: str, values: list[tuple[float, float]], status: str = "underresolved"):
    polarization = (
        "parallel_to_nanorod_long_axis"
        if field == "Ex"
        else "perpendicular_to_nanorod_long_axis"
    )
    request_id = f"request:{field}"
    request = {
        "request_id": request_id,
        "source_solver_job_id": f"job:{field}",
        "hypothesis_id": "hypothesis:test",
        "geometry_candidate_id": "geometry:test",
        "surrounding_medium": "water",
        "polarization": polarization,
        "reported_resonance_wavelengths_nm": [512.0, 772.0],
        "baseline_wavelength_nm": 785.0,
    }
    result = {
        "request_id": request_id,
        "worker_status": "completed",
        "field_component": field,
        "numerical_resolution_status": status,
        "evidence_eligibility": "execution_sanity_only" if status == "underresolved" else "screening_only",
        "resolution_px_per_um": 100.0,
        "grid_spacing_nm": 10.0,
        "wall_seconds": 10.0,
        "peak_rss_mb": 100.0,
        "spectrum": [
            {"wavelength_nm": w, "scattering_cross_section_um2": y}
            for w, y in values
        ],
    }
    return SERSCompletedMeepRun(request=request, result=result)


def test_pairs_polarizations_and_reports_distance_without_threshold():
    longitudinal = _run(
        "Ex",
        [(450, 1), (500, 2), (600, 4), (735, 9), (800, 4), (850, 1)],
    )
    transverse = _run(
        "Ey",
        [(450, 4), (471, 8), (520, 5), (650, 2), (772, 1), (850, 0.5)],
    )
    bundle = SERSResonanceCalibrationEvaluator().evaluate([longitudinal, transverse])
    assert bundle.observation_count == 1
    row = bundle.observations[0]
    assert row.paired_polarization_complete is True
    assert row.quality_gate == "observation_only_underresolved"
    assert row.calibration_scope == "distance_report_only_no_acceptance_threshold"
    assert row.geometry_selection_permitted is False
    assert row.scientific_interpretation_permitted is False
    by_source = {x.source_resonance_wavelength_nm: x for x in row.source_resonance_distances}
    assert by_source[512.0].nearest_observed_field_component == "Ey"
    assert by_source[512.0].nearest_observed_peak_wavelength_nm == 471.0
    assert by_source[772.0].nearest_observed_field_component == "Ex"
    assert by_source[772.0].nearest_observed_peak_wavelength_nm == 735.0


def test_missing_polarization_is_explicit_not_silently_completed():
    bundle = SERSResonanceCalibrationEvaluator().evaluate([
        _run("Ex", [(450, 1), (600, 2), (735, 4), (850, 1)])
    ])
    row = bundle.observations[0]
    assert row.paired_polarization_complete is False
    assert row.quality_gate == "incomplete_polarization_pair"


def test_coarse_pair_stays_screening_only():
    bundle = SERSResonanceCalibrationEvaluator().evaluate([
        _run("Ex", [(450, 1), (600, 2), (735, 4), (850, 1)], "coarse"),
        _run("Ey", [(450, 2), (471, 4), (600, 1), (850, 0.2)], "coarse"),
    ])
    assert bundle.observations[0].quality_gate == "screening_only_coarse"


def test_duplicate_field_component_is_rejected():
    evaluator = SERSResonanceCalibrationEvaluator()
    a = _run("Ex", [(450, 1), (600, 2), (735, 4), (850, 1)])
    b = _run("Ex", [(450, 1), (600, 2), (740, 4), (850, 1)])
    try:
        evaluator.evaluate([a, b])
    except ValueError as exc:
        assert "duplicate completed Ex run" in str(exc)
    else:
        raise AssertionError("expected duplicate-field rejection")


def test_noncompleted_worker_result_is_rejected():
    run = _run("Ex", [(450, 1), (600, 2), (735, 4), (850, 1)])
    run.result["worker_status"] = "validated_not_executed"
    try:
        SERSResonanceCalibrationEvaluator().evaluate([run])
    except ValueError as exc:
        assert "completed worker results only" in str(exc)
    else:
        raise AssertionError("expected worker-status rejection")


def test_peak_at_boundary_uses_global_fallback_when_no_local_peak_exists():
    run = _run("Ey", [(450, 4), (500, 3), (600, 2), (850, 1)])
    bundle = SERSResonanceCalibrationEvaluator().evaluate([run])
    spectrum = bundle.observations[0].spectra[0]
    assert spectrum.local_peaks == []
    assert spectrum.global_peak.wavelength_nm == 450.0
    assert spectrum.peak_location_status == "lower_boundary_censored"
    assert spectrum.spectrum_wavelength_min_nm == 450.0
    assert spectrum.spectrum_wavelength_max_nm == 850.0
    distance = bundle.observations[0].source_resonance_distances[0]
    assert distance.nearest_peak_kind == "global_peak_fallback"
