from __future__ import annotations

import pytest

from domains.sers.resonance_calibration import (
    SERSCompletedMeepRun,
    SERSResonanceCalibrationEvaluator,
)
from domains.sers.resolution_convergence import SERSResolutionConvergenceEvaluator


def _run(
    *,
    field: str,
    resolution: float,
    values: list[tuple[float, float]],
    wall: float,
    rss: float,
):
    polarization = (
        "parallel_to_nanorod_long_axis"
        if field == "Ex"
        else "perpendicular_to_nanorod_long_axis"
    )
    request_id = f"request:{field}:{resolution}"
    return SERSCompletedMeepRun(
        request={
            "request_id": request_id,
            "source_solver_job_id": f"job:{field}",
            "hypothesis_id": "hypothesis:test",
            "geometry_candidate_id": "geometry:test",
            "surrounding_medium": "water",
            "polarization": polarization,
            "reported_resonance_wavelengths_nm": [512.0, 772.0],
            "baseline_wavelength_nm": 785.0,
        },
        result={
            "request_id": request_id,
            "worker_status": "completed",
            "field_component": field,
            "numerical_resolution_status": "underresolved",
            "evidence_eligibility": "screening_only",
            "resolution_px_per_um": resolution,
            "grid_spacing_nm": 1000.0 / resolution,
            "wall_seconds": wall,
            "peak_rss_mb": rss,
            "spectrum": [
                {"wavelength_nm": wavelength, "scattering_cross_section_um2": value}
                for wavelength, value in values
            ],
        },
    )


def _calibration(resolution: float, *, boundary_transverse: bool):
    ex = _run(
        field="Ex",
        resolution=resolution,
        values=(
            [(450, 1), (600, 3), (735, 8), (800, 3), (850, 1)]
            if resolution == 100
            else [(450, 1), (600, 4), (683, 7.8), (800, 2), (850, 1)]
        ),
        wall=1900 if resolution == 100 else 7000,
        rss=565 if resolution == 100 else 1585,
    )
    if boundary_transverse:
        ey_values = [(450, 5), (500, 4), (600, 2), (850, 0.5)]
    else:
        ey_values = [(450, 2), (471, 5), (550, 3), (850, 0.5)]
    ey = _run(
        field="Ey",
        resolution=resolution,
        values=ey_values,
        wall=1750 if resolution == 100 else 7750,
        rss=566 if resolution == 100 else 1586,
    )
    return SERSResonanceCalibrationEvaluator().evaluate([ex, ey]).observations[0]


def test_reports_interior_peak_drift_without_convergence_threshold():
    bundle = SERSResolutionConvergenceEvaluator().evaluate([
        _calibration(100, boundary_transverse=False),
        _calibration(150, boundary_transverse=True),
    ])
    assert bundle.observation_count == 1
    observation = bundle.observations[0]
    assert observation.overall_status == "boundary_limited"
    assert observation.convergence_claim_permitted is False

    by_field = {row.field_component: row for row in observation.series}
    ex = by_field["Ex"]
    assert ex.series_status == "interior_drift_report_only"
    assert len(ex.drift_steps) == 1
    step = ex.drift_steps[0]
    assert step.peak_shift_status == "estimated_interior_to_interior"
    assert step.signed_peak_shift_nm == pytest.approx(683 - 735)
    assert step.wall_time_ratio == pytest.approx(7000 / 1900)
    assert step.peak_rss_ratio == pytest.approx(1585 / 565)


def test_boundary_censored_peak_has_no_numeric_shift():
    bundle = SERSResolutionConvergenceEvaluator().evaluate([
        _calibration(100, boundary_transverse=False),
        _calibration(150, boundary_transverse=True),
    ])
    ey = next(row for row in bundle.observations[0].series if row.field_component == "Ey")
    assert ey.series_status == "boundary_limited"
    assert ey.points[-1].peak_location_status == "lower_boundary_censored"
    step = ey.drift_steps[0]
    assert step.peak_shift_status == "not_estimable_boundary_censored"
    assert step.signed_peak_shift_nm is None
    assert step.absolute_peak_shift_nm is None
    assert step.relative_peak_shift_fraction is None


def test_single_resolution_is_insufficient_not_converged():
    bundle = SERSResolutionConvergenceEvaluator().evaluate([
        _calibration(100, boundary_transverse=False)
    ])
    observation = bundle.observations[0]
    assert observation.overall_status == "insufficient_resolution_series"
    assert all(row.series_status == "insufficient_resolution_series" for row in observation.series)


def test_duplicate_resolution_for_same_field_fails_closed():
    first = _calibration(100, boundary_transverse=False)
    second = _calibration(100, boundary_transverse=False).model_copy(
        update={"observation_id": "other"}
    )
    with pytest.raises(ValueError, match="duplicate resolution"):
        SERSResolutionConvergenceEvaluator().evaluate([first, second])
