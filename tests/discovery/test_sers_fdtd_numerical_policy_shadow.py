from __future__ import annotations

import pytest

from domains.sers.numerical_policy import SERSFDTDNumericalPolicyPlanner
from domains.sers.numerical_policy_contracts import (
    SERSFDTDNumericalPolicyOverride,
    SERSFDTDResourceBenchmark,
)
from domains.sers.solver_execution_contracts import (
    SERSFDTDSolverJob,
    SERSFDTDSolverJobBundle,
    SERSFDTDWavelengthCaseBinding,
)


def _job(*, job_id: str, shell: float, length: float, diameter: float, pol: str):
    return SERSFDTDSolverJob(
        job_id=job_id,
        design_id="design:1",
        hypothesis_id="hypothesis:1",
        source_spec_id="spec:1",
        geometry_candidate_id=f"g:{shell}",
        geometry_family="core_shell_nanorod",
        core_material="Au",
        shell_material="Ag",
        nanorod_length_nm=length,
        nanorod_diameter_nm=diameter,
        shell_thickness_nm=shell,
        polarization=pol,
        surrounding_medium="water",
        wavelength_case_bindings=[
            SERSFDTDWavelengthCaseBinding(wavelength_nm=450.0, source_case_id=f"{job_id}:450"),
            SERSFDTDWavelengthCaseBinding(wavelength_nm=512.0, source_case_id=f"{job_id}:512"),
            SERSFDTDWavelengthCaseBinding(wavelength_nm=772.0, source_case_id=f"{job_id}:772"),
            SERSFDTDWavelengthCaseBinding(wavelength_nm=785.0, source_case_id=f"{job_id}:785"),
            SERSFDTDWavelengthCaseBinding(wavelength_nm=850.0, source_case_id=f"{job_id}:850"),
        ],
        baseline_wavelength_nm=785.0,
        reported_resonance_wavelengths_nm=[512.0, 772.0],
    )


def _bundle():
    jobs = []
    for shell, length, diameter in [
        (2.5, 85.0, 25.0),
        (5.0, 80.0, 20.0),
        (7.5, 75.0, 15.0),
    ]:
        for pol in [
            "parallel_to_nanorod_long_axis",
            "perpendicular_to_nanorod_long_axis",
        ]:
            jobs.append(_job(
                job_id=f"job:{shell}:{pol}",
                shell=shell,
                length=length,
                diameter=diameter,
                pol=pol,
            ))
    return SERSFDTDSolverJobBundle(
        bundle_id="solver_bundle:1",
        source_simulation_case_bundle_id="cases:1",
        jobs=jobs,
        scientific_case_count=30,
        solver_job_count=6,
        coalesced_scientific_case_count=24,
    )


def _policy(*, resolution=400.0, benchmark=None, intended_use="coarse_resonance_calibration"):
    return SERSFDTDNumericalPolicyOverride(
        source_solver_job_bundle_id="solver_bundle:1",
        policy_id="policy:test",
        source_label="test",
        intended_use=intended_use,
        resolution_px_per_um=resolution,
        cells_per_min_feature_target=8.0,
        pml_thickness_um=0.3,
        padding_um=0.2,
        spectrum_frequency_points=201,
        resource_benchmark=benchmark,
    )


def test_request_reports_3200_px_per_um_for_2p5nm_shell_at_8_cells():
    request = SERSFDTDNumericalPolicyPlanner().request(_bundle())
    assert request.minimum_shell_thickness_nm == 2.5
    assert request.target_resolution_for_minimum_shell_px_per_um == pytest.approx(3200.0)
    assert request.solver_job_count == 6
    assert request.wavelength_min_nm == 450.0
    assert request.wavelength_max_nm == 850.0


def test_400_px_per_um_audits_shells_as_1_2_3_cells():
    result = SERSFDTDNumericalPolicyPlanner().apply(_bundle(), _policy())
    by_shell = {}
    for plan in result.plans:
        by_shell.setdefault(plan.minimum_geometric_feature_nm, plan)
    assert by_shell[2.5].minimum_feature_cells == pytest.approx(1.0)
    assert by_shell[2.5].resolution_status == "underresolved"
    assert by_shell[5.0].minimum_feature_cells == pytest.approx(2.0)
    assert by_shell[5.0].resolution_status == "coarse"
    assert by_shell[7.5].minimum_feature_cells == pytest.approx(3.0)
    assert by_shell[7.5].resolution_status == "coarse"
    assert all(row.evidence_eligibility == "screening_only" for row in result.plans)


def test_3200_px_per_um_resolves_thinnest_shell_at_target_cells():
    result = SERSFDTDNumericalPolicyPlanner().apply(
        _bundle(),
        _policy(resolution=3200.0, intended_use="quantitative_physics"),
    )
    thin = next(row for row in result.plans if row.minimum_geometric_feature_nm == 2.5)
    assert thin.minimum_feature_cells == pytest.approx(8.0)
    assert thin.resolution_status == "target_resolved"
    assert thin.evidence_eligibility == "quantitative_candidate"


def test_empirical_resource_scaling_can_defer_expensive_job():
    benchmark = SERSFDTDResourceBenchmark(
        reference_resolution_px_per_um=100.0,
        reference_cell_x_um=1.2,
        reference_cell_y_um=1.2,
        reference_cell_z_um=1.2,
        reference_wall_seconds=388.0895,
        reference_peak_rss_mb=261.5664,
        local_memory_limit_mb=24000.0,
        local_wall_seconds_limit_per_job=7200.0,
    )
    result = SERSFDTDNumericalPolicyPlanner().apply(
        _bundle(),
        _policy(resolution=400.0, benchmark=benchmark),
    )
    assert all(row.resource_estimate is not None for row in result.plans)
    assert any(
        row.resource_estimate.decision == "resource_deferred"
        for row in result.plans
        if row.resource_estimate is not None
    )


def test_no_resource_benchmark_remains_resource_unknown_without_blocking_audit():
    result = SERSFDTDNumericalPolicyPlanner().apply(_bundle(), _policy())
    assert all(row.resource_estimate is None for row in result.plans)
    assert len(result.plans) == 6


def test_wrong_source_bundle_fails_closed():
    policy = _policy().model_copy(update={"source_solver_job_bundle_id": "other"})
    with pytest.raises(ValueError, match="does not match"):
        SERSFDTDNumericalPolicyPlanner().apply(_bundle(), policy)


def test_numerical_planning_is_deterministic():
    planner = SERSFDTDNumericalPolicyPlanner()
    first = planner.apply(_bundle(), _policy())
    second = planner.apply(_bundle(), _policy())
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_spectral_guard_band_expands_solver_window_without_changing_scientific_bounds():
    policy = _policy().model_copy(update={"spectral_guard_band_nm": 50.0})
    result = SERSFDTDNumericalPolicyPlanner().apply(_bundle(), policy)
    plan = result.plans[0]
    assert plan.wavelength_min_nm == pytest.approx(450.0)
    assert plan.wavelength_max_nm == pytest.approx(850.0)
    assert plan.scientific_wavelength_min_nm == pytest.approx(450.0)
    assert plan.scientific_wavelength_max_nm == pytest.approx(850.0)
    assert plan.calibration_wavelength_min_nm == pytest.approx(400.0)
    assert plan.calibration_wavelength_max_nm == pytest.approx(900.0)
    assert plan.spectral_guard_band_nm == pytest.approx(50.0)
