from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from domains.sers.meep_execution import SERSMeepExecutionPlanner
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


def _jobs():
    job = SERSFDTDSolverJob(
        job_id="job:one",
        design_id="design:one",
        hypothesis_id="hypothesis:one",
        source_spec_id="spec:one",
        geometry_candidate_id="outer90x30_shell7p5",
        geometry_family="core_shell_nanorod",
        core_material="Au",
        shell_material="Ag",
        nanorod_length_nm=75.0,
        nanorod_diameter_nm=15.0,
        shell_thickness_nm=7.5,
        polarization="parallel_to_nanorod_long_axis",
        surrounding_medium="water",
        wavelength_case_bindings=[
            SERSFDTDWavelengthCaseBinding(wavelength_nm=450.0, source_case_id="case:450"),
            SERSFDTDWavelengthCaseBinding(wavelength_nm=512.0, source_case_id="case:512"),
            SERSFDTDWavelengthCaseBinding(wavelength_nm=772.0, source_case_id="case:772"),
            SERSFDTDWavelengthCaseBinding(wavelength_nm=785.0, source_case_id="case:785"),
            SERSFDTDWavelengthCaseBinding(wavelength_nm=850.0, source_case_id="case:850"),
        ],
        baseline_wavelength_nm=785.0,
        reported_resonance_wavelengths_nm=[512.0, 772.0],
    )
    return SERSFDTDSolverJobBundle(
        bundle_id="solver_bundle:one",
        source_simulation_case_bundle_id="cases:one",
        jobs=[job],
        scientific_case_count=5,
        solver_job_count=1,
        coalesced_scientific_case_count=4,
    )


def _numerical(*, intended_use="execution_sanity", memory_limit=24000.0, resolution=100.0):
    policy = SERSFDTDNumericalPolicyOverride(
        source_solver_job_bundle_id="solver_bundle:one",
        policy_id="smoke",
        source_label="test",
        intended_use=intended_use,
        resolution_px_per_um=resolution,
        cells_per_min_feature_target=8.0,
        pml_thickness_um=0.2,
        padding_um=0.15,
        spectrum_frequency_points=201,
        resource_benchmark=SERSFDTDResourceBenchmark(
            reference_resolution_px_per_um=100.0,
            reference_cell_x_um=0.6,
            reference_cell_y_um=0.6,
            reference_cell_z_um=0.6,
            reference_wall_seconds=388.0895,
            reference_peak_rss_mb=261.5664,
            local_memory_limit_mb=memory_limit,
            local_wall_seconds_limit_per_job=7200.0,
        ),
    )
    return SERSFDTDNumericalPolicyPlanner().apply(_jobs(), policy)


def test_request_maps_water_and_longitudinal_polarization_explicitly():
    result = SERSMeepExecutionPlanner().build_requests(_jobs(), _numerical())
    request = result.requests[0]
    assert request.background_index == pytest.approx(1.33)
    assert request.background_model == "water_nondispersive_n1p33"
    assert request.field_component == "Ex"
    assert request.coordinate_convention == "nanorod_x_incident_z"
    assert request.resource_decision == "run_local_candidate"
    assert request.resolution_status == "underresolved"
    assert request.evidence_eligibility == "execution_sanity_only"
    assert request.courant == pytest.approx(0.25)


def test_scientific_case_bindings_remain_attached_to_one_broadband_request():
    request = SERSMeepExecutionPlanner().build_requests(_jobs(), _numerical()).requests[0]
    assert [row.wavelength_nm for row in request.wavelength_case_bindings] == [
        450.0, 512.0, 772.0, 785.0, 850.0
    ]
    assert request.reported_resonance_wavelengths_nm == [512.0, 772.0]
    assert request.baseline_wavelength_nm == 785.0


def test_request_generation_is_deterministic():
    planner = SERSMeepExecutionPlanner()
    first = planner.build_requests(_jobs(), _numerical())
    second = planner.build_requests(_jobs(), _numerical())
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_resource_deferred_execution_fails_closed_without_override():
    numerical = _numerical(memory_limit=100.0)
    request = SERSMeepExecutionPlanner().build_requests(_jobs(), numerical).requests[0]
    assert request.resource_decision == "resource_deferred"
    with pytest.raises(ValueError, match="resource-deferred"):
        SERSMeepExecutionPlanner().assert_execution_allowed(request)


def test_non_sanity_execution_fails_closed_without_override():
    request = SERSMeepExecutionPlanner().build_requests(
        _jobs(),
        _numerical(intended_use="coarse_resonance_calibration", resolution=200.0),
    ).requests[0]
    with pytest.raises(ValueError, match="execution_sanity"):
        SERSMeepExecutionPlanner().assert_execution_allowed(
            request,
            allow_resource_deferred=True,
        )


def test_standalone_worker_validate_only_needs_no_meep(tmp_path: Path):
    request = SERSMeepExecutionPlanner().build_requests(_jobs(), _numerical()).requests[0]
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "result.json"
    request_path.write_text(
        json.dumps(request.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    worker = Path(__file__).resolve().parents[2] / "scripts" / "discovery" / "sers_fdtd_meep_worker.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--request",
            str(request_path),
            "--output",
            str(output_path),
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["worker_status"] == "validated_not_executed"
    assert result["summary"]["outer_length_um"] == pytest.approx(0.09)
    assert result["summary"]["outer_diameter_um"] == pytest.approx(0.03)
    assert result["summary"]["grid_spacing_nm"] == pytest.approx(10.0)


def test_guard_band_changes_solver_window_but_preserves_scientific_bindings():
    policy = SERSFDTDNumericalPolicyOverride(
        source_solver_job_bundle_id="solver_bundle:one",
        policy_id="guarded",
        source_label="test",
        intended_use="coarse_resonance_calibration",
        resolution_px_per_um=100.0,
        cells_per_min_feature_target=8.0,
        pml_thickness_um=0.2,
        padding_um=0.15,
        spectrum_frequency_points=201,
        spectral_guard_band_nm=50.0,
    )
    numerical = SERSFDTDNumericalPolicyPlanner().apply(_jobs(), policy)
    request = SERSMeepExecutionPlanner().build_requests(_jobs(), numerical).requests[0]
    assert request.wavelength_min_nm == pytest.approx(400.0)
    assert request.wavelength_max_nm == pytest.approx(900.0)
    assert request.scientific_wavelength_min_nm == pytest.approx(450.0)
    assert request.scientific_wavelength_max_nm == pytest.approx(850.0)
    assert request.spectral_guard_band_nm == pytest.approx(50.0)
    assert [row.wavelength_nm for row in request.wavelength_case_bindings] == [
        450.0, 512.0, 772.0, 785.0, 850.0
    ]
