from __future__ import annotations

import pytest

from domains.sers.solver_execution_planner import SERSFDTDSolverExecutionPlanner
from domains.sers.validation_design_contracts import (
    SERSFDTDSimulationCase,
    SERSFDTDSimulationCaseBundle,
)


def _case(
    *,
    case_id: str,
    geometry: str = "g1",
    length: float = 85.0,
    diameter: float = 25.0,
    shell: float = 2.5,
    wavelength: float = 512.0,
    polarization: str = "parallel_to_nanorod_long_axis",
    medium: str = "water",
) -> SERSFDTDSimulationCase:
    return SERSFDTDSimulationCase(
        case_id=case_id,
        design_id="design:1",
        hypothesis_id="hypothesis:1",
        source_spec_id="spec:1",
        geometry_candidate_id=geometry,
        geometry_family="core_shell_nanorod",
        core_material="Au",
        shell_material="Ag",
        nanorod_length_nm=length,
        nanorod_diameter_nm=diameter,
        shell_thickness_nm=shell,
        excitation_wavelength_nm=wavelength,
        polarization=polarization,
        surrounding_medium=medium,
        baseline_wavelength_nm=785.0,
        reported_resonance_wavelengths_nm=[512.0, 772.0],
    )


def _bundle(cases):
    return SERSFDTDSimulationCaseBundle(
        bundle_id="case_bundle:1",
        source_validation_design_bundle_id="design_bundle:1",
        cases=cases,
    )


def test_42_scientific_cases_coalesce_to_6_solver_jobs():
    cases = []
    geometries = [
        ("g1", 85.0, 25.0, 2.5),
        ("g2", 80.0, 20.0, 5.0),
        ("g3", 75.0, 15.0, 7.5),
    ]
    wavelengths = [450.0, 512.0, 600.0, 700.0, 772.0, 785.0, 850.0]
    pols = [
        "parallel_to_nanorod_long_axis",
        "perpendicular_to_nanorod_long_axis",
    ]
    i = 0
    for gid, length, diameter, shell in geometries:
        for wavelength in wavelengths:
            for pol in pols:
                i += 1
                cases.append(_case(
                    case_id=f"case:{i}",
                    geometry=gid,
                    length=length,
                    diameter=diameter,
                    shell=shell,
                    wavelength=wavelength,
                    polarization=pol,
                ))

    result = SERSFDTDSolverExecutionPlanner().plan(_bundle(cases))
    assert result.scientific_case_count == 42
    assert result.solver_job_count == 6
    assert result.coalesced_scientific_case_count == 36
    assert all(len(job.wavelength_case_bindings) == 7 for job in result.jobs)
    assert all(
        [x.wavelength_nm for x in job.wavelength_case_bindings] == wavelengths
        for job in result.jobs
    )


def test_polarizations_are_not_coalesced_together():
    cases = [
        _case(case_id="a", wavelength=512.0, polarization="parallel_to_nanorod_long_axis"),
        _case(case_id="b", wavelength=772.0, polarization="parallel_to_nanorod_long_axis"),
        _case(case_id="c", wavelength=512.0, polarization="perpendicular_to_nanorod_long_axis"),
        _case(case_id="d", wavelength=772.0, polarization="perpendicular_to_nanorod_long_axis"),
    ]
    result = SERSFDTDSolverExecutionPlanner().plan(_bundle(cases))
    assert result.solver_job_count == 2


def test_media_are_not_coalesced_together():
    cases = [
        _case(case_id="a", wavelength=512.0, medium="water"),
        _case(case_id="b", wavelength=772.0, medium="water"),
        _case(case_id="c", wavelength=512.0, medium="air"),
        _case(case_id="d", wavelength=772.0, medium="air"),
    ]
    result = SERSFDTDSolverExecutionPlanner().plan(_bundle(cases))
    assert result.solver_job_count == 2


def test_geometry_candidates_are_not_coalesced_together():
    cases = [
        _case(case_id="a", geometry="g1", wavelength=512.0),
        _case(case_id="b", geometry="g1", wavelength=772.0),
        _case(case_id="c", geometry="g2", length=80, diameter=20, shell=5, wavelength=512.0),
        _case(case_id="d", geometry="g2", length=80, diameter=20, shell=5, wavelength=772.0),
    ]
    result = SERSFDTDSolverExecutionPlanner().plan(_bundle(cases))
    assert result.solver_job_count == 2


def test_duplicate_wavelength_same_solver_condition_fails_closed():
    cases = [
        _case(case_id="a", wavelength=512.0),
        _case(case_id="b", wavelength=512.0),
    ]
    with pytest.raises(ValueError, match="same solver condition"):
        SERSFDTDSolverExecutionPlanner().plan(_bundle(cases))


def test_planning_is_deterministic_under_input_order():
    cases = [
        _case(case_id="a", wavelength=512.0),
        _case(case_id="b", wavelength=772.0),
        _case(case_id="c", wavelength=785.0),
    ]
    planner = SERSFDTDSolverExecutionPlanner()
    forward = planner.plan(_bundle(cases))
    reverse = planner.plan(_bundle(list(reversed(cases))))
    assert forward.model_dump(mode="json") == reverse.model_dump(mode="json")
