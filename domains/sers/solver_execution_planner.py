from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from domains.sers.solver_execution_contracts import (
    SERSFDTDSolverJob,
    SERSFDTDSolverJobBundle,
    SERSFDTDWavelengthCaseBinding,
)
from domains.sers.validation_design_contracts import (
    SERSFDTDSimulationCase,
    SERSFDTDSimulationCaseBundle,
)


_PLANNER_VERSION = "sers-fdtd-solver-execution-planner-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _group_key(case: SERSFDTDSimulationCase) -> tuple[object, ...]:
    return (
        case.design_id,
        case.hypothesis_id,
        case.source_spec_id,
        case.geometry_candidate_id,
        case.geometry_family,
        case.core_material,
        case.shell_material,
        case.nanorod_length_nm,
        case.nanorod_diameter_nm,
        case.shell_thickness_nm,
        case.polarization,
        case.surrounding_medium,
        case.baseline_wavelength_nm,
        tuple(case.reported_resonance_wavelengths_nm),
    )


class SERSFDTDSolverExecutionPlanner:
    """Coalesce solver-neutral wavelength cases into broadband Meep jobs.

    The scientific evidence unit remains the S1.3 simulation case. This planner
    changes only execution granularity: wavelength is coalesced when every
    non-wavelength condition is identical. Results must later be mapped back to
    the wavelength-specific source case ids.
    """

    planner_version = _PLANNER_VERSION

    def plan(
        self,
        case_bundle: SERSFDTDSimulationCaseBundle,
    ) -> SERSFDTDSolverJobBundle:
        groups: dict[tuple[object, ...], list[SERSFDTDSimulationCase]] = defaultdict(list)
        for case in case_bundle.cases:
            groups[_group_key(case)].append(case)

        jobs: list[SERSFDTDSolverJob] = []
        seen_case_ids: set[str] = set()

        for key in sorted(groups, key=lambda row: _canonical(row)):
            cases = groups[key]
            wavelengths: dict[float, SERSFDTDSimulationCase] = {}
            for case in cases:
                if case.case_id in seen_case_ids:
                    raise ValueError(f"duplicate source case id: {case.case_id}")
                seen_case_ids.add(case.case_id)
                if case.excitation_wavelength_nm in wavelengths:
                    other = wavelengths[case.excitation_wavelength_nm]
                    raise ValueError(
                        "multiple scientific cases map to the same solver condition "
                        f"and wavelength: {other.case_id}, {case.case_id}"
                    )
                wavelengths[case.excitation_wavelength_nm] = case

            ordered = [wavelengths[value] for value in sorted(wavelengths)]
            exemplar = ordered[0]
            bindings = [
                SERSFDTDWavelengthCaseBinding(
                    wavelength_nm=case.excitation_wavelength_nm,
                    source_case_id=case.case_id,
                )
                for case in ordered
            ]
            job = SERSFDTDSolverJob(
                job_id=_stable_id(
                    "sers_fdtd_solver_job",
                    key,
                    [(row.wavelength_nm, row.source_case_id) for row in bindings],
                    self.planner_version,
                ),
                design_id=exemplar.design_id,
                hypothesis_id=exemplar.hypothesis_id,
                source_spec_id=exemplar.source_spec_id,
                geometry_candidate_id=exemplar.geometry_candidate_id,
                geometry_family=exemplar.geometry_family,
                core_material=exemplar.core_material,
                shell_material=exemplar.shell_material,
                nanorod_length_nm=exemplar.nanorod_length_nm,
                nanorod_diameter_nm=exemplar.nanorod_diameter_nm,
                shell_thickness_nm=exemplar.shell_thickness_nm,
                polarization=exemplar.polarization,
                surrounding_medium=exemplar.surrounding_medium,
                wavelength_case_bindings=bindings,
                baseline_wavelength_nm=exemplar.baseline_wavelength_nm,
                reported_resonance_wavelengths_nm=list(
                    exemplar.reported_resonance_wavelengths_nm
                ),
            )
            jobs.append(job)

        if seen_case_ids != {row.case_id for row in case_bundle.cases}:
            raise ValueError("not every scientific simulation case was assigned")

        jobs.sort(key=lambda row: row.job_id)
        return SERSFDTDSolverJobBundle(
            bundle_id=_stable_id(
                "sers_fdtd_solver_job_bundle",
                case_bundle.bundle_id,
                *(row.job_id for row in jobs),
                self.planner_version,
            ),
            source_simulation_case_bundle_id=case_bundle.bundle_id,
            jobs=jobs,
            scientific_case_count=len(case_bundle.cases),
            solver_job_count=len(jobs),
            coalesced_scientific_case_count=len(case_bundle.cases) - len(jobs),
        )
