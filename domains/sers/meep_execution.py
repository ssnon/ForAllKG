from __future__ import annotations

import hashlib
import json

from domains.sers.meep_execution_contracts import (
    SERSMeepExecutionRequest,
    SERSMeepExecutionRequestBundle,
    SERSMeepWavelengthBinding,
)
from domains.sers.numerical_policy_contracts import SERSFDTDNumericalPlanBundle
from domains.sers.solver_execution_contracts import SERSFDTDSolverJobBundle


_EXECUTION_VERSION = "sers-meep-execution-planner-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _background(medium: str) -> tuple[float, str]:
    if medium == "air":
        return 1.0, "air_nondispersive_n1"
    if medium == "water":
        return 1.33, "water_nondispersive_n1p33"
    raise ValueError(f"unsupported surrounding medium {medium!r}")


def _field_component(polarization: str) -> str:
    if polarization == "parallel_to_nanorod_long_axis":
        return "Ex"
    if polarization == "perpendicular_to_nanorod_long_axis":
        return "Ey"
    raise ValueError(
        "Meep nanorod backend only supports nanorod longitudinal/transverse polarization"
    )


class SERSMeepExecutionPlanner:
    planner_version = _EXECUTION_VERSION

    def build_requests(
        self,
        jobs: SERSFDTDSolverJobBundle,
        numerical: SERSFDTDNumericalPlanBundle,
        *,
        job_ids: set[str] | None = None,
        geometry_candidate_id: str | None = None,
        polarization: str | None = None,
        max_requests: int | None = None,
    ) -> SERSMeepExecutionRequestBundle:
        if numerical.source_solver_job_bundle_id != jobs.bundle_id:
            raise ValueError(
                "numerical plan bundle does not refer to the supplied solver job bundle"
            )

        by_id = {row.job_id: row for row in jobs.jobs}
        if len(by_id) != len(jobs.jobs):
            raise ValueError("solver job ids must be unique")

        selected = []
        for plan in numerical.plans:
            job = by_id.get(plan.source_solver_job_id)
            if job is None:
                raise ValueError(
                    f"numerical plan refers to missing solver job {plan.source_solver_job_id}"
                )
            if job_ids is not None and job.job_id not in job_ids:
                continue
            if geometry_candidate_id is not None and job.geometry_candidate_id != geometry_candidate_id:
                continue
            if polarization is not None and job.polarization != polarization:
                continue
            selected.append((plan, job))

        selected.sort(key=lambda pair: pair[0].numerical_plan_id)
        if max_requests is not None:
            if max_requests <= 0:
                raise ValueError("max_requests must be > 0")
            selected = selected[:max_requests]
        if not selected:
            raise ValueError("no numerical plans matched the requested execution selection")

        requests: list[SERSMeepExecutionRequest] = []
        for plan, job in selected:
            bg_index, bg_model = _background(job.surrounding_medium)
            resource_decision = (
                plan.resource_estimate.decision
                if plan.resource_estimate is not None
                else "resource_unknown"
            )
            bindings = [
                SERSMeepWavelengthBinding(
                    wavelength_nm=row.wavelength_nm,
                    source_case_id=row.source_case_id,
                )
                for row in job.wavelength_case_bindings
            ]
            request_seed = {
                "plan": plan.numerical_plan_id,
                "job": job.job_id,
                "planner": self.planner_version,
            }
            scientific_min_nm = (
                plan.scientific_wavelength_min_nm
                if plan.scientific_wavelength_min_nm is not None
                else plan.wavelength_min_nm
            )
            scientific_max_nm = (
                plan.scientific_wavelength_max_nm
                if plan.scientific_wavelength_max_nm is not None
                else plan.wavelength_max_nm
            )
            solver_min_nm = (
                plan.calibration_wavelength_min_nm
                if plan.calibration_wavelength_min_nm is not None
                else plan.wavelength_min_nm
            )
            solver_max_nm = (
                plan.calibration_wavelength_max_nm
                if plan.calibration_wavelength_max_nm is not None
                else plan.wavelength_max_nm
            )
            requests.append(SERSMeepExecutionRequest(
                request_id=_stable_id("sers_meep_request", request_seed),
                source_numerical_plan_id=plan.numerical_plan_id,
                source_numerical_plan_bundle_id=numerical.bundle_id,
                source_solver_job_id=job.job_id,
                source_solver_job_bundle_id=jobs.bundle_id,
                hypothesis_id=job.hypothesis_id,
                geometry_candidate_id=job.geometry_candidate_id,
                intended_use=plan.intended_use,
                resolution_status=plan.resolution_status,
                evidence_eligibility=plan.evidence_eligibility,
                resource_decision=resource_decision,
                geometry_family=job.geometry_family,
                core_material=job.core_material,
                shell_material=job.shell_material,
                nanorod_length_nm=job.nanorod_length_nm,
                nanorod_diameter_nm=job.nanorod_diameter_nm,
                shell_thickness_nm=job.shell_thickness_nm,
                polarization=job.polarization,
                surrounding_medium=job.surrounding_medium,
                background_index=bg_index,
                background_model=bg_model,
                wavelength_case_bindings=bindings,
                baseline_wavelength_nm=job.baseline_wavelength_nm,
                reported_resonance_wavelengths_nm=job.reported_resonance_wavelengths_nm,
                resolution_px_per_um=plan.resolution_px_per_um,
                pml_thickness_um=plan.pml_thickness_um,
                padding_um=plan.padding_um,
                cell_x_um=plan.cell_x_um,
                cell_y_um=plan.cell_y_um,
                cell_z_um=plan.cell_z_um,
                spectrum_frequency_points=plan.spectrum_frequency_points,
                wavelength_min_nm=solver_min_nm,
                wavelength_max_nm=solver_max_nm,
                scientific_wavelength_min_nm=scientific_min_nm,
                scientific_wavelength_max_nm=scientific_max_nm,
                spectral_guard_band_nm=plan.spectral_guard_band_nm,
                field_component=_field_component(job.polarization),
            ))

        bundle_id = _stable_id(
            "sers_meep_request_bundle",
            numerical.bundle_id,
            *(row.request_id for row in requests),
            self.planner_version,
        )
        return SERSMeepExecutionRequestBundle(
            bundle_id=bundle_id,
            source_numerical_plan_bundle_id=numerical.bundle_id,
            requests=requests,
            request_count=len(requests),
        )

    def assert_execution_allowed(
        self,
        request: SERSMeepExecutionRequest,
        *,
        allow_resource_unknown: bool = False,
        allow_resource_deferred: bool = False,
        allow_non_sanity: bool = False,
    ) -> None:
        if request.intended_use != "execution_sanity" and not allow_non_sanity:
            raise ValueError(
                "v0 execution defaults to execution_sanity plans only; "
                "use an explicit override to execute non-sanity plans"
            )
        if request.resource_decision == "resource_deferred" and not allow_resource_deferred:
            raise ValueError(
                "resource-deferred plan is blocked from local execution by default"
            )
        if request.resource_decision == "resource_unknown" and not allow_resource_unknown:
            raise ValueError(
                "resource-unknown plan is blocked from local execution by default"
            )
