from __future__ import annotations

import hashlib
import json

from domains.sers.numerical_policy_contracts import (
    SERSFDTDNumericalJobPlan,
    SERSFDTDNumericalPlanBundle,
    SERSFDTDNumericalPolicyOverride,
    SERSFDTDNumericalPolicyRequest,
    SERSFDTDNumericalPolicyRequirement,
    SERSFDTDResourceEstimate,
)
from domains.sers.solver_execution_contracts import (
    SERSFDTDSolverJob,
    SERSFDTDSolverJobBundle,
)


_POLICY_VERSION = "sers-fdtd-numerical-policy-v0"
_DEFAULT_TARGET_CELLS = 8.0


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


def _outer_dimensions_nm(job: SERSFDTDSolverJob) -> tuple[float, float]:
    return (
        job.nanorod_length_nm + 2.0 * job.shell_thickness_nm,
        job.nanorod_diameter_nm + 2.0 * job.shell_thickness_nm,
    )


def _all_wavelengths(bundle: SERSFDTDSolverJobBundle) -> list[float]:
    values = {
        binding.wavelength_nm
        for job in bundle.jobs
        for binding in job.wavelength_case_bindings
    }
    return sorted(values)


def _resolution_status(cells: float, target: float) -> str:
    if cells < 2.0:
        return "underresolved"
    if cells < target:
        return "coarse"
    return "target_resolved"


def _evidence_eligibility(intended_use: str, status: str) -> str:
    if intended_use == "execution_sanity":
        return "execution_sanity_only"
    if status != "target_resolved":
        return "screening_only"
    if intended_use == "coarse_resonance_calibration":
        return "calibration_candidate"
    return "quantitative_candidate"


def _resource_estimate(
    *,
    policy: SERSFDTDNumericalPolicyOverride,
    cell_x_um: float,
    cell_y_um: float,
    cell_z_um: float,
) -> SERSFDTDResourceEstimate | None:
    benchmark = policy.resource_benchmark
    if benchmark is None:
        return None

    resolution = policy.resolution_px_per_um
    estimated_voxels = (
        cell_x_um * cell_y_um * cell_z_um * resolution ** 3
    )
    reference_voxels = (
        benchmark.reference_cell_x_um
        * benchmark.reference_cell_y_um
        * benchmark.reference_cell_z_um
        * benchmark.reference_resolution_px_per_um ** 3
    )
    voxel_ratio = estimated_voxels / reference_voxels
    resolution_ratio = resolution / benchmark.reference_resolution_px_per_um

    estimated_peak_rss_mb = benchmark.reference_peak_rss_mb * voxel_ratio
    estimated_wall_seconds = (
        benchmark.reference_wall_seconds * voxel_ratio * resolution_ratio
    )

    over_memory = (
        benchmark.local_memory_limit_mb is not None
        and estimated_peak_rss_mb > benchmark.local_memory_limit_mb
    )
    over_wall = (
        benchmark.local_wall_seconds_limit_per_job is not None
        and estimated_wall_seconds > benchmark.local_wall_seconds_limit_per_job
    )
    decision = "resource_deferred" if (over_memory or over_wall) else "run_local_candidate"

    return SERSFDTDResourceEstimate(
        estimated_voxel_count=estimated_voxels,
        voxel_ratio_to_reference=voxel_ratio,
        estimated_peak_rss_mb=estimated_peak_rss_mb,
        estimated_wall_seconds=estimated_wall_seconds,
        decision=decision,
        caveat=(
            "Rough empirical scaling only: memory is scaled by voxel count and "
            "wall time by voxel count times resolution ratio. Broadband decay, "
            "material dispersion, MPI/threading, cache effects, and geometry "
            "complexity can change actual cost substantially."
        ),
    )


class SERSFDTDNumericalPolicyPlanner:
    policy_version = _POLICY_VERSION

    def request(
        self,
        jobs: SERSFDTDSolverJobBundle,
    ) -> SERSFDTDNumericalPolicyRequest:
        if not jobs.jobs:
            raise ValueError("numerical policy requires at least one solver job")

        shells = [row.shell_thickness_nm for row in jobs.jobs]
        outer = [_outer_dimensions_nm(row) for row in jobs.jobs]
        wavelengths = _all_wavelengths(jobs)
        if len(wavelengths) < 2:
            raise ValueError("broadband numerical policy requires >=2 wavelengths")

        min_shell = min(shells)
        target_resolution = 1000.0 * _DEFAULT_TARGET_CELLS / min_shell
        notes = [
            (
                f"Minimum shell thickness is {min_shell:g} nm; resolving it with "
                f"{_DEFAULT_TARGET_CELLS:g} cells requires approximately "
                f"{target_resolution:g} pixels/um."
            ),
            (
                "Resolution adequacy is numerical quality only. An underresolved "
                "or resource-deferred job must not be interpreted as evidence "
                "against the scientific hypothesis."
            ),
            (
                "The first solver observable is a broadband scattering spectrum "
                "for source-resonance calibration. Near-field/SERS metrics belong "
                "to a later selected-geometry stage."
            ),
        ]

        requirements = [
            SERSFDTDNumericalPolicyRequirement(
                field="resolution_px_per_um",
                rationale=(
                    "Choose an explicit grid resolution. The planner reports cells "
                    "per thinnest shell and does not silently upgrade a coarse run "
                    "to quantitative evidence."
                ),
                unit="pixels/um",
            ),
            SERSFDTDNumericalPolicyRequirement(
                field="pml_thickness_um",
                rationale=(
                    "PML thickness is a solver assumption and must be explicit; it "
                    "will later require a boundary-reflection/convergence check."
                ),
                unit="um",
            ),
            SERSFDTDNumericalPolicyRequirement(
                field="padding_um",
                rationale=(
                    "Particle-to-PML/source/monitor clearance affects open-boundary "
                    "scattering calculations and must be explicit."
                ),
                unit="um",
            ),
            SERSFDTDNumericalPolicyRequirement(
                field="spectrum_frequency_points",
                rationale=(
                    "Scientific cases remain wavelength-specific, but resonance "
                    "calibration needs a denser broadband spectrum than the seven "
                    "landmark wavelengths alone."
                ),
                unit="frequency samples",
            ),
            SERSFDTDNumericalPolicyRequirement(
                field="spectral_guard_band_nm",
                rationale=(
                    "Keep the scientific wavelength bindings unchanged while allowing "
                    "the solver calibration spectrum to extend beyond their min/max. "
                    "This prevents a resonance maximum at the scientific-window edge "
                    "from being misread as a resolved peak."
                ),
                unit="nm",
            ),
            SERSFDTDNumericalPolicyRequirement(
                field="incident_propagation",
                rationale=(
                    "A nanorod polarization label is incomplete without an incident "
                    "wave direction. v0 supports propagation normal to the nanorod "
                    "long axis so both longitudinal and transverse E polarizations "
                    "remain physically transverse to propagation."
                ),
                allowed_values=["normal_to_nanorod_axis"],
            ),
            SERSFDTDNumericalPolicyRequirement(
                field="resource_benchmark",
                rationale=(
                    "Optional empirical local benchmark enables rough memory/time "
                    "scaling and RUN_LOCAL versus RESOURCE_DEFERRED planning."
                ),
                unit=None,
            ),
        ]

        return SERSFDTDNumericalPolicyRequest(
            request_id=_stable_id(
                "sers_fdtd_numerical_policy_request",
                jobs.bundle_id,
                self.policy_version,
            ),
            source_solver_job_bundle_id=jobs.bundle_id,
            solver_job_count=len(jobs.jobs),
            minimum_shell_thickness_nm=min_shell,
            maximum_outer_length_nm=max(row[0] for row in outer),
            maximum_outer_diameter_nm=max(row[1] for row in outer),
            wavelength_min_nm=min(wavelengths),
            wavelength_max_nm=max(wavelengths),
            default_target_cells_per_min_feature=_DEFAULT_TARGET_CELLS,
            target_resolution_for_minimum_shell_px_per_um=target_resolution,
            requirements=requirements,
            numerical_risk_notes=notes,
        )

    def apply(
        self,
        jobs: SERSFDTDSolverJobBundle,
        policy: SERSFDTDNumericalPolicyOverride,
    ) -> SERSFDTDNumericalPlanBundle:
        if policy.source_solver_job_bundle_id != jobs.bundle_id:
            raise ValueError(
                "numerical policy source_solver_job_bundle_id does not match input bundle"
            )
        if not jobs.jobs:
            raise ValueError("numerical policy requires at least one solver job")

        plans: list[SERSFDTDNumericalJobPlan] = []
        for job in jobs.jobs:
            outer_length_nm, outer_diameter_nm = _outer_dimensions_nm(job)
            min_feature_nm = job.shell_thickness_nm
            grid_nm = 1000.0 / policy.resolution_px_per_um
            feature_cells = min_feature_nm / grid_nm
            target_resolution = (
                1000.0 * policy.cells_per_min_feature_target / min_feature_nm
            )
            status = _resolution_status(
                feature_cells,
                policy.cells_per_min_feature_target,
            )

            # Coordinate convention reserved for the Meep backend:
            # rod long axis = x, plane-wave propagation = z, so longitudinal
            # polarization uses Ex and transverse polarization uses Ey.
            margin_um = policy.pml_thickness_um + policy.padding_um
            cell_x_um = outer_length_nm / 1000.0 + 2.0 * margin_um
            cell_y_um = outer_diameter_nm / 1000.0 + 2.0 * margin_um
            cell_z_um = outer_diameter_nm / 1000.0 + 2.0 * margin_um

            wavelengths = [
                row.wavelength_nm for row in job.wavelength_case_bindings
            ]
            scientific_min_nm = min(wavelengths)
            scientific_max_nm = max(wavelengths)
            calibration_min_nm = scientific_min_nm - policy.spectral_guard_band_nm
            calibration_max_nm = scientific_max_nm + policy.spectral_guard_band_nm
            if calibration_min_nm <= 0:
                raise ValueError(
                    "spectral_guard_band_nm extends the calibration window to a "
                    "non-positive wavelength"
                )
            resource = _resource_estimate(
                policy=policy,
                cell_x_um=cell_x_um,
                cell_y_um=cell_y_um,
                cell_z_um=cell_z_um,
            )

            plans.append(SERSFDTDNumericalJobPlan(
                numerical_plan_id=_stable_id(
                    "sers_fdtd_numerical_plan",
                    job.job_id,
                    policy,
                    self.policy_version,
                ),
                policy_id=policy.policy_id,
                source_solver_job_id=job.job_id,
                source_solver_job_bundle_id=jobs.bundle_id,
                hypothesis_id=job.hypothesis_id,
                geometry_candidate_id=job.geometry_candidate_id,
                intended_use=policy.intended_use,
                dimension=policy.dimension,
                incident_propagation=policy.incident_propagation,
                outer_length_nm=outer_length_nm,
                outer_diameter_nm=outer_diameter_nm,
                minimum_geometric_feature_nm=min_feature_nm,
                resolution_px_per_um=policy.resolution_px_per_um,
                grid_spacing_nm=grid_nm,
                minimum_feature_cells=feature_cells,
                cells_per_min_feature_target=policy.cells_per_min_feature_target,
                target_resolution_px_per_um=target_resolution,
                resolution_status=status,
                evidence_eligibility=_evidence_eligibility(
                    policy.intended_use,
                    status,
                ),
                pml_thickness_um=policy.pml_thickness_um,
                padding_um=policy.padding_um,
                cell_x_um=cell_x_um,
                cell_y_um=cell_y_um,
                cell_z_um=cell_z_um,
                wavelength_min_nm=scientific_min_nm,
                wavelength_max_nm=scientific_max_nm,
                spectrum_frequency_points=policy.spectrum_frequency_points,
                scientific_wavelength_min_nm=scientific_min_nm,
                scientific_wavelength_max_nm=scientific_max_nm,
                spectral_guard_band_nm=policy.spectral_guard_band_nm,
                calibration_wavelength_min_nm=calibration_min_nm,
                calibration_wavelength_max_nm=calibration_max_nm,
                resource_estimate=resource,
            ))

        plans.sort(key=lambda row: row.numerical_plan_id)
        return SERSFDTDNumericalPlanBundle(
            bundle_id=_stable_id(
                "sers_fdtd_numerical_plan_bundle",
                jobs.bundle_id,
                policy.policy_id,
                *(row.numerical_plan_id for row in plans),
                self.policy_version,
            ),
            source_solver_job_bundle_id=jobs.bundle_id,
            policy_id=policy.policy_id,
            plans=plans,
            solver_job_count=len(plans),
        )
