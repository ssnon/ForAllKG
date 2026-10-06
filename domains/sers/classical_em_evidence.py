from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from domains.sers.classical_em_evidence_contracts import (
    SERSClassicalEMPhysicsEvidence,
    SERSClassicalEMPhysicsEvidenceBundle,
    SERSEMConvergenceSummary,
    SERSEMSpectrumSnapshot,
    SERSSourceResonanceSnapshot,
)
from domains.sers.classical_em_handoff_contracts import (
    SERSClassicalEMHandoff,
    SERSClassicalEMHandoffBundle,
)
from domains.sers.resonance_calibration_contracts import (
    SERSResonanceCalibrationBundle,
    SERSResonanceCalibrationObservation,
)
from domains.sers.resolution_convergence_contracts import (
    SERSResolutionConvergenceBundle,
    SERSResolutionConvergenceObservation,
)
from domains.sers.solver_execution_contracts import SERSFDTDSolverJobBundle


_ASSEMBLER_VERSION = "sers-classical-em-evidence-assembler-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _snapshot_signature(row) -> tuple[object, ...]:
    return (
        row.field_component,
        row.polarization,
        float(row.resolution_px_per_um),
        row.numerical_resolution_status,
        row.evidence_eligibility,
        float(row.global_peak.wavelength_nm),
        float(row.global_peak.scattering_cross_section_um2),
        row.peak_location_status,
        float(row.spectrum_wavelength_min_nm),
        float(row.spectrum_wavelength_max_nm),
    )


def _point_signature(series, point) -> tuple[object, ...]:
    return (
        series.field_component,
        series.polarization,
        float(point.resolution_px_per_um),
        point.numerical_resolution_status,
        point.evidence_eligibility,
        float(point.peak_wavelength_nm),
        float(point.peak_scattering_cross_section_um2),
        point.peak_location_status,
        float(point.spectrum_wavelength_min_nm),
        float(point.spectrum_wavelength_max_nm),
    )


def _convergence_summary(
    row: SERSResolutionConvergenceObservation,
) -> SERSEMConvergenceSummary:
    shifts = [
        step.absolute_peak_shift_nm
        for series in row.series
        for step in series.drift_steps
        if step.absolute_peak_shift_nm is not None
    ]
    return SERSEMConvergenceSummary(
        source_convergence_observation_id=row.observation_id,
        overall_status=row.overall_status,
        resolution_count=row.resolution_count,
        series_status_by_field={
            series.field_component: series.series_status
            for series in row.series
        },
        boundary_limited_fields=sorted(
            series.field_component
            for series in row.series
            if series.series_status == "boundary_limited"
        ),
        maximum_reported_interior_peak_shift_nm=max(shifts) if shifts else None,
    )


def _state_and_actions(
    calibrations: list[SERSResonanceCalibrationObservation],
    convergence: SERSResolutionConvergenceObservation | None,
) -> tuple[str, list[str], list[str]]:
    spectra = [spectrum for row in calibrations for spectrum in row.spectra]
    quality_gates = {row.quality_gate for row in calibrations}
    resolution_statuses = {row.numerical_resolution_status for row in spectra}
    censored = any(row.peak_location_status != "interior_peak" for row in spectra)

    blockers: list[str] = []
    actions: list[str] = []

    if "incomplete_polarization_pair" in quality_gates:
        blockers.append("incomplete_polarization_pair")
        actions.append("complete_polarization_pair")
    if censored or (
        convergence is not None and convergence.overall_status == "boundary_limited"
    ):
        blockers.append("spectral_boundary_censoring")
        actions.append("resolve_spectral_boundary")
    if "underresolved" in resolution_statuses:
        blockers.append("underresolved_geometry")
        actions.append("improve_spatial_resolution_or_solver_fidelity")
    elif "coarse" in resolution_statuses:
        blockers.append("coarse_only")
        actions.append("improve_spatial_resolution_or_solver_fidelity")

    if convergence is None:
        blockers.append("missing_resolution_series")
        actions.append("add_resolution_point")
    elif convergence.overall_status == "insufficient_resolution_series":
        blockers.append("missing_resolution_series")
        actions.append("add_resolution_point")
    elif convergence.overall_status == "drift_report_only_no_threshold":
        blockers.append("no_convergence_threshold")
        actions.append("establish_convergence_acceptance_policy")

    if "incomplete_polarization_pair" in quality_gates:
        state = "incomplete_observation"
    elif "spectral_boundary_censoring" in blockers:
        state = "boundary_limited_observation"
    elif "underresolved_geometry" in blockers:
        state = "underresolved_observation"
    elif "missing_resolution_series" in blockers:
        state = "insufficient_resolution_series"
    elif "coarse_only" in blockers:
        state = "coarse_screening_observation"
    elif "no_convergence_threshold" in blockers:
        state = "drift_report_only_no_threshold"
    else:
        state = "calibration_candidate_review"
        actions.append("review_calibration_candidate")

    return state, list(dict.fromkeys(blockers)), list(dict.fromkeys(actions))


class SERSClassicalEMEvidenceAssembler:
    """Normalize routed FDTD observations into classical-EM evidence.

    v0 is intentionally observational. It verifies lineage from the routed
    classical-EM handoff through solver jobs into calibration/convergence
    artifacts, then reports numerical limitations without issuing a support,
    contradiction, experiment-promotion, or self-feedback verdict.
    """

    assembler_version = _ASSEMBLER_VERSION

    def assemble(
        self,
        handoffs: SERSClassicalEMHandoffBundle,
        solver_jobs: SERSFDTDSolverJobBundle,
        calibration_bundles: list[SERSResonanceCalibrationBundle],
        convergence_bundles: list[SERSResolutionConvergenceBundle] | None = None,
    ) -> SERSClassicalEMPhysicsEvidenceBundle:
        if not calibration_bundles:
            raise ValueError("at least one resonance-calibration bundle is required")
        convergence_bundles = convergence_bundles or []

        handoff_by_hypothesis = {row.hypothesis_id: row for row in handoffs.handoffs}
        if len(handoff_by_hypothesis) != len(handoffs.handoffs):
            raise ValueError("duplicate classical-EM handoff hypothesis")

        jobs_by_id = {row.job_id: row for row in solver_jobs.jobs}
        if len(jobs_by_id) != len(solver_jobs.jobs):
            raise ValueError("duplicate solver job_id")

        calibration_bundle_ids = [row.bundle_id for row in calibration_bundles]
        if len(calibration_bundle_ids) != len(set(calibration_bundle_ids)):
            raise ValueError("duplicate calibration bundle_id")
        convergence_bundle_ids = [row.bundle_id for row in convergence_bundles]
        if len(convergence_bundle_ids) != len(set(convergence_bundle_ids)):
            raise ValueError("duplicate convergence bundle_id")

        calibration_groups: dict[
            tuple[str, str, str], list[SERSResonanceCalibrationObservation]
        ] = defaultdict(list)
        for bundle in calibration_bundles:
            for observation in bundle.observations:
                calibration_groups[
                    (
                        observation.hypothesis_id,
                        observation.geometry_candidate_id,
                        observation.surrounding_medium,
                    )
                ].append(observation)

        convergence_by_key: dict[
            tuple[str, str, str], SERSResolutionConvergenceObservation
        ] = {}
        for bundle in convergence_bundles:
            for observation in bundle.observations:
                key = (
                    observation.hypothesis_id,
                    observation.geometry_candidate_id,
                    observation.surrounding_medium,
                )
                if key in convergence_by_key:
                    raise ValueError(f"duplicate convergence observation for {key}")
                convergence_by_key[key] = observation

        evidence_rows: list[SERSClassicalEMPhysicsEvidence] = []
        observed_handoffs: set[str] = set()

        for key, calibration_rows in sorted(calibration_groups.items()):
            hypothesis_id, geometry_candidate_id, surrounding_medium = key
            handoff = handoff_by_hypothesis.get(hypothesis_id)
            if handoff is None:
                raise ValueError(
                    "calibration observation has no routed classical-EM handoff: "
                    + hypothesis_id
                )
            observed_handoffs.add(handoff.handoff_id)

            spectrum_rows = [
                spectrum
                for calibration in calibration_rows
                for spectrum in calibration.spectra
            ]
            if not spectrum_rows:
                raise ValueError(f"calibration group has no spectra: {key}")

            source_solver_job_ids = sorted({
                spectrum.source_solver_job_id for spectrum in spectrum_rows
            })
            for solver_job_id in source_solver_job_ids:
                job = jobs_by_id.get(solver_job_id)
                if job is None:
                    raise ValueError(
                        f"calibration references solver job absent from supplied bundle: {solver_job_id}"
                    )
                if job.hypothesis_id != hypothesis_id:
                    raise ValueError("solver job/calibration hypothesis mismatch")
                if job.source_spec_id != handoff.source_simulation_spec_id:
                    raise ValueError(
                        "solver job does not descend from routed simulation spec: "
                        f"job={solver_job_id}, expected_spec={handoff.source_simulation_spec_id}, "
                        f"actual_spec={job.source_spec_id}"
                    )
                if job.geometry_candidate_id != geometry_candidate_id:
                    raise ValueError("solver job/calibration geometry mismatch")
                if job.surrounding_medium != surrounding_medium:
                    raise ValueError("solver job/calibration surrounding-medium mismatch")

            # Calibration landmarks and baseline must agree across resolutions.
            reference = calibration_rows[0]
            for row in calibration_rows[1:]:
                if row.baseline_wavelength_nm != reference.baseline_wavelength_nm:
                    raise ValueError("calibration baseline differs across evidence group")
                if (
                    row.reported_resonance_wavelengths_nm
                    != reference.reported_resonance_wavelengths_nm
                ):
                    raise ValueError(
                        "reported resonance landmarks differ across evidence group"
                    )

            convergence = convergence_by_key.get(key)
            if convergence is not None:
                available_signatures = {
                    _snapshot_signature(spectrum)
                    for spectrum in spectrum_rows
                }
                for series in convergence.series:
                    for point in series.points:
                        signature = _point_signature(series, point)
                        if signature not in available_signatures:
                            raise ValueError(
                                "convergence point is not reproducible from supplied "
                                f"calibration observations for {key}: {signature}"
                            )
                if convergence.baseline_wavelength_nm != reference.baseline_wavelength_nm:
                    raise ValueError("convergence/calibration baseline mismatch")
                if (
                    convergence.reported_resonance_wavelengths_nm
                    != reference.reported_resonance_wavelengths_nm
                ):
                    raise ValueError("convergence/calibration resonance-landmark mismatch")

            spectrum_snapshots = sorted(
                (
                    SERSEMSpectrumSnapshot(
                        source_calibration_observation_id=calibration.observation_id,
                        request_id=spectrum.request_id,
                        source_solver_job_id=spectrum.source_solver_job_id,
                        polarization=spectrum.polarization,
                        field_component=spectrum.field_component,
                        resolution_px_per_um=spectrum.resolution_px_per_um,
                        numerical_resolution_status=spectrum.numerical_resolution_status,
                        evidence_eligibility=spectrum.evidence_eligibility,
                        peak_wavelength_nm=spectrum.global_peak.wavelength_nm,
                        peak_scattering_cross_section_um2=(
                            spectrum.global_peak.scattering_cross_section_um2
                        ),
                        peak_location_status=spectrum.peak_location_status,
                        spectrum_wavelength_min_nm=spectrum.spectrum_wavelength_min_nm,
                        spectrum_wavelength_max_nm=spectrum.spectrum_wavelength_max_nm,
                        parent_quality_gate=calibration.quality_gate,
                    )
                    for calibration in calibration_rows
                    for spectrum in calibration.spectra
                ),
                key=lambda row: (
                    row.resolution_px_per_um,
                    row.field_component,
                    row.request_id,
                ),
            )

            resonance_snapshots = sorted(
                (
                    SERSSourceResonanceSnapshot(
                        source_calibration_observation_id=calibration.observation_id,
                        source_resonance_wavelength_nm=distance.source_resonance_wavelength_nm,
                        nearest_observed_field_component=(
                            distance.nearest_observed_field_component
                        ),
                        nearest_observed_peak_wavelength_nm=(
                            distance.nearest_observed_peak_wavelength_nm
                        ),
                        absolute_distance_nm=distance.absolute_distance_nm,
                        relative_distance_fraction=distance.relative_distance_fraction,
                    )
                    for calibration in calibration_rows
                    for distance in calibration.source_resonance_distances
                ),
                key=lambda row: (
                    row.source_calibration_observation_id,
                    row.source_resonance_wavelength_nm,
                ),
            )

            state, blockers, actions = _state_and_actions(
                calibration_rows,
                convergence,
            )
            convergence_summary = (
                _convergence_summary(convergence)
                if convergence is not None
                else None
            )
            calibration_ids = sorted(row.observation_id for row in calibration_rows)

            evidence_rows.append(SERSClassicalEMPhysicsEvidence(
                evidence_id=_stable_id(
                    "sers_classical_em_evidence",
                    handoff.handoff_id,
                    geometry_candidate_id,
                    surrounding_medium,
                    tuple(calibration_ids),
                    convergence.observation_id if convergence is not None else None,
                    self.assembler_version,
                ),
                hypothesis_id=hypothesis_id,
                source_classical_em_handoff_id=handoff.handoff_id,
                source_validation_plan_id=handoff.source_validation_plan_id,
                source_validation_route_id=handoff.source_validation_route_id,
                source_fdtd_subclaim_ids=handoff.source_fdtd_subclaim_ids,
                source_simulation_spec_id=handoff.source_simulation_spec_id,
                source_solver_job_ids=source_solver_job_ids,
                source_calibration_observation_ids=calibration_ids,
                source_convergence_observation_id=(
                    convergence.observation_id if convergence is not None else None
                ),
                geometry_candidate_id=geometry_candidate_id,
                surrounding_medium=surrounding_medium,
                required_for_outcome_assessment=handoff.required_for_outcome_assessment,
                required_for_mechanism_assessment=(
                    handoff.required_for_mechanism_assessment
                ),
                spectrum_snapshots=spectrum_snapshots,
                source_resonance_snapshots=resonance_snapshots,
                convergence_summary=convergence_summary,
                numerical_evidence_state=state,
                numerical_blockers=blockers,
                next_required_actions=actions,
            ))

        unobserved = sorted(
            row.handoff_id
            for row in handoffs.handoffs
            if row.handoff_id not in observed_handoffs
        )

        return SERSClassicalEMPhysicsEvidenceBundle(
            bundle_id=_stable_id(
                "sers_classical_em_evidence_bundle",
                handoffs.bundle_id,
                solver_jobs.bundle_id,
                tuple(sorted(calibration_bundle_ids)),
                tuple(sorted(convergence_bundle_ids)),
                *(row.evidence_id for row in evidence_rows),
                self.assembler_version,
            ),
            source_handoff_bundle_id=handoffs.bundle_id,
            source_solver_job_bundle_id=solver_jobs.bundle_id,
            source_calibration_bundle_ids=sorted(calibration_bundle_ids),
            source_convergence_bundle_ids=sorted(convergence_bundle_ids),
            evidence=evidence_rows,
            evidence_count=len(evidence_rows),
            unobserved_handoff_ids=unobserved,
        )
