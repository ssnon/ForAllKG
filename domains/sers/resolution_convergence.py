from __future__ import annotations

import hashlib
import json

from domains.sers.resonance_calibration_contracts import (
    SERSResonanceCalibrationObservation,
)
from domains.sers.resolution_convergence_contracts import (
    SERSPolarizationResolutionSeries,
    SERSResolutionConvergenceBundle,
    SERSResolutionConvergenceObservation,
    SERSResolutionConvergencePoint,
    SERSResolutionDriftStep,
)


_CONVERGENCE_VERSION = "sers-resolution-convergence-evaluator-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _relative_change(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new - old) / old


class SERSResolutionConvergenceEvaluator:
    evaluator_version = _CONVERGENCE_VERSION

    def evaluate(
        self,
        calibrations: list[SERSResonanceCalibrationObservation],
    ) -> SERSResolutionConvergenceBundle:
        if not calibrations:
            raise ValueError("at least one resonance-calibration observation is required")

        groups: dict[tuple[str, str, str], list[SERSResonanceCalibrationObservation]] = {}
        for row in calibrations:
            key = (row.hypothesis_id, row.geometry_candidate_id, row.surrounding_medium)
            groups.setdefault(key, []).append(row)

        observations: list[SERSResolutionConvergenceObservation] = []
        for key, rows in sorted(groups.items()):
            hypothesis_id, geometry_candidate_id, surrounding_medium = key
            reference = rows[0]
            for row in rows[1:]:
                if row.reported_resonance_wavelengths_nm != reference.reported_resonance_wavelengths_nm:
                    raise ValueError("reported resonance landmarks disagree across resolution series")
                if row.baseline_wavelength_nm != reference.baseline_wavelength_nm:
                    raise ValueError("baseline wavelength disagrees across resolution series")

            by_field: dict[str, list] = {}
            for row in rows:
                for spectrum in row.spectra:
                    by_field.setdefault(spectrum.field_component, []).append(spectrum)

            series_rows: list[SERSPolarizationResolutionSeries] = []
            resolution_values: set[float] = set()
            for field, spectra in sorted(by_field.items()):
                spectra.sort(key=lambda item: item.resolution_px_per_um)
                resolutions = [item.resolution_px_per_um for item in spectra]
                if len(resolutions) != len(set(resolutions)):
                    raise ValueError(
                        f"duplicate resolution for field {field} in convergence group {key}"
                    )
                resolution_values.update(resolutions)

                points = [
                    SERSResolutionConvergencePoint(
                        resolution_px_per_um=item.resolution_px_per_um,
                        grid_spacing_nm=item.grid_spacing_nm,
                        numerical_resolution_status=item.numerical_resolution_status,
                        evidence_eligibility=item.evidence_eligibility,
                        peak_wavelength_nm=item.global_peak.wavelength_nm,
                        peak_scattering_cross_section_um2=(
                            item.global_peak.scattering_cross_section_um2
                        ),
                        peak_location_status=item.peak_location_status,
                        spectrum_wavelength_min_nm=item.spectrum_wavelength_min_nm,
                        spectrum_wavelength_max_nm=item.spectrum_wavelength_max_nm,
                        wall_seconds=item.wall_seconds,
                        peak_rss_mb=item.peak_rss_mb,
                    )
                    for item in spectra
                ]

                drift_steps: list[SERSResolutionDriftStep] = []
                for before, after in zip(points, points[1:]):
                    censored = (
                        before.peak_location_status != "interior_peak"
                        or after.peak_location_status != "interior_peak"
                    )
                    if censored:
                        signed_shift = None
                        absolute_shift = None
                        relative_shift = None
                        peak_shift_status = "not_estimable_boundary_censored"
                    else:
                        signed_shift = after.peak_wavelength_nm - before.peak_wavelength_nm
                        absolute_shift = abs(signed_shift)
                        relative_shift = absolute_shift / before.peak_wavelength_nm
                        peak_shift_status = "estimated_interior_to_interior"

                    drift_steps.append(SERSResolutionDriftStep(
                        from_resolution_px_per_um=before.resolution_px_per_um,
                        to_resolution_px_per_um=after.resolution_px_per_um,
                        from_peak_location_status=before.peak_location_status,
                        to_peak_location_status=after.peak_location_status,
                        peak_shift_status=peak_shift_status,
                        signed_peak_shift_nm=signed_shift,
                        absolute_peak_shift_nm=absolute_shift,
                        relative_peak_shift_fraction=relative_shift,
                        peak_cross_section_relative_change=_relative_change(
                            after.peak_scattering_cross_section_um2,
                            before.peak_scattering_cross_section_um2,
                        ),
                        wall_time_ratio=_ratio(after.wall_seconds, before.wall_seconds),
                        peak_rss_ratio=_ratio(after.peak_rss_mb, before.peak_rss_mb),
                    ))

                if len(points) < 2:
                    series_status = "insufficient_resolution_series"
                elif any(point.peak_location_status != "interior_peak" for point in points):
                    series_status = "boundary_limited"
                else:
                    series_status = "interior_drift_report_only"

                series_rows.append(SERSPolarizationResolutionSeries(
                    polarization=spectra[0].polarization,
                    field_component=field,
                    points=points,
                    drift_steps=drift_steps,
                    series_status=series_status,
                ))

            statuses = {row.series_status for row in series_rows}
            if "boundary_limited" in statuses:
                overall_status = "boundary_limited"
            elif "insufficient_resolution_series" in statuses:
                overall_status = "insufficient_resolution_series"
            else:
                overall_status = "drift_report_only_no_threshold"

            observation_id = _stable_id(
                "sers_resolution_convergence",
                hypothesis_id,
                geometry_candidate_id,
                surrounding_medium,
                *(row.observation_id for row in sorted(rows, key=lambda item: item.observation_id)),
                self.evaluator_version,
            )
            observations.append(SERSResolutionConvergenceObservation(
                observation_id=observation_id,
                hypothesis_id=hypothesis_id,
                geometry_candidate_id=geometry_candidate_id,
                surrounding_medium=surrounding_medium,
                baseline_wavelength_nm=reference.baseline_wavelength_nm,
                reported_resonance_wavelengths_nm=reference.reported_resonance_wavelengths_nm,
                series=series_rows,
                resolution_count=len(resolution_values),
                overall_status=overall_status,
            ))

        bundle_id = _stable_id(
            "sers_resolution_convergence_bundle",
            *(row.observation_id for row in observations),
            self.evaluator_version,
        )
        return SERSResolutionConvergenceBundle(
            bundle_id=bundle_id,
            observations=observations,
            observation_count=len(observations),
            source_calibration_observation_count=len(calibrations),
        )
