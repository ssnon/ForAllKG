from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from domains.sers.resonance_calibration_contracts import (
    SERSObservedResonancePeak,
    SERSPerPolarizationResonanceDistance,
    SERSPolarizationSpectrumObservation,
    SERSResonanceCalibrationBundle,
    SERSResonanceCalibrationObservation,
    SERSSourceResonanceDistanceObservation,
)


_CALIBRATION_VERSION = "sers-resonance-calibration-evaluator-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _polarization_from_field(field_component: str) -> str:
    if field_component == "Ex":
        return "parallel_to_nanorod_long_axis"
    if field_component == "Ey":
        return "perpendicular_to_nanorod_long_axis"
    raise ValueError(f"unsupported field component {field_component!r}")


def _validate_spectrum(raw: dict[str, Any]) -> list[tuple[float, float]]:
    rows = raw.get("spectrum")
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("worker result spectrum must contain at least three samples")
    parsed: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("spectrum rows must be objects")
        wavelength = row.get("wavelength_nm")
        cross_section = row.get("scattering_cross_section_um2")
        if not isinstance(wavelength, (int, float)) or wavelength <= 0:
            raise ValueError("spectrum wavelength must be positive numeric")
        if not isinstance(cross_section, (int, float)) or cross_section < 0:
            raise ValueError("scattering cross section must be non-negative numeric")
        parsed.append((float(wavelength), float(cross_section)))
    wavelengths = [row[0] for row in parsed]
    if wavelengths != sorted(wavelengths):
        raise ValueError("spectrum wavelengths must be ascending")
    if len(wavelengths) != len(set(wavelengths)):
        raise ValueError("spectrum wavelengths must be unique")
    return parsed


def _extract_peaks(
    spectrum: list[tuple[float, float]],
) -> tuple[SERSObservedResonancePeak, list[SERSObservedResonancePeak], str]:
    global_index = max(range(len(spectrum)), key=lambda i: spectrum[i][1])
    global_wavelength, global_value = spectrum[global_index]
    denominator = global_value if global_value > 0 else 1.0
    if global_index == 0:
        peak_location_status = "lower_boundary_censored"
    elif global_index == len(spectrum) - 1:
        peak_location_status = "upper_boundary_censored"
    else:
        peak_location_status = "interior_peak"

    global_peak = SERSObservedResonancePeak(
        wavelength_nm=global_wavelength,
        scattering_cross_section_um2=global_value,
        relative_to_global_peak=1.0 if global_value > 0 else 0.0,
        peak_kind="sampled_local_maximum"
        if 0 < global_index < len(spectrum) - 1
        else "global_peak_fallback",
    )

    local: list[SERSObservedResonancePeak] = []
    for i in range(1, len(spectrum) - 1):
        y0 = spectrum[i - 1][1]
        y1 = spectrum[i][1]
        y2 = spectrum[i + 1][1]
        if y1 >= y0 and y1 >= y2 and (y1 > y0 or y1 > y2):
            local.append(SERSObservedResonancePeak(
                wavelength_nm=spectrum[i][0],
                scattering_cross_section_um2=y1,
                relative_to_global_peak=(y1 / denominator) if global_value > 0 else 0.0,
                peak_kind="sampled_local_maximum",
            ))
    local.sort(key=lambda row: row.wavelength_nm)
    return global_peak, local, peak_location_status


@dataclass(frozen=True)
class SERSCompletedMeepRun:
    request: dict[str, Any]
    result: dict[str, Any]


class SERSResonanceCalibrationEvaluator:
    evaluator_version = _CALIBRATION_VERSION

    def evaluate(self, runs: list[SERSCompletedMeepRun]) -> SERSResonanceCalibrationBundle:
        if not runs:
            raise ValueError("at least one completed Meep run is required")

        groups: dict[tuple[str, str, str], list[SERSCompletedMeepRun]] = {}
        for run in runs:
            request = run.request
            result = run.result
            if result.get("worker_status") != "completed":
                raise ValueError("resonance calibration accepts completed worker results only")
            if request.get("request_id") != result.get("request_id"):
                raise ValueError("request/result request_id mismatch")
            key = (
                str(request.get("hypothesis_id")),
                str(request.get("geometry_candidate_id")),
                str(request.get("surrounding_medium")),
            )
            groups.setdefault(key, []).append(run)

        observations: list[SERSResonanceCalibrationObservation] = []
        for key, grouped in sorted(groups.items()):
            hypothesis_id, geometry_candidate_id, surrounding_medium = key
            if not all(key):
                raise ValueError("calibration grouping keys must be non-empty")

            by_field: dict[str, SERSCompletedMeepRun] = {}
            for run in grouped:
                field = str(run.result.get("field_component"))
                if field in by_field:
                    raise ValueError(
                        f"duplicate completed {field} run for calibration group {key}"
                    )
                by_field[field] = run

            reference = grouped[0].request
            reported = [float(x) for x in reference.get("reported_resonance_wavelengths_nm", [])]
            baseline = reference.get("baseline_wavelength_nm")
            for run in grouped[1:]:
                other_reported = [
                    float(x) for x in run.request.get("reported_resonance_wavelengths_nm", [])
                ]
                if other_reported != reported:
                    raise ValueError("reported source resonance landmarks disagree across paired runs")
                if run.request.get("baseline_wavelength_nm") != baseline:
                    raise ValueError("baseline wavelength disagrees across paired runs")

            spectra: list[SERSPolarizationSpectrumObservation] = []
            peak_pool: list[tuple[str, str, SERSObservedResonancePeak]] = []
            resolution_statuses: set[str] = set()
            eligibility: set[str] = set()
            for field in sorted(by_field):
                run = by_field[field]
                spectrum = _validate_spectrum(run.result)
                global_peak, local_peaks, peak_location_status = _extract_peaks(spectrum)
                candidates = local_peaks or [SERSObservedResonancePeak(
                    wavelength_nm=global_peak.wavelength_nm,
                    scattering_cross_section_um2=global_peak.scattering_cross_section_um2,
                    relative_to_global_peak=global_peak.relative_to_global_peak,
                    peak_kind="global_peak_fallback",
                )]
                polarization = str(run.request.get("polarization"))
                expected_polarization = _polarization_from_field(field)
                if polarization != expected_polarization:
                    raise ValueError("request polarization/field-component mapping is inconsistent")
                resolution_status = str(run.result.get("numerical_resolution_status"))
                evidence_eligibility = str(run.result.get("evidence_eligibility"))
                resolution_statuses.add(resolution_status)
                eligibility.add(evidence_eligibility)
                spectra.append(SERSPolarizationSpectrumObservation(
                    request_id=str(run.request["request_id"]),
                    source_solver_job_id=str(run.request["source_solver_job_id"]),
                    polarization=polarization,
                    field_component=field,
                    numerical_resolution_status=resolution_status,
                    evidence_eligibility=evidence_eligibility,
                    resolution_px_per_um=float(run.result["resolution_px_per_um"]),
                    grid_spacing_nm=float(run.result["grid_spacing_nm"]),
                    wall_seconds=(
                        float(run.result["wall_seconds"])
                        if run.result.get("wall_seconds") is not None
                        else None
                    ),
                    peak_rss_mb=(
                        float(run.result["peak_rss_mb"])
                        if run.result.get("peak_rss_mb") is not None
                        else None
                    ),
                    spectrum_wavelength_min_nm=spectrum[0][0],
                    spectrum_wavelength_max_nm=spectrum[-1][0],
                    peak_location_status=peak_location_status,
                    global_peak=global_peak,
                    local_peaks=local_peaks,
                ))
                for peak in candidates:
                    peak_pool.append((polarization, field, peak))

            paired = "Ex" in by_field and "Ey" in by_field
            distance_rows: list[SERSSourceResonanceDistanceObservation] = []
            for source_wavelength in reported:
                per_pol: list[SERSPerPolarizationResonanceDistance] = []
                for polarization in (
                    "parallel_to_nanorod_long_axis",
                    "perpendicular_to_nanorod_long_axis",
                ):
                    candidates = [row for row in peak_pool if row[0] == polarization]
                    if not candidates:
                        continue
                    p, field, peak = min(
                        candidates,
                        key=lambda row: abs(row[2].wavelength_nm - source_wavelength),
                    )
                    distance = abs(peak.wavelength_nm - source_wavelength)
                    per_pol.append(SERSPerPolarizationResonanceDistance(
                        polarization=p,
                        field_component=field,
                        source_resonance_wavelength_nm=source_wavelength,
                        observed_peak_wavelength_nm=peak.wavelength_nm,
                        absolute_distance_nm=distance,
                        relative_distance_fraction=distance / source_wavelength,
                        observed_peak_relative_to_global=peak.relative_to_global_peak,
                        observed_peak_kind=peak.peak_kind,
                    ))
                if not per_pol:
                    continue
                nearest = min(per_pol, key=lambda row: row.absolute_distance_nm)
                distance_rows.append(SERSSourceResonanceDistanceObservation(
                    source_resonance_wavelength_nm=source_wavelength,
                    nearest_observed_polarization=nearest.polarization,
                    nearest_observed_field_component=nearest.field_component,
                    nearest_observed_peak_wavelength_nm=nearest.observed_peak_wavelength_nm,
                    absolute_distance_nm=nearest.absolute_distance_nm,
                    relative_distance_fraction=nearest.relative_distance_fraction,
                    nearest_peak_relative_to_global=nearest.observed_peak_relative_to_global,
                    nearest_peak_kind=nearest.observed_peak_kind,
                    per_polarization=per_pol,
                ))

            if not paired:
                quality_gate = "incomplete_polarization_pair"
            elif "underresolved" in resolution_statuses:
                quality_gate = "observation_only_underresolved"
            elif "coarse" in resolution_statuses:
                quality_gate = "screening_only_coarse"
            elif eligibility <= {"calibration_candidate", "quantitative_candidate"}:
                quality_gate = "calibration_candidate_review"
            else:
                quality_gate = "screening_only_coarse"

            observation_id = _stable_id(
                "sers_resonance_calibration",
                hypothesis_id,
                geometry_candidate_id,
                surrounding_medium,
                *(row.request_id for row in spectra),
                self.evaluator_version,
            )
            observations.append(SERSResonanceCalibrationObservation(
                observation_id=observation_id,
                hypothesis_id=hypothesis_id,
                geometry_candidate_id=geometry_candidate_id,
                surrounding_medium=surrounding_medium,
                baseline_wavelength_nm=(float(baseline) if baseline is not None else None),
                reported_resonance_wavelengths_nm=reported,
                spectra=spectra,
                paired_polarization_complete=paired,
                source_resonance_distances=distance_rows,
                quality_gate=quality_gate,
            ))

        bundle_id = _stable_id(
            "sers_resonance_calibration_bundle",
            *(row.observation_id for row in observations),
            self.evaluator_version,
        )
        return SERSResonanceCalibrationBundle(
            bundle_id=bundle_id,
            observations=observations,
            observation_count=len(observations),
            execution_result_count=len(runs),
        )
