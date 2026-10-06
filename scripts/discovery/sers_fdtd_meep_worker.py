#!/usr/bin/env python3
"""Standalone Meep worker for one shadow SERS FDTD execution request.

This file intentionally imports no ForAllKG modules so it can be launched inside
an isolated ``fdtd-meep`` conda environment. The worker performs two time-domain
runs: a background normalization and a structure run with incident-field flux
subtraction. Its output is raw numerical data only and creates no scientific
or hypothesis authority.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request JSON must be an object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _required_number(request: dict[str, Any], name: str) -> float:
    value = request.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("schema_version") != "sers-meep-execution-request-v0":
        raise ValueError("unsupported request schema_version")
    if request.get("geometry_family") != "core_shell_nanorod":
        raise ValueError("v0 worker supports core_shell_nanorod only")
    if request.get("coordinate_convention") != "nanorod_x_incident_z":
        raise ValueError("v0 worker requires nanorod_x_incident_z coordinates")
    if request.get("field_component") not in {"Ex", "Ey"}:
        raise ValueError("field_component must be Ex or Ey")
    if request.get("core_material") not in {"Au", "Ag"}:
        raise ValueError("unsupported core material")
    if request.get("shell_material") not in {"Au", "Ag"}:
        raise ValueError("unsupported shell material")

    length_nm = _required_number(request, "nanorod_length_nm")
    diameter_nm = _required_number(request, "nanorod_diameter_nm")
    shell_nm = _required_number(request, "shell_thickness_nm")
    if length_nm <= diameter_nm:
        raise ValueError("nanorod core length must exceed diameter")

    pml = _required_number(request, "pml_thickness_um")
    padding = _required_number(request, "padding_um")
    cell_x = _required_number(request, "cell_x_um")
    cell_y = _required_number(request, "cell_y_um")
    cell_z = _required_number(request, "cell_z_um")
    resolution = _required_number(request, "resolution_px_per_um")
    background_index = _required_number(request, "background_index")
    courant = _required_number(request, "courant")
    if courant > 0.5:
        raise ValueError("courant must be <= 0.5")

    wavelength_min = _required_number(request, "wavelength_min_nm")
    wavelength_max = _required_number(request, "wavelength_max_nm")
    if wavelength_min >= wavelength_max:
        raise ValueError("wavelength_min_nm must be below wavelength_max_nm")
    nfreq = request.get("spectrum_frequency_points")
    if not isinstance(nfreq, int) or isinstance(nfreq, bool) or nfreq < 3:
        raise ValueError("spectrum_frequency_points must be integer >= 3")

    outer_length_um = (length_nm + 2.0 * shell_nm) / 1000.0
    outer_diameter_um = (diameter_nm + 2.0 * shell_nm) / 1000.0
    expected_x = outer_length_um + 2.0 * (pml + padding)
    expected_yz = outer_diameter_um + 2.0 * (pml + padding)
    tolerance = max(1.0 / resolution, 1e-9)
    if abs(cell_x - expected_x) > tolerance:
        raise ValueError("cell_x_um is inconsistent with geometry/PML/padding")
    if abs(cell_y - expected_yz) > tolerance or abs(cell_z - expected_yz) > tolerance:
        raise ValueError("cell_y/z_um is inconsistent with geometry/PML/padding")

    bindings = request.get("wavelength_case_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("wavelength_case_bindings must be a non-empty list")
    binding_wavelengths = []
    for row in bindings:
        if not isinstance(row, dict) or not row.get("source_case_id"):
            raise ValueError("invalid wavelength binding")
        wavelength = row.get("wavelength_nm")
        if not isinstance(wavelength, (int, float)) or wavelength <= 0:
            raise ValueError("binding wavelength must be > 0")
        binding_wavelengths.append(float(wavelength))
    if binding_wavelengths != sorted(binding_wavelengths):
        raise ValueError("binding wavelengths must be sorted")

    return {
        "outer_length_um": outer_length_um,
        "outer_diameter_um": outer_diameter_um,
        "cell_um": [cell_x, cell_y, cell_z],
        "grid_spacing_nm": 1000.0 / resolution,
        "background_index": background_index,
        "frequency_min_inv_um": 1.0 / (wavelength_max / 1000.0),
        "frequency_max_inv_um": 1.0 / (wavelength_min / 1000.0),
        "binding_wavelengths_nm": binding_wavelengths,
    }


def _rss_mb() -> float:
    # Linux ru_maxrss is KiB; macOS reports bytes. The target backend is Linux,
    # but keep a defensive conversion for standalone use elsewhere.
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _spherocylinder(mp, *, length_um: float, diameter_um: float, material) -> list[Any]:
    radius = 0.5 * diameter_um
    barrel = length_um - diameter_um
    if barrel <= 0:
        raise ValueError("spherocylinder length must exceed diameter")
    half_barrel = 0.5 * barrel
    return [
        mp.Cylinder(
            radius=radius,
            height=barrel,
            axis=mp.Vector3(1, 0, 0),
            center=mp.Vector3(),
            material=material,
        ),
        mp.Sphere(
            radius=radius,
            center=mp.Vector3(-half_barrel, 0, 0),
            material=material,
        ),
        mp.Sphere(
            radius=radius,
            center=mp.Vector3(+half_barrel, 0, 0),
            material=material,
        ),
    ]


def _material_by_name(materials, name: str):
    getter = getattr(materials, "get_material", None)
    if getter is not None:
        return getter(name)
    return getattr(materials, name)


def _valid_frequency_range(material) -> tuple[float | None, float | None]:
    value = getattr(material, "valid_freq_range", None)
    if value is None:
        return None, None
    return getattr(value, "min", None), getattr(value, "max", None)


def _execute(request: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import meep as mp
    from meep import materials

    start = time.monotonic()

    resolution = float(request["resolution_px_per_um"])
    pml = float(request["pml_thickness_um"])
    padding = float(request["padding_um"])
    cell_x = float(request["cell_x_um"])
    cell_y = float(request["cell_y_um"])
    cell_z = float(request["cell_z_um"])
    background_index = float(request["background_index"])
    courant = float(request["courant"])
    until_after = float(request["run_until_after_sources"])
    nfreq = int(request["spectrum_frequency_points"])

    wavelength_min_um = float(request["wavelength_min_nm"]) / 1000.0
    wavelength_max_um = float(request["wavelength_max_nm"]) / 1000.0
    freq_min = 1.0 / wavelength_max_um
    freq_max = 1.0 / wavelength_min_um
    fcen = 0.5 * (freq_min + freq_max)
    df = freq_max - freq_min

    core = _material_by_name(materials, request["core_material"])
    shell = _material_by_name(materials, request["shell_material"])
    material_ranges = {}
    for name, material in [(request["core_material"], core), (request["shell_material"], shell)]:
        lo, hi = _valid_frequency_range(material)
        material_ranges[name] = {"min": lo, "max": hi}
        if lo is not None and freq_min < lo - 1e-12:
            raise ValueError(f"{name} material model invalid below requested frequency range")
        if hi is not None and freq_max > hi + 1e-12:
            raise ValueError(f"{name} material model invalid above requested frequency range")

    background = mp.Medium(index=background_index)
    component = mp.Ex if request["field_component"] == "Ex" else mp.Ey
    cell = mp.Vector3(cell_x, cell_y, cell_z)
    pml_layers = [mp.PML(thickness=pml)]

    # Use the standard Meep broadband plane-wave construction: the source spans
    # the full transverse cell and is integrated because it extends into PML.
    source_z = -0.5 * cell_z + pml
    source = mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        component=component,
        center=mp.Vector3(0, 0, source_z),
        size=mp.Vector3(cell_x, cell_y, 0),
    )

    outer_length_um = summary["outer_length_um"]
    outer_diameter_um = summary["outer_diameter_um"]
    core_length_um = float(request["nanorod_length_nm"]) / 1000.0
    core_diameter_um = float(request["nanorod_diameter_nm"]) / 1000.0

    # Later geometry entries take precedence in Meep. Define the Ag/Au outer
    # body first and the core second so the inner material overrides overlap.
    geometry = []
    geometry.extend(_spherocylinder(
        mp,
        length_um=outer_length_um,
        diameter_um=outer_diameter_um,
        material=shell,
    ))
    geometry.extend(_spherocylinder(
        mp,
        length_um=core_length_um,
        diameter_um=core_diameter_um,
        material=core,
    ))

    monitor_margin = padding / 3.0
    hx = 0.5 * outer_length_um + monitor_margin
    hy = 0.5 * outer_diameter_um + monitor_margin
    hz = 0.5 * outer_diameter_um + monitor_margin
    non_pml_half_z = 0.5 * cell_z - pml
    source_clearance = padding / 6.0
    source_z_expected = -non_pml_half_z + source_clearance
    # Keep the source definition tied to the standard boundary placement. The
    # expected point is reported as a diagnostic because the source plane itself
    # sits exactly at the inner PML boundary in the standard Meep pattern.
    incident_z = 0.5 * (source_z + (-hz))

    if hx >= 0.5 * cell_x - pml or hy >= 0.5 * cell_y - pml or hz >= 0.5 * cell_z - pml:
        raise ValueError("scattering flux box does not fit inside the non-PML region")
    if not source_z < incident_z < -hz:
        raise ValueError("source/incident monitor/scattering box ordering is invalid")

    def make_sim(geometry_objects):
        return mp.Simulation(
            resolution=resolution,
            cell_size=cell,
            boundary_layers=pml_layers,
            geometry=geometry_objects,
            default_material=background,
            sources=[source],
            k_point=mp.Vector3(),
            Courant=courant,
            eps_averaging=True,
        )

    def add_box(sim):
        regions = [
            mp.FluxRegion(center=mp.Vector3(+hx, 0, 0), size=mp.Vector3(0, 2 * hy, 2 * hz), weight=+1),
            mp.FluxRegion(center=mp.Vector3(-hx, 0, 0), size=mp.Vector3(0, 2 * hy, 2 * hz), weight=-1),
            mp.FluxRegion(center=mp.Vector3(0, +hy, 0), size=mp.Vector3(2 * hx, 0, 2 * hz), weight=+1),
            mp.FluxRegion(center=mp.Vector3(0, -hy, 0), size=mp.Vector3(2 * hx, 0, 2 * hz), weight=-1),
            mp.FluxRegion(center=mp.Vector3(0, 0, +hz), size=mp.Vector3(2 * hx, 2 * hy, 0), weight=+1),
            mp.FluxRegion(center=mp.Vector3(0, 0, -hz), size=mp.Vector3(2 * hx, 2 * hy, 0), weight=-1),
        ]
        return sim.add_flux(fcen, df, nfreq, *regions)

    def add_incident(sim):
        return sim.add_flux(
            fcen,
            df,
            nfreq,
            mp.FluxRegion(
                center=mp.Vector3(0, 0, incident_z),
                size=mp.Vector3(2 * hx, 2 * hy, 0),
            ),
        )

    norm = make_sim([])
    norm_box = add_box(norm)
    incident = add_incident(norm)
    norm.run(until_after_sources=until_after)
    frequencies = np.asarray(mp.get_flux_freqs(incident), dtype=float)
    incident_flux = np.asarray(mp.get_fluxes(incident), dtype=float)
    norm_box_data = norm.get_flux_data(norm_box)
    norm.reset_meep()

    structured = make_sim(geometry)
    scattered_box = add_box(structured)
    structured.load_minus_flux_data(scattered_box, norm_box_data)
    structured.run(until_after_sources=until_after)
    scattered_power = np.asarray(mp.get_fluxes(scattered_box), dtype=float)
    structured.reset_meep()

    monitor_area_um2 = (2.0 * hx) * (2.0 * hy)
    incident_intensity = np.abs(incident_flux) / monitor_area_um2
    if np.any(incident_intensity <= 0):
        raise RuntimeError("non-positive incident intensity in broadband normalization")
    scattering_cross_section = scattered_power / incident_intensity
    wavelengths_nm = 1000.0 / frequencies

    order = np.argsort(wavelengths_nm)
    wavelengths_sorted = wavelengths_nm[order]
    frequencies_sorted = frequencies[order]
    inc_sorted = incident_flux[order]
    scatt_power_sorted = scattered_power[order]
    xsec_sorted = scattering_cross_section[order]

    spectrum = [
        {
            "wavelength_nm": float(w),
            "frequency_inv_um": float(f),
            "incident_flux": float(inc),
            "scattered_power": float(sp),
            "scattering_cross_section_um2": float(xs),
        }
        for w, f, inc, sp, xs in zip(
            wavelengths_sorted,
            frequencies_sorted,
            inc_sorted,
            scatt_power_sorted,
            xsec_sorted,
        )
    ]

    requested_samples = []
    for row in request["wavelength_case_bindings"]:
        w = float(row["wavelength_nm"])
        xs = float(np.interp(w, wavelengths_sorted, xsec_sorted))
        requested_samples.append({
            "source_case_id": row["source_case_id"],
            "wavelength_nm": w,
            "scattering_cross_section_um2": xs,
        })

    elapsed = time.monotonic() - start
    return {
        "schema_version": "sers-meep-worker-result-v0",
        "request_id": request["request_id"],
        "worker_status": "completed",
        "backend": "meep",
        "backend_version": getattr(mp, "__version__", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "wall_seconds": elapsed,
        "peak_rss_mb": _rss_mb(),
        "coordinate_convention": request["coordinate_convention"],
        "field_component": request["field_component"],
        "background_model": request["background_model"],
        "background_index": background_index,
        "material_model": request["material_model"],
        "material_valid_frequency_ranges_inv_um": material_ranges,
        "resolution_px_per_um": resolution,
        "grid_spacing_nm": 1000.0 / resolution,
        "courant": courant,
        "pml_thickness_um": pml,
        "padding_um": padding,
        "cell_um": [cell_x, cell_y, cell_z],
        "source_frequency_center_inv_um": fcen,
        "source_frequency_width_inv_um": df,
        "source_plane_z_um": source_z,
        "diagnostic_source_clearance_point_z_um": source_z_expected,
        "incident_monitor_z_um": incident_z,
        "scattering_box_half_sizes_um": [hx, hy, hz],
        "monitor_area_um2": monitor_area_um2,
        "run_until_after_sources": until_after,
        "spectrum_frequency_points": nfreq,
        "scientific_wavelength_min_nm": request.get("scientific_wavelength_min_nm"),
        "scientific_wavelength_max_nm": request.get("scientific_wavelength_max_nm"),
        "spectral_guard_band_nm": request.get("spectral_guard_band_nm", 0.0),
        "solver_wavelength_min_nm": float(request["wavelength_min_nm"]),
        "solver_wavelength_max_nm": float(request["wavelength_max_nm"]),
        "spectrum": spectrum,
        "requested_case_samples": requested_samples,
        "reported_resonance_wavelengths_nm": request.get("reported_resonance_wavelengths_nm", []),
        "baseline_wavelength_nm": request.get("baseline_wavelength_nm"),
        "numerical_resolution_status": request["resolution_status"],
        "evidence_eligibility": request["evidence_eligibility"],
        "scientific_interpretation_permitted": False,
        "shadow_only": True,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one standalone SERS Meep shadow request")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the request and write geometry/frequency diagnostics without importing Meep.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    request_path = Path(args.request).resolve()
    output_path = Path(args.output).resolve()
    request = _load_json(request_path)
    summary = _validate_request(request)

    if args.validate_only:
        _write_json(output_path, {
            "schema_version": "sers-meep-worker-validation-v0",
            "request_id": request["request_id"],
            "worker_status": "validated_not_executed",
            "summary": summary,
            "shadow_only": True,
            "physics_authority_created": False,
            "hypothesis_rejection_authority": False,
        })
        return 0

    try:
        result = _execute(request, summary)
    except Exception as exc:
        _write_json(output_path, {
            "schema_version": "sers-meep-worker-result-v0",
            "request_id": request.get("request_id", "unknown"),
            "worker_status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "peak_rss_mb": _rss_mb(),
            "scientific_interpretation_permitted": False,
            "shadow_only": True,
            "physics_authority_created": False,
            "hypothesis_rejection_authority": False,
            "feedback_generation_authority": False,
            "canonical_graph_mutated": False,
        })
        raise

    _write_json(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
