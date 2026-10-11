"""M15-R4: fail-closed SCUFF-EM candidate preparation for matched Au/Ag rods.

This module NEVER runs the solver, approves material models, or certifies science.
It emits Gmsh CAD source geometry and a BEM handoff plan from an existing
Meep request with explicit provenance and physical-scope restrictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

VERSION = "m15r4-bem-candidate-v0"


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as f:
        f.write(content)


def _json_new(path: Path, data: object) -> None:
    _write_new(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _validate(request: dict) -> None:
    if request.get("schema_version") != "sers-meep-execution-request-v0":
        raise ValueError("REQUIRES_MEEP_REQUEST_V0")
    if request.get("geometry_family") != "core_shell_nanorod":
        raise ValueError("ONLY_CORE_SHELL_NANOROD")
    for key in ("core_material", "shell_material"):
        if request.get(key) not in {"Au", "Ag"}:
            raise ValueError("ONLY_AU_AG_MATERIALS")
    if request["core_material"] == request["shell_material"]:
        raise ValueError("REQUIRES_TWO_DISTINCT_METALS_FOR_FIRST_PILOT")
    if request.get("field_component") not in {"Ex", "Ey"}:
        raise ValueError("ONLY_EX_OR_EY")
    if request.get("coordinate_convention") != "nanorod_x_incident_z":
        raise ValueError("COORDINATE_CONVENTION_MISMATCH")
    if request.get("incident_propagation") != "normal_to_nanorod_axis":
        raise ValueError("PROPAGATION_CONVENTION_MISMATCH")
    if request.get("surrounding_medium") not in {"air", "water"}:
        raise ValueError("MEDIUM_REQUIRES_OPERATOR_REVIEW")
    for key in ("nanorod_length_nm", "nanorod_diameter_nm", "shell_thickness_nm"):
        if not isinstance(request.get(key), (int, float)) or isinstance(request[key], bool) or request[key] <= 0:
            raise ValueError(f"BAD_DIMENSION_{key}")
    if request["nanorod_length_nm"] <= request["nanorod_diameter_nm"]:
        raise ValueError("NOT_A_CAPSULE")
    if not request.get("wavelength_min_nm", 0) < request.get("wavelength_max_nm", 0):
        raise ValueError("BAD_SPECTRAL_WINDOW")


def _capsule_geo(length_nm: float, diameter_nm: float, edge_nm: float, title: str) -> str:
    """Same spherocylinder convention as Meep: cylinder + two hemispherical caps.

    CAD booleans and mesh are NOT claimed validated before actual Gmsh execution.
    """
    radius = diameter_nm / 2000.0
    barrel = (length_nm - diameter_nm) / 1000.0
    half = barrel / 2.0
    edge_um = edge_nm / 1000.0
    return (
        f'// M15-R4 {title}; lengths in micrometers; candidate geometry only\n'
        'SetFactory("OpenCASCADE");\n'
        f'Mesh.MeshSizeMin = {edge_um:.12g};\n'
        f'Mesh.MeshSizeMax = {edge_um:.12g};\n'
        'Mesh.MshFileVersion = 2.2;\n'
        f'Cylinder(1) = {{{-half:.12g}, 0, 0, {barrel:.12g}, 0, 0, {radius:.12g}}};\n'
        f'Sphere(2) = {{{-half:.12g}, 0, 0, {radius:.12g}}};\n'
        f'Sphere(3) = {{{half:.12g}, 0, 0, {radius:.12g}}};\n'
        'parts[] = BooleanUnion{ Volume{1}; Delete; }{ Volume{2, 3}; Delete; };\n'
        'boundary[] = Boundary{ Volume{parts[]}; };\n'
        'Physical Surface(101) = {boundary[]};\n'
    )


def _args_template(polarization: str, wavelength_nm: float) -> str:
    pol = "1 0 0" if polarization == "Ex" else "0 1 0"
    return (
        "# Candidate only: do not execute without material/mesh validation.\n"
        "geometry candidate.scuffgeo\n"
        f"Lambda {wavelength_nm / 1000.0:.12g}\n"
        "pwDirection 0 0 1\n"
        f"pwPolarization {pol}\n"
        "PFTFile candidate.PFT\n"
    )


def _geo_template(core_material: str, shell_material: str, medium: str) -> str:
    # Absorbed and scattered power must be transformed into cross section with
    # independently checked incident intensity and scuff units. No direct
    # equality to Meep scattering cross section is assumed here.
    medium_expr = "CONST_EPS_1.7689" if medium == "water" else "VACUUM"
    return (
        "# DO NOT RUN before optical constants and mesh are verified.\n"
        "# FILE_Au_eps.dat and FILE_Ag_eps.dat must be genuine dispersion tables\n"
        "# in rad/s: omega epsilon_complex. Not supplied by this patch.\n"
        f"REGION Exterior MATERIAL {medium_expr}\n"
        f"OBJECT OuterShell\n  MESHFILE outer_shell.msh\n  MATERIAL FILE_{shell_material}_eps.dat\nENDOBJECT\n"
        f"OBJECT InnerCore\n  MESHFILE inner_core.msh\n  MATERIAL FILE_{core_material}_eps.dat\nENDOBJECT\n"
    )


def create(source: Path, out: Path, edge_nm: float, wavelength_nm: float | None) -> dict:
    if not 1.0 <= edge_nm <= 30.0:
        raise ValueError("MESH_EDGE_NM_MUST_BE_1_TO_30")
    raw = source.resolve().read_bytes()
    req = json.loads(raw)
    _validate(req)
    outer_len = req["nanorod_length_nm"] + 2 * req["shell_thickness_nm"]
    outer_dia = req["nanorod_diameter_nm"] + 2 * req["shell_thickness_nm"]
    wmin, wmax = req["wavelength_min_nm"], req["wavelength_max_nm"]
    wl = (wmin + wmax) / 2 if wavelength_nm is None else wavelength_nm
    if not wmin <= wl <= wmax:
        raise ValueError("WAVELENGTH_OUTSIDE_ORIGINAL_MODEL_WINDOW")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("OUTPUT_DIRECTORY_NOT_EMPTY_NO_OVERWRITE")
    out.mkdir(parents=True, exist_ok=True)
    _write_new(out / "inner_core.geo", _capsule_geo(req["nanorod_length_nm"], req["nanorod_diameter_nm"], edge_nm, "INNER_CORE"))
    _write_new(out / "outer_shell.geo", _capsule_geo(outer_len, outer_dia, edge_nm, "OUTER_SURFACE"))
    _write_new(out / "candidate.scuffgeo.template", _geo_template(req["core_material"], req["shell_material"], req["surrounding_medium"]))
    _write_new(out / "scuff-scatter.args.template", _args_template(req["field_component"], wl))
    plan = {
        "schema_version": VERSION,
        "source_request": str(source.resolve()),
        "source_request_sha256": _hash(raw),
        "source_request_id": req["request_id"],
        "hypothesis_id": req["hypothesis_id"],
        "geometry_candidate_id": req["geometry_candidate_id"],
        "core_material": req["core_material"],
        "shell_material": req["shell_material"],
        "medium": req["surrounding_medium"],
        "field_component": req["field_component"],
        "spectrum_wavelength_bounds_nm": [wmin, wmax],
        "first_test_wavelength_nm": wl,
        "mesh_edge_nm": edge_nm,
        "geometry_um": {
            "inner_length": req["nanorod_length_nm"] / 1000,
            "inner_diameter": req["nanorod_diameter_nm"] / 1000,
            "outer_length": outer_len / 1000,
            "outer_diameter": outer_dia / 1000,
        },
        "generated_files": {
            name: _hash((out / name).read_bytes())
            for name in ["inner_core.geo", "outer_shell.geo", "candidate.scuffgeo.template", "scuff-scatter.args.template"]
        },
        "science_status": "NO_QUALIFIED_BACKEND",
        "review_gate": {
            "geometry_mesh_generated": False,
            "geometry_watertightness_validated": False,
            "material_dispersion_source_validated": False,
            "material_units_and_sign_validated": False,
            "incident_power_to_cross_section_normalization_validated": False,
            "benchmark_comparison_passed": False,
            "mesh_convergence_passed": False,
            "matched_nanorod_science_qualified": False,
        },
        "must_not_claim": ["SERS_band_ratio", "novel_physics", "quantitative_nanorod_scattering", "converged_solver"],
    }
    _json_new(out / "M15R4_BEM_CANDIDATE_PLAN.json", plan)
    return {
        "status": "M15R4_BEM_CANDIDATE_PREPARED_UNVALIDATED",
        "geometry": "nested_spherocylinders",
        "material_data_provided": False,
        "gmsh_available": shutil.which("gmsh") is not None,
        "scuff_scatter_available": shutil.which("scuff-scatter") is not None,
        "science_gate": "BLOCKED",
        "out_dir": str(out.resolve()),
    }


def check(out: Path) -> dict:
    plan = json.loads((out / "M15R4_BEM_CANDIDATE_PLAN.json").read_text())
    assert plan["schema_version"] == VERSION
    assert _hash(Path(plan["source_request"]).read_bytes()) == plan["source_request_sha256"], "SOURCE_HASH_CHANGED"
    for name, expected in plan["generated_files"].items():
        assert _hash((out / name).read_bytes()) == expected, "GENERATED_FILE_CHANGED_" + name
    assert plan["review_gate"]["matched_nanorod_science_qualified"] is False
    return {
        "status": "M15R4_PREFLIGHT_OK_NO_SCIENCE_AUTHORITY",
        "gmsh_available": shutil.which("gmsh") is not None,
        "scuff_scatter_available": shutil.which("scuff-scatter") is not None,
        "optical_files_present": {m: (out / f"{m}_eps.dat").exists() for m in ("Au", "Ag")},
        "science_gate": "BLOCKED",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--request", type=Path, required=True)
    q.add_argument("--out-dir", type=Path, required=True)
    q.add_argument("--mesh-edge-nm", type=float, default=8.0)
    q.add_argument("--wavelength-nm", type=float)
    q = sub.add_parser("check")
    q.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        result = create(args.request, args.out_dir, args.mesh_edge_nm, args.wavelength_nm) if args.cmd == "prepare" else check(args.out_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, AssertionError, FileExistsError, KeyError, OSError, json.JSONDecodeError) as e:
        p.exit(2, f"M15R4_ERROR: {e}\n")


if __name__ == "__main__":
    main()
