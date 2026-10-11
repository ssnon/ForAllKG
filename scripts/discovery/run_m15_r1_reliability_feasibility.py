"""M15-R0/R1: auditable *read-only* numerical feasibility assessment.

No solver/LLM calls. No scientific acceptance is granted by this module.
This add-only command consumes M15B_EVIDENCE_GATE.json and its five actual
Meep diagnostic request/result pairs, checking source hashes before analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "m15-r1-feasibility-v1"
CASES = {
    "ex_t8_r60": "diagnostic_short_v1",
    "ey_t8_r60": "diagnostic_short_trans_v1",
    "ex_t16_r60": "diagnostic_time16_ex_v1",
    "ey_t16_r60": "diagnostic_time16_ey_v1",
    "ey_t16_r90": "diagnostic_resolution90_ey_v1",
}
METRIC_TOLERANCE = 5e-5
FIXED_FIELDS = (
    "geometry_family", "core_material", "shell_material", "nanorod_length_nm",
    "nanorod_diameter_nm", "shell_thickness_nm", "surrounding_medium",
    "background_index", "material_model", "wavelength_min_nm", "wavelength_max_nm",
    "spectrum_frequency_points", "pml_thickness_um", "padding_um",
    "cell_x_um", "cell_y_um", "cell_z_um", "courant",
)


class FeasibilityError(ValueError):
    """Input/lineage problem. Never silently accept bad evidence."""


def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FeasibilityError(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FeasibilityError(f"expected JSON object: {path}")
    return data


def finite_number(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise FeasibilityError(f"missing or nonfinite numeric {name}: {v!r}")
    return float(v)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        p = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                           text=True, timeout=10, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def verify_gate(gate_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, str]]]:
    gate_path = gate_path.resolve()
    gate = read_json(gate_path)
    if gate.get("schema_version") != "m15b-evidence-gate-v1":
        raise FeasibilityError("unsupported M15B evidence gate schema")
    if (gate.get("quantitative_physics_feedback") != "BLOCKED"
            or gate.get("spatial_stability") != "FAIL"):
        raise FeasibilityError("expected M15B BLOCKED/FAIL; this tool cannot qualify earlier spectra")
    if gate.get("scientific_novelty") != "NOT_TESTED":
        raise FeasibilityError("unexpected novelty authority in M15B gate")
    if gate.get("permitted_use") != "numerical_diagnostic_feedback_only":
        raise FeasibilityError("unexpected M15B permitted_use")

    sources = gate.get("sources")
    if not isinstance(sources, list) or len(sources) != 10:
        raise FeasibilityError("expected exactly 10 frozen source files in M15B gate")

    expected_paths = {
        str(gate_path.parent / case / filename)
        for case in CASES.values() for filename in ("request.json", "result.json")
    }
    seen: set[str] = set()
    verified: list[dict[str, str]] = []
    for item in sources:
        if not isinstance(item, dict):
            raise FeasibilityError("invalid source receipt")
        p = Path(item.get("path", "")).resolve()
        if str(p) not in expected_paths or str(p) in seen:
            raise FeasibilityError(f"source path mismatch or duplicate: {p}")
        seen.add(str(p))
        try:
            actual_hash = sha(p.read_bytes())
        except OSError as exc:
            raise FeasibilityError(f"missing source file: {p}") from exc
        if actual_hash != item.get("sha256"):
            raise FeasibilityError(f"SOURCE_SHA256_MISMATCH {p}")
        verified.append({"path": str(p), "sha256": actual_hash})
    if seen != expected_paths:
        raise FeasibilityError("frozen source set mismatch")

    cases: dict[str, dict[str, Any]] = {}
    for key, directory in CASES.items():
        request = read_json(gate_path.parent / directory / "request.json")
        result = read_json(gate_path.parent / directory / "result.json")
        if result.get("worker_status") != "completed":
            raise FeasibilityError(f"incomplete worker result: {key}")
        if request.get("request_id") != result.get("request_id"):
            raise FeasibilityError(f"request-to-result ID mismatch: {key}")
        if request.get("field_component") != result.get("field_component"):
            raise FeasibilityError(f"request-to-result polarization mismatch: {key}")
        if request.get("spectral_observable") != "scattering_cross_section_um2":
            raise FeasibilityError("worker observable is not the expected scattering cross section")
        if request.get("evidence_eligibility") != "execution_sanity_only":
            raise FeasibilityError("expected unqualified diagnostic source")
        for numeric in ("resolution_px_per_um", "run_until_after_sources"):
            if not math.isclose(finite_number(request.get(numeric), numeric),
                                finite_number(result.get(numeric), numeric), rel_tol=1e-10):
                raise FeasibilityError(f"request/result mismatch for {numeric}: {key}")
        if result.get("scientific_interpretation_permitted") is not False:
            raise FeasibilityError(f"unexpected scientific authority in raw result: {key}")
        points = result.get("spectrum")
        if not isinstance(points, list) or len(points) < 3:
            raise FeasibilityError(f"invalid spectrum length: {key}")
        wave, scat = [], []
        for j, row in enumerate(points):
            if not isinstance(row, dict):
                raise FeasibilityError(f"invalid spectrum row: {key} {j}")
            wave.append(finite_number(row.get("wavelength_nm"), "wavelength_nm"))
            scat.append(finite_number(row.get("scattering_cross_section_um2"), "scattering_cross_section_um2"))
        if any(v <= 0 for v in wave) or any(v < 0 for v in scat):
            raise FeasibilityError(f"negative or zero unphysical spectrum quantity: {key}")
        if len(set(wave)) != len(wave):
            raise FeasibilityError(f"duplicate wavelength: {key}")
        if sorted(wave) != wave and sorted(wave, reverse=True) != wave:
            raise FeasibilityError(f"wavelength samples unsorted: {key}")
        cases[key] = {"request": request, "result": result, "wavelength": wave, "scattering": scat}
    return gate, cases, verified


def _same(a: dict[str, Any], b: dict[str, Any], fields: tuple[str, ...], reason: str) -> None:
    for name in fields:
        if a.get(name) != b.get(name):
            raise FeasibilityError(f"{reason}: {name}")


def _relative_l1(a: dict[str, Any], b: dict[str, Any]) -> float:
    wa, wb = a["wavelength"], b["wavelength"]
    if len(wa) != len(wb) or any(abs(x-y) > 1e-6 for x,y in zip(wa,wb)):
        raise FeasibilityError("cannot compare nonmatching wavelength grids")
    xa, xb = a["scattering"], b["scattering"]
    den = sum(abs(x) for x in xb)
    if den <= 0:
        raise FeasibilityError("zero scattering denominator")
    return sum(abs(x-y) for x,y in zip(xa,xb)) / den


def analyze(gate: dict[str, Any], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ref = cases["ex_t8_r60"]["request"]
    for key, item in cases.items():
        _same(ref, item["request"], FIXED_FIELDS, "physical configuration drift in " + key)
        pol = "Ex" if key.startswith("ex_") else "Ey"
        res = 90 if key == "ey_t16_r90" else 60
        t = 8 if "t8" in key else 16
        if (item["request"].get("field_component") != pol
                or float(item["request"].get("resolution_px_per_um", 0)) != res
                or float(item["request"].get("run_until_after_sources", 0)) != t):
            raise FeasibilityError("unexpected polarization/time/resolution fixture in " + key)
    ex8, ex16 = cases["ex_t8_r60"], cases["ex_t16_r60"]
    ey8, ey16, ey90 = cases["ey_t8_r60"], cases["ey_t16_r60"], cases["ey_t16_r90"]
    ex_time = _relative_l1(ex8, ex16)
    ey_time = _relative_l1(ey8, ey16)
    spatial = _relative_l1(ey16, ey90)
    for key, new in [("ex_time_relative_l1", ex_time), ("ey_time_relative_l1", ey_time),
                     ("ey_spatial_relative_l1", spatial)]:
        # M15-B only explicitly froze the spatial metric; other metrics are checked
        # against their independently recomputed samples, not invented gate fields.
        old = gate.get(key)
        if old is not None and abs(finite_number(old, key) - new) > METRIC_TOLERANCE:
            raise FeasibilityError(f"stored metric inconsistent: {key}")
    gate_spatial = finite_number(gate.get("ey_spatial_relative_l1_change"), "gate spatial L1")
    if abs(gate_spatial - spatial) > METRIC_TOLERANCE:
        raise FeasibilityError("M15B stored spatial L1 does not match raw spectra")

    ex_peak_idx = max(range(len(ex16["scattering"])), key=lambda i: ex16["scattering"][i])
    ey60_idx = max(range(len(ey16["scattering"])), key=lambda i: ey16["scattering"][i])
    ey90_idx = max(range(len(ey90["scattering"])), key=lambda i: ey90["scattering"][i])
    ex_upper = ex_peak_idx == max(range(len(ex16["wavelength"])), key=lambda i: ex16["wavelength"][i])
    if not ex_upper or gate.get("ex_peak_location") != "UPPER_BOUNDARY_CENSORED":
        raise FeasibilityError("Ex peak censoring inconsistent with frozen gate")
    shell_nm = finite_number(ref.get("shell_thickness_nm"), "shell thickness")
    if abs(shell_nm - 20.0) > 1e-9:
        raise FeasibilityError("this R1 evaluation expects the actual M15 20nm shell cohort")
    return {
        "ex_time_relative_l1": ex_time,
        "ey_time_relative_l1": ey_time,
        "ey_spatial_relative_l1": spatial,
        "ey_60_peak_sample_nm": ey16["wavelength"][ey60_idx],
        "ey_90_peak_sample_nm": ey90["wavelength"][ey90_idx],
        "ex_peak_status": "RIGHT_CENSORED_NOT_LOCATED",
        "shell_thickness_nm": shell_nm,
        "shell_cells_r60": shell_nm * 60 / 1000,
        "shell_cells_r90": shell_nm * 90 / 1000,
        "latest_measured_wall_seconds": finite_number(ey90["result"].get("wall_seconds"), "Ey 90 wall_seconds"),
        "latest_measured_peak_rss_mb": (finite_number(ey90["result"].get("peak_rss_mb"), "peak_rss_mb")
                                         if ey90["result"].get("peak_rss_mb") is not None else None),
        "geometry": {k: ref[k] for k in ("geometry_family", "core_material", "shell_material",
                                           "nanorod_length_nm", "nanorod_diameter_nm", "shell_thickness_nm",
                                           "surrounding_medium", "background_index")},
        "current_observable": "scattering_cross_section_um2",
        "direct_sers_band_ratio_available": False,
    }


def _cost_estimates(metrics: dict[str, Any], target_cells: float) -> dict[str, Any]:
    target_res = 1000.0 * target_cells / metrics["shell_thickness_nm"]
    ratio = target_res / 90.0
    # fixed-domain grid cost scales ~r^3; timestep count ~r, hence ~r^4 wall-time.
    # This is a heuristic upper/lower-uncertain budget indication, not a promise.
    return {
        "estimated_target_resolution_px_per_um": target_res,
        "target_shell_cells": target_cells,
        "reference_resolution_px_per_um": 90.0,
        "reference_wall_seconds": metrics["latest_measured_wall_seconds"],
        "wall_scaling_ratio_r4_heuristic": ratio**4,
        "estimated_wall_seconds_per_polarization_heuristic": metrics["latest_measured_wall_seconds"] * ratio**4,
        "memory_scaling_ratio_r3_heuristic": ratio**3,
        "reference_peak_rss_mb": metrics["latest_measured_peak_rss_mb"],
        "estimated_peak_rss_mb_heuristic": (metrics["latest_measured_peak_rss_mb"] * ratio**3
                                            if metrics["latest_measured_peak_rss_mb"] is not None else None),
        "validity": "rough fixed-domain 3D FDTD scaling only; no authorized solver run",
    }


def _snapshot(repo_dir: Path) -> dict[str, Any]:
    return {
        "repo_directory": str(repo_dir.resolve()),
        "git_branch": _git(["branch", "--show-current"], repo_dir),
        "git_commit": _git(["rev-parse", "HEAD"], repo_dir),
        "worktree_porcelain": _git(["status", "--porcelain", "--untracked-files=no"], repo_dir),
        "git_status_is_clean": _git(["status", "--porcelain", "--untracked-files=no"], repo_dir) == "",
        "note": "Git metadata is descriptive; the script does not change repo files.",
    }


def create_assessment(gate_path: Path, repo_dir: Path, target_cells: float,
                      prior_art_path: Path | None = None) -> dict[str, Any]:
    if not 2 <= target_cells <= 20:
        raise FeasibilityError("target-shell-cells must be in [2,20]; it is a planning parameter, not certified resolution")
    gate, cases, sources = verify_gate(gate_path)
    metrics = analyze(gate, cases)
    cost = _cost_estimates(metrics, target_cells)
    prior_art = {"status": "NOT_SUPPLIED", "claim_novelty_permitted": False}
    if prior_art_path is not None:
        raw = prior_art_path.resolve().read_bytes()
        prior_art = {"status": "USER_SUPPLIED_UNREVIEWED", "path": str(prior_art_path.resolve()),
                     "sha256": sha(raw), "claim_novelty_permitted": False}
    frozen_seed = {
        "schema_version": "m15r-frozen-seed-v1",
        "seed_type": "EXPLICIT_BENCHMARK_SEED_NOT_M14_DISCOVERY",
        "parent_research_idea_id": None,
        "research_question": (
            "For isolated Au-core/Ag-shell nanorods, can a qualified EM model distinguish "
            "the effect of changing uniform shell thickness from 20 to 30 nm on "
            "single-particle Ey scattering cross section at 633 nm?"
        ),
        "applicability": "EM_DIRECT",
        "requested_observable": "scattering_cross_section_um2",
        "geometry": metrics["geometry"],
        "x0": {"shell_thickness_nm": 20.0, "polarization": "Ey", "wavelength_nm": 633.0,
               "evidence": "PREEXISTING_BROADBAND_SOURCE_UNQUALIFIED", "may_feed_science_arm": False},
        "x1_held_out": {"shell_thickness_nm": 30.0, "polarization": "Ey", "wavelength_nm": 633.0,
                        "evidence": "NOT_ACQUIRED", "may_feed_science_arm": False},
        "scientific_null": "Any apparent x0/x1 contrast is no greater than qualified numerical/model uncertainty.",
        "alternative": "An x0/x1 contrast exceeds preregistered combined numerical/model uncertainty.",
        "numeric_effect_prediction": None,
        "contrast_and_tolerance_status": "REQUIRES_PREDECLARED_PREDICTION_AND_ERROR_BUDGET_BEFORE_X1",
        "prior_art": prior_art,
        "warning": "Neither alternative constitutes new physics or novelty; no x1 calculation exists in this file.",
    }
    # No in-scope positive control/model-form validation or convergence-qualified
    # evidence is present. Do NOT choose MEEP_REPAIR as a completed solver.
    backend = {
        "schema_version": "m15r-backend-decision-v1",
        "science_backend_choice": "NO_QUALIFIED_BACKEND",
        "science_x0_x1_status": "BLOCKED",
        "decision_reason": [
            "M15B spatial L1 drift is 27.35%; no spatial convergence",
            "20nm Ag shell spans only 1.2 to 1.8 grid cells",
            "Ex peak is upper-boundary censored",
            "No independent positive-control result or material-model validation attached",
            "No verified independent FEM/BEM implementation attached",
            "Current Meep worker reports scattering only, not SERS band ratios",
        ],
        "candidate_backends": [
            {"choice": "MEEP_REPAIR", "status": "UNVALIDATED_COSTED_OPTION", "cost": cost,
             "next_test": "Predeclare actual scattering accuracy target, validate matched control and run scoped mesh/domain/PML sensitivity only after resource approval"},
            {"choice": "INDEPENDENT_EM_SOLVER", "status": "NOT_DEMONSTRATED",
             "next_test": "Find an available FEM/BEM backend that matches core-shell geometry, dispersive Au/Ag and scattering observable; validate independently"},
            {"choice": "ANALYTIC_BENCHMARK_ONLY", "status": "OPTIONAL_SOFTWARE_LOOP_FALLBACK",
             "next_test": "Use independently validated dielectric-sphere Mie/Rayleigh benchmark only to test the algorithmic feedback plumbing; no SERS or nanorod claims"},
            {"choice": "NO_QUALIFIED_BACKEND", "status": "ACTIVE_SCIENCE_GATE", "next_test": "Wait for qualified x0 and held-out x1 oracle"},
        ],
        "must_not_use_as_quantitative_science": list(CASES.values()),
        "automated_solver_execution": False,
        "automated_model_calls": False,
        "permission_to_modify_canonical_graph": False,
        "scientific_interpretation_permitted": False,
        "physics_claim_authority": False,
        "novelty_claim_authority": False,
    }
    evidence = {
        "schema_version": "m15r-qualified-evidence-v1",
        "qualification": "NOT_QUALIFIED",
        "numerical_resolution": "FAILED_SPATIAL_STABILITY",
        "model_validation": "NOT_ASSESSED",
        "positive_control": "NOT_ATTACHED",
        "intended_use": "planning_and_repair_only",
        "scientific_interpretation_permitted": False,
        "no_authority_over_sers_mechanism": True,
        "metrics": metrics,
    }
    return {
        "provenance": {
            "schema_version": "m15r-r0-provenance-v1",
            "source_gate_path": str(gate_path.resolve()),
            "source_gate_sha256": sha(gate_path.read_bytes()),
            "source_count": len(sources),
            "sources": sources,
            "git": _snapshot(repo_dir),
            "original_gate_unmodified": True,
            "policy": "READ_ONLY_OLD_ARTIFACTS_ADD_ONLY_NEW_REPORTS",
        },
        "seed": frozen_seed,
        "backend": backend,
        "evidence": evidence,
    }


def _report(outputs: dict[str, Any]) -> str:
    d = outputs["backend"]
    m = outputs["evidence"]["metrics"]
    cost = d["candidate_backends"][0]["cost"]
    return ("# M15-R0/R1 — Direct patch feasibility report\n\n"
            "## Outcome\n\n"
            f"**Scientific backend: {d['science_backend_choice']}**; **R2/R3: BLOCKED**.\n\n"
            "This is a negative feasibility finding for *current* sources, not a claim that reliable numerical EM is impossible.\n\n"
            "## Verified local computational evidence\n\n"
            f"- Ex temporal relative L1: {m['ex_time_relative_l1']:.6%}\n"
            f"- Ey temporal relative L1: {m['ey_time_relative_l1']:.6%}\n"
            f"- Ey spatial relative L1 (60→90 px/um): **{m['ey_spatial_relative_l1']:.4%}**\n"
            f"- Thin shell sampled by {m['shell_cells_r60']:.2f} / {m['shell_cells_r90']:.2f} cells\n"
            "- Ex observed global maximum: **right-censored**, no identified resonance position\n\n"
            "## Resource assessment (heuristic, not a planned automatic run)\n\n"
            f"- Example target {cost['target_shell_cells']} cells/shell requires {cost['estimated_target_resolution_px_per_um']:.0f} px/um\n"
            f"- 3D r^4 scaling vs measured 90 px/um job: ~{cost['wall_scaling_ratio_r4_heuristic']:.1f}×, "
            f"or ~{cost['estimated_wall_seconds_per_polarization_heuristic']/3600:.1f} hours per job (high uncertainty)\n"
            "- Not authorization for any simulation. This excludes possible geometry-window and solver changes.\n\n"
            "## Path to a genuine G1 → numerical feedback → G2 loop\n\n"
            "1. Select a directly relevant EM observable and obtain *qualified*, matched-model x0 (with numerical error bounds).\n"
            "2. Pre-register held-out x1, rival predictions, and numerical resolution targets **before** its computation.\n"
            "3. Freeze matched Control/Treatment G1 responses, compute independent x1, then track changed G2 prediction.\n"
            "4. Independently evaluate conditional novelty; no Maxwell calculation itself proves novelty.\n\n"
            "## Recommended bounded next action\n\n"
            "Do **not** run 250 px/um Meep by default. First locate a validated local-mesh FEM/BEM "
            "solver/independent matched experimental reference for the existing geometry and scattering observable. "
            "If not realistically available, use a sphere benchmark solely to demonstrate algorithmic feedback causality, "
            "or record NO_QUALIFIED_BACKEND and stop expensive numerical escalation.\n\n"
            "## Authority\n\n"
            "M15B stays BLOCKED. No prior source changed. No model calls, no Meep runs, no scientific support/rejection authority.\n")


def _exclusive_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)


def run(gate_path: Path, out_dir: Path, repo_dir: Path, target_cells: float,
        prior_art_path: Path | None = None) -> dict[str, Any]:
    outputs = create_assessment(gate_path, repo_dir, target_cells, prior_art_path)
    paths = {
        "provenance": "M15R_R0_PROVENANCE.json",
        "seed": "M15R_FROZEN_SEED.json",
        "backend": "M15R_BACKEND_DECISION.json",
        "evidence": "M15R_QUALIFIED_EVIDENCE.json",
    }
    artifacts = {filename: canonical_bytes(outputs[key]) for key, filename in paths.items()}
    artifacts["M15R_R1_REPORT.md"] = _report(outputs).encode("utf-8")
    manifest = {
        "schema_version": "m15r-r1-manifest-v1",
        "files": {name: sha(raw) for name, raw in sorted(artifacts.items())},
        "no_paid_calls": True,
        "no_solver_calls": True,
        "scientific_interpretation_permitted": False,
    }
    artifacts["M15R_R1_MANIFEST.json"] = canonical_bytes(manifest)
    out_dir = out_dir.resolve()
    # Exclusively new output directory to prevent any overwrite or replay confusion.
    if out_dir.exists():
        if any(out_dir.iterdir()):
            raise FeasibilityError("output directory is not empty; choose a new run directory")
    else:
        out_dir.mkdir(parents=True)
    for name, content in artifacts.items():
        _exclusive_write(out_dir / name, content)
    return {"status": "M15R_R1_ASSESSED", "verified_sources": 10,
            "science_backend": outputs["backend"]["science_backend_choice"],
            "relative_spatial_l1": outputs["evidence"]["metrics"]["ey_spatial_relative_l1"],
            "out_dir": str(out_dir), "new_fdtd_runs": 0, "new_paid_calls": 0,
            "science_evidence_qualified": False}


def check(out_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    manifest = read_json(out_dir / "M15R_R1_MANIFEST.json")
    if manifest.get("schema_version") != "m15r-r1-manifest-v1":
        raise FeasibilityError("wrong manifest schema")
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise FeasibilityError("unsafe manifest filename")
        p = out_dir / name
        if sha(p.read_bytes()) != expected:
            raise FeasibilityError(f"ARTIFACT_SHA256_MISMATCH {name}")
    provenance = read_json(out_dir / "M15R_R0_PROVENANCE.json")
    gate_path = Path(provenance["source_gate_path"])
    if sha(gate_path.read_bytes()) != provenance["source_gate_sha256"]:
        raise FeasibilityError("SOURCE_GATE_CHANGED")
    _, _, verified = verify_gate(gate_path)
    if {(r["path"],r["sha256"]) for r in verified} != \
        {(r["path"],r["sha256"]) for r in provenance["sources"]}:
        raise FeasibilityError("sources have changed")
    backend = read_json(out_dir / "M15R_BACKEND_DECISION.json")
    evidence = read_json(out_dir / "M15R_QUALIFIED_EVIDENCE.json")
    if backend.get("science_backend_choice") != "NO_QUALIFIED_BACKEND" or \
       evidence.get("scientific_interpretation_permitted") is not False:
        raise FeasibilityError("unexpected science permission")
    return {"status":"M15R_R1_CHECK_OK", "source_hashes_valid": True,
            "r2_r3":"BLOCKED_NO_QUALIFIED_BACKEND", "new_fdtd_runs":0, "new_paid_calls":0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    assess = sub.add_parser("assess", help="read and evaluate frozen M15B inputs; no numerical run")
    assess.add_argument("--gate", type=Path, required=True)
    assess.add_argument("--out-dir", type=Path, required=True)
    assess.add_argument("--repo-dir", type=Path, default=Path.cwd())
    assess.add_argument("--target-shell-cells", type=float, default=5)
    assess.add_argument("--prior-art-snapshot", type=Path)
    verify = sub.add_parser("check", help="verify output and source receipts")
    verify.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "assess":
            output = run(args.gate,args.out_dir,args.repo_dir,args.target_shell_cells,args.prior_art_snapshot)
        else:
            output = check(args.out_dir)
    except (FeasibilityError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, "M15R_R1_BLOCKED_INPUT_ERROR: " + str(exc) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
