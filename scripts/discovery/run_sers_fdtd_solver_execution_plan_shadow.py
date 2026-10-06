#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from domains.sers.solver_execution_planner import SERSFDTDSolverExecutionPlanner
from domains.sers.validation_design_contracts import SERSFDTDSimulationCaseBundle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coalesce SERS solver-neutral wavelength cases into shadow-only "
            "broadband Meep subprocess jobs. This slice does not execute Meep "
            "and intentionally leaves numerical settings unresolved."
        )
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path = Path(args.cases).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_bundle = SERSFDTDSimulationCaseBundle.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    jobs = SERSFDTDSolverExecutionPlanner().plan(case_bundle)

    jobs_path = output_dir / "sers_fdtd.solver_jobs.shadow.json"
    manifest_path = output_dir / "sers_fdtd.solver_execution_plan_shadow.manifest.json"
    _write_json(jobs_path, jobs)

    manifest = {
        "schema_version": "sers-fdtd-solver-execution-plan-shadow-manifest-v0",
        "source_cases": str(source_path),
        "source_cases_sha256": _sha256_file(source_path),
        "solver_jobs": str(jobs_path),
        "solver_jobs_sha256": _sha256_file(jobs_path),
        "scientific_case_count": jobs.scientific_case_count,
        "solver_job_count": jobs.solver_job_count,
        "coalesced_scientific_case_count": jobs.coalesced_scientific_case_count,
        "execution_strategy_counts": {
            "broadband_frequency_sweep": len(jobs.jobs),
        },
        "numerical_policy_status_counts": {
            "requires_numerical_policy": len(jobs.jobs),
        },
        "shadow_only": True,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"solver_jobs={jobs_path}")
    print(f"manifest={manifest_path}")
    print(f"scientific_case_count={jobs.scientific_case_count}")
    print(f"solver_job_count={jobs.solver_job_count}")
    print(f"coalesced_scientific_case_count={jobs.coalesced_scientific_case_count}")
    print("execution_status=planned_not_executable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
