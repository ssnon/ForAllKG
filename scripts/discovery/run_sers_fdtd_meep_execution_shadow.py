#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from domains.sers.meep_execution import SERSMeepExecutionPlanner
from domains.sers.meep_execution_contracts import (
    SERSMeepExecutionRecord,
    SERSMeepExecutionRecordBundle,
)
from domains.sers.numerical_policy_contracts import SERSFDTDNumericalPlanBundle
from domains.sers.solver_execution_contracts import SERSFDTDSolverJobBundle


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
            "Compile one or more numerical SERS FDTD plans into standalone Meep "
            "worker requests. Execution is opt-in and shadow-only."
        )
    )
    parser.add_argument("--solver-jobs", required=True)
    parser.add_argument("--numerical-plans", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-id", action="append")
    parser.add_argument("--geometry-candidate-id")
    parser.add_argument("--polarization")
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--validate-worker-only", action="store_true")
    parser.add_argument("--solver-python")
    parser.add_argument("--conda-executable", default="conda")
    parser.add_argument("--conda-env", default="fdtd-meep")
    parser.add_argument("--worker-script")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--allow-resource-unknown", action="store_true")
    parser.add_argument("--allow-resource-deferred", action="store_true")
    parser.add_argument("--allow-non-sanity", action="store_true")
    return parser.parse_args()


def _command(args: argparse.Namespace, worker: Path, request: Path, result: Path) -> list[str]:
    base = []
    if args.solver_python:
        base = [args.solver_python]
    else:
        base = [
            args.conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            args.conda_env,
            "python",
        ]
    cmd = base + [
        str(worker),
        "--request",
        str(request),
        "--output",
        str(result),
    ]
    if args.validate_worker_only:
        cmd.append("--validate-only")
    return cmd


def main() -> int:
    args = _parse_args()
    solver_path = Path(args.solver_jobs).resolve()
    numerical_path = Path(args.numerical_plans).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = SERSFDTDSolverJobBundle.model_validate_json(
        solver_path.read_text(encoding="utf-8")
    )
    numerical = SERSFDTDNumericalPlanBundle.model_validate_json(
        numerical_path.read_text(encoding="utf-8")
    )

    planner = SERSMeepExecutionPlanner()
    requests = planner.build_requests(
        jobs,
        numerical,
        job_ids=set(args.job_id) if args.job_id else None,
        geometry_candidate_id=args.geometry_candidate_id,
        polarization=args.polarization,
        max_requests=args.max_requests,
    )

    request_bundle_path = output_dir / "sers_fdtd.meep_execution_requests.shadow.json"
    _write_json(request_bundle_path, requests)

    request_dir = output_dir / "requests"
    raw_dir = output_dir / "raw"
    log_dir = output_dir / "logs"
    request_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    worker = (
        Path(args.worker_script).resolve()
        if args.worker_script
        else (Path(__file__).resolve().parent / "sers_fdtd_meep_worker.py")
    )
    if not worker.exists():
        raise FileNotFoundError(f"Meep worker not found: {worker}")

    records = []
    any_failure = False
    for request in requests.requests:
        request_path = request_dir / f"{request.request_id.replace(':', '_')}.json"
        result_path = raw_dir / f"{request.request_id.replace(':', '_')}.result.json"
        stdout_path = log_dir / f"{request.request_id.replace(':', '_')}.stdout.txt"
        stderr_path = log_dir / f"{request.request_id.replace(':', '_')}.stderr.txt"
        _write_json(request_path, request)

        cmd = _command(args, worker, request_path, result_path)
        if not args.execute:
            records.append(SERSMeepExecutionRecord(
                request_id=request.request_id,
                source_numerical_plan_id=request.source_numerical_plan_id,
                source_solver_job_id=request.source_solver_job_id,
                hypothesis_id=request.hypothesis_id,
                geometry_candidate_id=request.geometry_candidate_id,
                execution_status="planned_not_executed",
                command=cmd,
                numerical_resolution_status=request.resolution_status,
                evidence_eligibility=request.evidence_eligibility,
            ))
            continue

        planner.assert_execution_allowed(
            request,
            allow_resource_unknown=args.allow_resource_unknown,
            allow_resource_deferred=args.allow_resource_deferred,
            allow_non_sanity=args.allow_non_sanity,
        )

        started = time.monotonic()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                completed = subprocess.run(
                    cmd,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=env,
                    timeout=args.timeout_seconds,
                    check=False,
                )
            wall = time.monotonic() - started
            status = "completed" if completed.returncode == 0 else "failed"
            error_message = None
            if completed.returncode != 0:
                error_message = f"worker exited with return code {completed.returncode}"
                any_failure = True
            result_sha = _sha256_file(result_path) if result_path.exists() else None
            records.append(SERSMeepExecutionRecord(
                request_id=request.request_id,
                source_numerical_plan_id=request.source_numerical_plan_id,
                source_solver_job_id=request.source_solver_job_id,
                hypothesis_id=request.hypothesis_id,
                geometry_candidate_id=request.geometry_candidate_id,
                execution_status=status,
                command=cmd,
                return_code=completed.returncode,
                wall_seconds=wall,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                worker_result_path=str(result_path) if result_path.exists() else None,
                worker_result_sha256=result_sha,
                error_message=error_message,
                numerical_resolution_status=request.resolution_status,
                evidence_eligibility=request.evidence_eligibility,
            ))
        except subprocess.TimeoutExpired:
            wall = time.monotonic() - started
            any_failure = True
            records.append(SERSMeepExecutionRecord(
                request_id=request.request_id,
                source_numerical_plan_id=request.source_numerical_plan_id,
                source_solver_job_id=request.source_solver_job_id,
                hypothesis_id=request.hypothesis_id,
                geometry_candidate_id=request.geometry_candidate_id,
                execution_status="timed_out",
                command=cmd,
                wall_seconds=wall,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error_message=f"worker exceeded timeout {args.timeout_seconds:g} s",
                numerical_resolution_status=request.resolution_status,
                evidence_eligibility=request.evidence_eligibility,
            ))

    records_bundle = SERSMeepExecutionRecordBundle(
        source_request_bundle_id=requests.bundle_id,
        records=records,
        record_count=len(records),
    )
    records_path = output_dir / "sers_fdtd.meep_execution_records.shadow.json"
    _write_json(records_path, records_bundle)

    status_counts = Counter(row.execution_status for row in records)
    manifest = {
        "schema_version": "sers-fdtd-meep-execution-shadow-manifest-v0",
        "source_solver_jobs": str(solver_path),
        "source_solver_jobs_sha256": _sha256_file(solver_path),
        "source_numerical_plans": str(numerical_path),
        "source_numerical_plans_sha256": _sha256_file(numerical_path),
        "request_bundle": str(request_bundle_path),
        "request_bundle_sha256": _sha256_file(request_bundle_path),
        "execution_records": str(records_path),
        "execution_records_sha256": _sha256_file(records_path),
        "worker_script": str(worker),
        "worker_script_sha256": _sha256_file(worker),
        "request_count": requests.request_count,
        "execution_status_counts": dict(sorted(status_counts.items())),
        "execute_requested": bool(args.execute),
        "validate_worker_only": bool(args.validate_worker_only),
        "shadow_only": True,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    manifest_path = output_dir / "sers_fdtd.meep_execution_shadow.manifest.json"
    _write_json(manifest_path, manifest)

    print(f"requests={request_bundle_path}")
    print(f"records={records_path}")
    print(f"manifest={manifest_path}")
    print(f"request_count={requests.request_count}")
    print("execution_status_counts=" + json.dumps(dict(sorted(status_counts.items())), sort_keys=True))
    print("physics_authority_created=false")

    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
