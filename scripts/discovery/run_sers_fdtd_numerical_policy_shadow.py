#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from domains.sers.numerical_policy import SERSFDTDNumericalPolicyPlanner
from domains.sers.numerical_policy_contracts import (
    SERSFDTDNumericalPolicyOverrideBundle,
)
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


def _template(request) -> dict:
    return {
        "schema_version": "sers-fdtd-numerical-policy-override-bundle-v0",
        "policies": [
            {
                "source_solver_job_bundle_id": request.source_solver_job_bundle_id,
                "policy_id": "EDIT_ME_policy_id",
                "source_label": "EDIT_ME_numerical_policy",
                "intended_use": "execution_sanity",
                "dimension": "3d",
                "incident_propagation": "normal_to_nanorod_axis",
                "resolution_px_per_um": None,
                "cells_per_min_feature_target": 8.0,
                "pml_thickness_um": None,
                "padding_um": None,
                "spectrum_frequency_points": None,
                "spectral_guard_band_nm": 0.0,
                "resource_benchmark": None,
                "operator_note": (
                    "Numerical policy only. Resolution/resource outcomes do not "
                    "support or reject the scientific hypothesis."
                ),
            }
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a shadow-only numerical-resolution/resource request for SERS "
            "FDTD solver jobs and optionally apply one explicit numerical policy. "
            "This slice does not execute Meep."
        )
    )
    parser.add_argument("--solver-jobs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-overrides")
    parser.add_argument("--policy-id")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path = Path(args.solver_jobs).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = SERSFDTDSolverJobBundle.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    planner = SERSFDTDNumericalPolicyPlanner()
    request = planner.request(jobs)

    request_path = output_dir / "sers_fdtd.numerical_policy_request.shadow.json"
    template_path = output_dir / "sers_fdtd.numerical_policy.template.json"
    manifest_path = output_dir / "sers_fdtd.numerical_policy_shadow.manifest.json"
    _write_json(request_path, request)
    _write_json(template_path, _template(request))

    manifest = {
        "schema_version": "sers-fdtd-numerical-policy-shadow-manifest-v0",
        "source_solver_jobs": str(source_path),
        "source_solver_jobs_sha256": _sha256_file(source_path),
        "request": str(request_path),
        "request_sha256": _sha256_file(request_path),
        "template": str(template_path),
        "template_sha256": _sha256_file(template_path),
        "solver_job_count": request.solver_job_count,
        "minimum_shell_thickness_nm": request.minimum_shell_thickness_nm,
        "target_resolution_for_minimum_shell_px_per_um": (
            request.target_resolution_for_minimum_shell_px_per_um
        ),
        "shadow_only": True,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }

    print(f"request={request_path}")
    print(f"template={template_path}")
    print(f"solver_job_count={request.solver_job_count}")
    print(f"minimum_shell_thickness_nm={request.minimum_shell_thickness_nm:g}")
    print(
        "target_resolution_for_minimum_shell_px_per_um="
        f"{request.target_resolution_for_minimum_shell_px_per_um:g}"
    )

    if args.policy_overrides:
        override_path = Path(args.policy_overrides).resolve()
        overrides = SERSFDTDNumericalPolicyOverrideBundle.model_validate_json(
            override_path.read_text(encoding="utf-8")
        )
        candidates = [
            row
            for row in overrides.policies
            if row.source_solver_job_bundle_id == jobs.bundle_id
        ]
        if args.policy_id:
            candidates = [row for row in candidates if row.policy_id == args.policy_id]
        if len(candidates) != 1:
            raise ValueError(
                "expected exactly one matching numerical policy override; "
                f"found {len(candidates)}"
            )

        plans = planner.apply(jobs, candidates[0])
        plans_path = output_dir / "sers_fdtd.numerical_plans.shadow.json"
        _write_json(plans_path, plans)

        resolution_counts = Counter(row.resolution_status for row in plans.plans)
        evidence_counts = Counter(row.evidence_eligibility for row in plans.plans)
        resource_counts = Counter(
            (
                row.resource_estimate.decision
                if row.resource_estimate is not None
                else "resource_unknown"
            )
            for row in plans.plans
        )
        manifest.update({
            "policy_overrides": str(override_path),
            "policy_overrides_sha256": _sha256_file(override_path),
            "policy_id": plans.policy_id,
            "numerical_plans": str(plans_path),
            "numerical_plans_sha256": _sha256_file(plans_path),
            "resolution_status_counts": dict(sorted(resolution_counts.items())),
            "evidence_eligibility_counts": dict(sorted(evidence_counts.items())),
            "resource_decision_counts": dict(sorted(resource_counts.items())),
        })
        print(f"numerical_plans={plans_path}")
        print(
            "resolution_status_counts="
            + json.dumps(dict(sorted(resolution_counts.items())), sort_keys=True)
        )
        print(
            "resource_decision_counts="
            + json.dumps(dict(sorted(resource_counts.items())), sort_keys=True)
        )

    _write_json(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    print("execution_status=numerically_assessed_not_executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
