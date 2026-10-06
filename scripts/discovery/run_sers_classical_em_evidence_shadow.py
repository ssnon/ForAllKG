#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from domains.sers.classical_em_evidence import SERSClassicalEMEvidenceAssembler
from domains.sers.classical_em_handoff_contracts import SERSClassicalEMHandoffBundle
from domains.sers.resonance_calibration_contracts import SERSResonanceCalibrationBundle
from domains.sers.resolution_convergence_contracts import SERSResolutionConvergenceBundle
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
            "Normalize routed SERS classical-EM/FDTD calibration and numerical-"
            "convergence observations into shadow PhysicsEvidence without issuing "
            "a hypothesis, experiment-promotion, or self-feedback verdict."
        )
    )
    parser.add_argument("--handoffs", required=True)
    parser.add_argument("--solver-jobs", required=True)
    parser.add_argument(
        "--calibration",
        action="append",
        required=True,
        help="Resonance-calibration bundle JSON. Repeat for multiple resolutions.",
    )
    parser.add_argument(
        "--convergence",
        action="append",
        default=[],
        help="Optional resolution-convergence bundle JSON. Repeat if needed.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    handoff_path = Path(args.handoffs).expanduser().resolve()
    solver_jobs_path = Path(args.solver_jobs).expanduser().resolve()
    calibration_paths = [Path(row).expanduser().resolve() for row in args.calibration]
    convergence_paths = [Path(row).expanduser().resolve() for row in args.convergence]
    output_dir = Path(args.output_dir).expanduser().resolve()

    handoffs = SERSClassicalEMHandoffBundle.model_validate_json(
        handoff_path.read_text(encoding="utf-8")
    )
    solver_jobs = SERSFDTDSolverJobBundle.model_validate_json(
        solver_jobs_path.read_text(encoding="utf-8")
    )
    calibrations = [
        SERSResonanceCalibrationBundle.model_validate_json(path.read_text(encoding="utf-8"))
        for path in calibration_paths
    ]
    convergences = [
        SERSResolutionConvergenceBundle.model_validate_json(path.read_text(encoding="utf-8"))
        for path in convergence_paths
    ]

    bundle = SERSClassicalEMEvidenceAssembler().assemble(
        handoffs,
        solver_jobs,
        calibrations,
        convergences,
    )

    evidence_path = output_dir / "sers.classical_em_physics_evidence.shadow.json"
    manifest_path = output_dir / "sers.classical_em_physics_evidence_shadow.manifest.json"
    _write_json(evidence_path, bundle)

    state_counts = Counter(row.numerical_evidence_state for row in bundle.evidence)
    blocker_counts = Counter(
        blocker
        for row in bundle.evidence
        for blocker in row.numerical_blockers
    )
    manifest = {
        "schema_version": "sers-classical-em-physics-evidence-shadow-manifest-v0",
        "source_handoffs": str(handoff_path),
        "source_handoffs_sha256": _sha256_file(handoff_path),
        "source_solver_jobs": str(solver_jobs_path),
        "source_solver_jobs_sha256": _sha256_file(solver_jobs_path),
        "source_calibrations": [str(path) for path in calibration_paths],
        "source_calibration_sha256": {
            str(path): _sha256_file(path) for path in calibration_paths
        },
        "source_convergences": [str(path) for path in convergence_paths],
        "source_convergence_sha256": {
            str(path): _sha256_file(path) for path in convergence_paths
        },
        "evidence_count": bundle.evidence_count,
        "unobserved_handoff_ids": bundle.unobserved_handoff_ids,
        "numerical_evidence_state_counts": dict(sorted(state_counts.items())),
        "numerical_blocker_counts": dict(sorted(blocker_counts.items())),
        "artifact": str(evidence_path),
        "interpretation": (
            "This artifact is routed classical-EM observational evidence only. "
            "Numerical blockers remain distinct from scientific contradiction; "
            "no support/reject, experiment-promotion, or feedback authority is created."
        ),
        "shadow_only": True,
        "evidence_authority_created": False,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"evidence={evidence_path}")
    print(f"manifest={manifest_path}")
    print(f"evidence_count={bundle.evidence_count}")
    print(f"unobserved_handoff_count={len(bundle.unobserved_handoff_ids)}")
    print(
        "numerical_evidence_state_counts="
        + json.dumps(dict(sorted(state_counts.items())), sort_keys=True)
    )
    print(
        "numerical_blocker_counts="
        + json.dumps(dict(sorted(blocker_counts.items())), sort_keys=True)
    )
    print("mechanism_support_verdict_permitted=false")
    print("experiment_promotion_permitted=false")
    print("feedback_generation_authority=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
