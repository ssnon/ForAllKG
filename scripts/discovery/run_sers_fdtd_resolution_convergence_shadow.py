#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from domains.sers.resonance_calibration_contracts import SERSResonanceCalibrationBundle
from domains.sers.resolution_convergence import SERSResolutionConvergenceEvaluator


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
            "Compare SERS resonance-calibration observations across resolutions. "
            "Reports raw peak drift and boundary censoring only; no convergence "
            "acceptance threshold or physics authority is created."
        )
    )
    parser.add_argument(
        "--calibration",
        action="append",
        required=True,
        help="S2.3 resonance-calibration observation bundle; repeat for resolutions.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = []
    sources = []
    for item in args.calibration:
        path = Path(item).resolve()
        bundle = SERSResonanceCalibrationBundle.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        observations.extend(bundle.observations)
        sources.append({
            "path": str(path),
            "sha256": _sha256_file(path),
            "observation_count": bundle.observation_count,
        })

    result = SERSResolutionConvergenceEvaluator().evaluate(observations)
    result_path = output_dir / "sers_fdtd.resolution_convergence_observations.shadow.json"
    _write_json(result_path, result)

    status_counts = Counter(row.overall_status for row in result.observations)
    manifest = {
        "schema_version": "sers-fdtd-resolution-convergence-shadow-manifest-v0",
        "source_calibration_bundles": sources,
        "convergence_bundle": str(result_path),
        "convergence_bundle_sha256": _sha256_file(result_path),
        "source_calibration_observation_count": result.source_calibration_observation_count,
        "convergence_group_count": result.observation_count,
        "overall_status_counts": dict(sorted(status_counts.items())),
        "assessment_scope": "drift_report_only_no_convergence_threshold",
        "shadow_only": True,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    manifest_path = output_dir / "sers_fdtd.resolution_convergence_shadow.manifest.json"
    _write_json(manifest_path, manifest)

    print(f"observations={result_path}")
    print(f"manifest={manifest_path}")
    print(f"source_calibration_observation_count={result.source_calibration_observation_count}")
    print(f"convergence_group_count={result.observation_count}")
    print("overall_status_counts=" + json.dumps(dict(sorted(status_counts.items())), sort_keys=True))
    print("convergence_claim_permitted=false")
    print("physics_authority_created=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
