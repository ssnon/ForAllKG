#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from domains.sers.classical_em_backend_gap_contracts import (
    SERSEMBackendGapAssessmentBundle,
)
from domains.sers.classical_em_extension_design import (
    SERSClassicalEMJointExtensionDesigner,
)
from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlQualificationBundle,
)


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
            "Design a shadow-only Pareto frontier of joint classical-EM backend "
            "extensions capable of unlocking already-qualified positive controls. "
            "This stage does not authorize implementation or execution."
        )
    )
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--backend-gap", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    qualification_path = Path(args.qualification).expanduser().resolve()
    gap_path = Path(args.backend_gap).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qualifications = SERSEMPositiveControlQualificationBundle.model_validate_json(
        qualification_path.read_text(encoding="utf-8")
    )
    gaps = SERSEMBackendGapAssessmentBundle.model_validate_json(
        gap_path.read_text(encoding="utf-8")
    )
    designer = SERSClassicalEMJointExtensionDesigner()
    result = designer.design(qualifications, gaps)

    result_path = output_dir / "sers.classical_em_joint_extension_design.shadow.json"
    manifest_path = output_dir / "sers.classical_em_joint_extension_design_shadow.manifest.json"
    _write_json(result_path, result)

    manifest = {
        "schema_version": "sers-em-joint-extension-design-shadow-manifest-v0",
        "source_qualification": str(qualification_path),
        "source_qualification_sha256": _sha256_file(qualification_path),
        "source_backend_gap": str(gap_path),
        "source_backend_gap_sha256": _sha256_file(gap_path),
        "design_bundle": str(result_path),
        "design_bundle_sha256": _sha256_file(result_path),
        "backend_profile": result.backend_profile,
        "candidate_count": result.candidate_count,
        "minimum_axis_count_for_any_unlock": result.minimum_axis_count_for_any_unlock,
        "minimum_axis_count_for_cross_source_unlock": result.minimum_axis_count_for_cross_source_unlock,
        "minimum_axis_count_for_full_coverage": result.minimum_axis_count_for_full_coverage,
        "design_decision": result.design_decision,
        "interpretation": (
            "Candidates are design alternatives only. A smaller extension can unlock "
            "one control without providing cross-source model validation. No candidate "
            "authorizes backend implementation, positive-control execution, or a "
            "scientific verdict."
        ),
        "shadow_only": True,
        "backend_extension_authorized": False,
        "positive_control_execution_authorized": False,
        "model_validation_permitted": False,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"design={result_path}")
    print(f"manifest={manifest_path}")
    print(f"candidate_count={result.candidate_count}")
    print(f"minimum_axis_count_for_any_unlock={result.minimum_axis_count_for_any_unlock}")
    print(
        "minimum_axis_count_for_cross_source_unlock="
        f"{result.minimum_axis_count_for_cross_source_unlock}"
    )
    print(f"minimum_axis_count_for_full_coverage={result.minimum_axis_count_for_full_coverage}")
    print(f"design_decision={result.design_decision}")
    print("backend_extension_authorized=false")
    print("model_validation_permitted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
