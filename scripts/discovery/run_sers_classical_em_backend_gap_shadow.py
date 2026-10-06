#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from domains.sers.classical_em_backend_gap import SERSClassicalEMBackendGapAssessor
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
            "Aggregate SERS classical-EM positive-control qualification gaps, "
            "distinguishing within-source repetition from cross-source evidence. "
            "No backend extension or model-validation authority is created."
        )
    )
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path = Path(args.qualification).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    qualifications = SERSEMPositiveControlQualificationBundle.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    assessor = SERSClassicalEMBackendGapAssessor()
    result = assessor.assess(qualifications)

    result_path = output_dir / "sers.classical_em_backend_gap_assessment.shadow.json"
    manifest_path = output_dir / "sers.classical_em_backend_gap_assessment_shadow.manifest.json"
    _write_json(result_path, result)

    priority_counts = Counter(row.priority_status for row in result.gap_assessments)
    manifest = {
        "schema_version": "sers-em-backend-gap-assessment-shadow-manifest-v0",
        "source_qualification": str(source_path),
        "source_qualification_sha256": _sha256_file(source_path),
        "assessment_bundle": str(result_path),
        "assessment_bundle_sha256": _sha256_file(result_path),
        "backend_profile": result.backend_profile,
        "control_count": result.control_count,
        "independent_source_count": result.independent_source_count,
        "gap_count": len(result.gap_assessments),
        "gap_priority_counts": dict(sorted(priority_counts.items())),
        "extension_decision": result.extension_decision,
        "interpretation": (
            "Recurring mismatches from multiple controls in one source are not "
            "treated as cross-source evidence. This stage identifies candidates "
            "for later design review only and never authorizes a backend change."
        ),
        "shadow_only": True,
        "backend_extension_authorized": False,
        "model_validation_permitted": False,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"assessment={result_path}")
    print(f"manifest={manifest_path}")
    print(f"control_count={result.control_count}")
    print(f"independent_source_count={result.independent_source_count}")
    print(f"gap_count={len(result.gap_assessments)}")
    print("gap_priority_counts=" + json.dumps(dict(sorted(priority_counts.items())), sort_keys=True))
    print(f"extension_decision={result.extension_decision}")
    print("backend_extension_authorized=false")
    print("model_validation_permitted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
