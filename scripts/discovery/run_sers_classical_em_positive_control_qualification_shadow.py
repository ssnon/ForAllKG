#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from domains.sers.classical_em_positive_control import (
    SERSClassicalEMPositiveControlQualifier,
)
from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlReferenceBundle,
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
            "Fail-closed qualification of literature/experimental SERS classical-EM "
            "positive controls against the currently implemented Meep model. No "
            "simulation or model-validation verdict is created."
        )
    )
    parser.add_argument("--references", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_path = Path(args.references).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    references = SERSEMPositiveControlReferenceBundle.model_validate_json(
        source_path.read_text(encoding="utf-8")
    )
    qualifier = SERSClassicalEMPositiveControlQualifier()
    result = qualifier.qualify(references)

    result_path = output_dir / "sers.classical_em_positive_control_qualification.shadow.json"
    manifest_path = output_dir / "sers.classical_em_positive_control_qualification_shadow.manifest.json"
    _write_json(result_path, result)

    status_counts = Counter(row.status for row in result.qualifications)
    blocker_counts = Counter(
        blocker
        for row in result.qualifications
        for blocker in row.blockers
    )
    manifest = {
        "schema_version": "sers-em-positive-control-qualification-shadow-manifest-v0",
        "source_references": str(source_path),
        "source_references_sha256": _sha256_file(source_path),
        "qualification_bundle": str(result_path),
        "qualification_bundle_sha256": _sha256_file(result_path),
        "backend_profile": qualifier.backend_profile,
        "qualification_count": result.qualification_count,
        "status_counts": dict(sorted(status_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "interpretation": (
            "Qualification means only that the reference measurement is representable "
            "by the current backend. It is not a model-validation pass and creates no "
            "authority over the target hypothesis."
        ),
        "shadow_only": True,
        "evidence_authority_created": False,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"qualification={result_path}")
    print(f"manifest={manifest_path}")
    print(f"qualification_count={result.qualification_count}")
    print("status_counts=" + json.dumps(dict(sorted(status_counts.items())), sort_keys=True))
    print("blocker_counts=" + json.dumps(dict(sorted(blocker_counts.items())), sort_keys=True))
    print("model_validation_permitted=false")
    print("physics_authority_created=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
