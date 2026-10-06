#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from domains.sers.classical_em_evidence_contracts import (
    SERSClassicalEMPhysicsEvidenceBundle,
)
from domains.sers.classical_em_readiness import SERSClassicalEMReadinessGate
from domains.sers.validation_design_contracts import SERSValidationDesignBundle
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlanBundle,
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
            "Assess routed SERS classical-EM evidence for numerical, model-form, "
            "and observable-scope readiness before any scientific component verdict."
        )
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--validation-plans", required=True)
    parser.add_argument("--validation-designs", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    evidence_path = Path(args.evidence).expanduser().resolve()
    plans_path = Path(args.validation_plans).expanduser().resolve()
    designs_path = Path(args.validation_designs).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    evidence = SERSClassicalEMPhysicsEvidenceBundle.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    plans = SERSHypothesisValidationPlanBundle.model_validate_json(
        plans_path.read_text(encoding="utf-8")
    )
    designs = SERSValidationDesignBundle.model_validate_json(
        designs_path.read_text(encoding="utf-8")
    )

    gate = SERSClassicalEMReadinessGate()
    result = gate.assess(evidence, plans, designs)

    result_path = output_dir / "sers.classical_em_readiness.shadow.json"
    manifest_path = output_dir / "sers.classical_em_readiness_shadow.manifest.json"
    _write_json(result_path, result)

    readiness_counts = Counter(row.overall_readiness for row in result.assessments)
    blocker_counts = Counter(
        blocker
        for row in result.assessments
        for blocker in row.readiness_blockers
    )
    scope_counts = Counter(row.evidence_scope_status for row in result.assessments)

    manifest = {
        "schema_version": "sers-classical-em-readiness-shadow-manifest-v0",
        "gate_version": gate.gate_version,
        "sources": {
            "evidence": str(evidence_path),
            "evidence_sha256": _sha256_file(evidence_path),
            "validation_plans": str(plans_path),
            "validation_plans_sha256": _sha256_file(plans_path),
            "validation_designs": str(designs_path),
            "validation_designs_sha256": _sha256_file(designs_path),
        },
        "assessment_count": result.assessment_count,
        "overall_readiness_counts": dict(sorted(readiness_counts.items())),
        "readiness_blocker_counts": dict(sorted(blocker_counts.items())),
        "evidence_scope_counts": dict(sorted(scope_counts.items())),
        "interpretation": (
            "This gate evaluates evidence readiness only. Operator-concretized "
            "model choices and absent positive-control validation remain explicit; "
            "no mechanism support, hypothesis rejection, experiment promotion, or "
            "self-feedback authority is created."
        ),
        "shadow_only": True,
        "evidence_authority_created": False,
        "physics_authority_created": False,
        "hypothesis_rejection_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"readiness={result_path}")
    print(f"manifest={manifest_path}")
    print(f"assessment_count={result.assessment_count}")
    print(
        "overall_readiness_counts="
        + json.dumps(dict(sorted(readiness_counts.items())), sort_keys=True)
    )
    print(
        "readiness_blocker_counts="
        + json.dumps(dict(sorted(blocker_counts.items())), sort_keys=True)
    )
    print(
        "evidence_scope_counts="
        + json.dumps(dict(sorted(scope_counts.items())), sort_keys=True)
    )
    print("component_review_permitted=false")
    print("experiment_promotion_permitted=false")
    print("feedback_generation_authority=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
