#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from domains.sers.validation_evidence_intake import SERSRouteEvidenceIntakeCompiler
from domains.sers.validation_evidence_intake_contracts import (
    SERSRouteEvidenceSubmissionBundle,
)
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
            "Resolve curated/validator SERS evidence submissions onto routed "
            "non-EM validation claims with explicit independence checks."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--validation-plans", required=True)
    parser.add_argument("--submissions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    portfolio_path = Path(args.portfolio).expanduser().resolve()
    plans_path = Path(args.validation_plans).expanduser().resolve()
    submissions_path = Path(args.submissions).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    plans = SERSHypothesisValidationPlanBundle.model_validate_json(
        plans_path.read_text(encoding="utf-8")
    )
    submissions = SERSRouteEvidenceSubmissionBundle.model_validate_json(
        submissions_path.read_text(encoding="utf-8")
    )

    compiler = SERSRouteEvidenceIntakeCompiler()
    result = compiler.compile(portfolio, plans, submissions)

    result_path = output_dir / "sers.route_evidence.shadow.json"
    manifest_path = output_dir / "sers.route_evidence_shadow.manifest.json"
    _write_json(result_path, result)

    route_counts = Counter(row.route_kind for row in result.records)
    relation_counts = Counter(row.relation_to_claim for row in result.records)
    role_counts = Counter(row.evidence_role for row in result.records)
    manifest = {
        "schema_version": "sers-route-evidence-intake-shadow-manifest-v0",
        "compiler_version": compiler.compiler_version,
        "sources": {
            "portfolio": str(portfolio_path),
            "portfolio_sha256": _sha256_file(portfolio_path),
            "validation_plans": str(plans_path),
            "validation_plans_sha256": _sha256_file(plans_path),
            "submissions": str(submissions_path),
            "submissions_sha256": _sha256_file(submissions_path),
        },
        "record_count": result.record_count,
        "route_counts": dict(sorted(route_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "shadow_only": True,
        "evidence_authority_created": False,
        "hypothesis_rejection_authority": False,
        "experiment_promotion_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"route_evidence={result_path}")
    print(f"manifest={manifest_path}")
    print(f"record_count={result.record_count}")
    print("route_counts=" + json.dumps(dict(sorted(route_counts.items())), sort_keys=True))
    print("relation_counts=" + json.dumps(dict(sorted(relation_counts.items())), sort_keys=True))
    print("evidence_authority_created=false")
    print("hypothesis_rejection_authority=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
