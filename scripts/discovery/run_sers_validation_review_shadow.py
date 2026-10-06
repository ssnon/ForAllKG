#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from domains.sers.validation_orchestration_contracts import (
    SERSRouteEvidenceBundle,
    SERSValidationOrchestrationBundle,
)
from domains.sers.validation_review import SERSMultiValidatorScientificReviewer
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
            "Perform provisional multi-validator SERS scientific review and "
            "claim-scoped research-decision planning without canonical authority."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--validation-plans", required=True)
    parser.add_argument("--orchestration", required=True)
    parser.add_argument("--route-evidence")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    portfolio_path = Path(args.portfolio).expanduser().resolve()
    plans_path = Path(args.validation_plans).expanduser().resolve()
    orchestration_path = Path(args.orchestration).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    plans = SERSHypothesisValidationPlanBundle.model_validate_json(
        plans_path.read_text(encoding="utf-8")
    )
    orchestration = SERSValidationOrchestrationBundle.model_validate_json(
        orchestration_path.read_text(encoding="utf-8")
    )
    route_evidence = None
    evidence_path = None
    if args.route_evidence:
        evidence_path = Path(args.route_evidence).expanduser().resolve()
        route_evidence = SERSRouteEvidenceBundle.model_validate_json(
            evidence_path.read_text(encoding="utf-8")
        )

    reviewer = SERSMultiValidatorScientificReviewer()
    result = reviewer.review(
        portfolio,
        plans,
        orchestration,
        route_evidence=route_evidence,
    )

    result_path = output_dir / "sers.validation_review.shadow.json"
    manifest_path = output_dir / "sers.validation_review_shadow.manifest.json"
    _write_json(result_path, result)

    decision_counts = Counter(row.research_decision for row in result.reviews)
    experiment_counts = Counter(row.experiment_review_state for row in result.reviews)
    route_verdict_counts = Counter(
        assessment.verdict
        for review in result.reviews
        for assessment in review.route_assessments
    )
    manifest = {
        "schema_version": "sers-validation-review-shadow-manifest-v0",
        "reviewer_version": reviewer.reviewer_version,
        "sources": {
            "portfolio": str(portfolio_path),
            "portfolio_sha256": _sha256_file(portfolio_path),
            "validation_plans": str(plans_path),
            "validation_plans_sha256": _sha256_file(plans_path),
            "orchestration": str(orchestration_path),
            "orchestration_sha256": _sha256_file(orchestration_path),
            "route_evidence": str(evidence_path) if evidence_path else None,
            "route_evidence_sha256": _sha256_file(evidence_path) if evidence_path else None,
        },
        "review_count": result.review_count,
        "research_decision_counts": dict(sorted(decision_counts.items())),
        "experiment_review_state_counts": dict(sorted(experiment_counts.items())),
        "route_verdict_counts": dict(sorted(route_verdict_counts.items())),
        "interpretation": (
            "Scientific review is provisional and claim-scoped. Route verdicts "
            "do not create canonical rejection, experiment execution, or feedback authority."
        ),
        "shadow_only": True,
        "provisional_scientific_review_only": True,
        "scientific_verdict_authority_created": False,
        "hypothesis_rejection_authority": False,
        "experiment_promotion_authority": False,
        "experiment_execution_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"review={result_path}")
    print(f"manifest={manifest_path}")
    print(f"review_count={result.review_count}")
    print(
        "research_decision_counts="
        + json.dumps(dict(sorted(decision_counts.items())), sort_keys=True)
    )
    print(
        "experiment_review_state_counts="
        + json.dumps(dict(sorted(experiment_counts.items())), sort_keys=True)
    )
    print(
        "route_verdict_counts="
        + json.dumps(dict(sorted(route_verdict_counts.items())), sort_keys=True)
    )
    print("scientific_verdict_authority_created=false")
    print("experiment_promotion_authority=false")
    print("feedback_generation_authority=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
