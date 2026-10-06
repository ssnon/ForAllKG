#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from domains.sers.integrated_experiment_cycle import SERSIntegratedExperimentReviewPlanner
from domains.sers.validation_orchestration_contracts import SERSValidationOrchestrationBundle
from domains.sers.validation_review_contracts import SERSValidationReviewBundle
from domains.sers.validation_routing_contracts import SERSHypothesisValidationPlanBundle


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
            "Build a shadow-only, non-procedural integrated SERS experiment-review "
            "plan from S4.0 orchestration and S4.1 scientific review."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--validation-plans", required=True)
    parser.add_argument("--orchestration", required=True)
    parser.add_argument("--scientific-review", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    portfolio_path = Path(args.portfolio).expanduser().resolve()
    plans_path = Path(args.validation_plans).expanduser().resolve()
    orchestration_path = Path(args.orchestration).expanduser().resolve()
    review_path = Path(args.scientific_review).expanduser().resolve()
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
    review = SERSValidationReviewBundle.model_validate_json(
        review_path.read_text(encoding="utf-8")
    )

    planner = SERSIntegratedExperimentReviewPlanner()
    result = planner.plan(portfolio, plans, orchestration, review)

    result_path = output_dir / "sers.integrated_experiment_review_plans.shadow.json"
    manifest_path = output_dir / "sers.integrated_experiment_review_plans_shadow.manifest.json"
    _write_json(result_path, result)

    status_counts = Counter(row.plan_status for row in result.plans)
    priority_counts = Counter(row.information_priority for row in result.plans)
    manifest = {
        "schema_version": "sers-integrated-experiment-review-plan-shadow-manifest-v0",
        "planner_version": planner.planner_version,
        "sources": {
            "portfolio": str(portfolio_path),
            "portfolio_sha256": _sha256_file(portfolio_path),
            "validation_plans": str(plans_path),
            "validation_plans_sha256": _sha256_file(plans_path),
            "orchestration": str(orchestration_path),
            "orchestration_sha256": _sha256_file(orchestration_path),
            "scientific_review": str(review_path),
            "scientific_review_sha256": _sha256_file(review_path),
        },
        "plan_count": result.plan_count,
        "plan_status_counts": dict(sorted(status_counts.items())),
        "information_priority_counts": dict(sorted(priority_counts.items())),
        "interpretation": (
            "This artifact defines what a reviewed integrated comparison must resolve. "
            "It is not a fabrication recipe or executable laboratory protocol."
        ),
        "shadow_only": True,
        "protocol_generation_permitted": False,
        "experiment_execution_authority": False,
        "experiment_promotion_authority": False,
        "hypothesis_verdict_authority": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"experiment_plans={result_path}")
    print(f"manifest={manifest_path}")
    print(f"plan_count={result.plan_count}")
    print("plan_status_counts=" + json.dumps(dict(sorted(status_counts.items()))))
    print("information_priority_counts=" + json.dumps(dict(sorted(priority_counts.items()))))
    print("protocol_generation_permitted=false")
    print("experiment_execution_authority=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
