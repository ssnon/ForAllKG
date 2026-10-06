#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.candidate_contracts import CandidateDecisionPortfolio

from domains.sers.feasibility_lifecycle_projection import (
    SERSCanonicalLifecycleProjector,
)
from domains.sers.integrated_experiment_cycle_contracts import (
    SERSIntegratedExperimentFollowupBundle,
    SERSIntegratedExperimentReviewPlanBundle,
)
from domains.sers.validation_review_contracts import SERSValidationReviewBundle


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
            "Project SERS multi-validator lifecycle state beside canonical feasibility "
            "candidate decisions without creating rejection, promotion, feedback, or "
            "canonical graph authority."
        )
    )
    parser.add_argument("--candidate-decisions", required=True)
    parser.add_argument("--scientific-review", required=True)
    parser.add_argument("--experiment-plans")
    parser.add_argument("--followups")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    decisions_path = Path(args.candidate_decisions).expanduser().resolve()
    review_path = Path(args.scientific_review).expanduser().resolve()
    plans_path = (
        Path(args.experiment_plans).expanduser().resolve()
        if args.experiment_plans
        else None
    )
    followups_path = (
        Path(args.followups).expanduser().resolve()
        if args.followups
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()

    decisions = CandidateDecisionPortfolio.model_validate_json(
        decisions_path.read_text(encoding="utf-8")
    )
    review = SERSValidationReviewBundle.model_validate_json(
        review_path.read_text(encoding="utf-8")
    )
    plans = (
        SERSIntegratedExperimentReviewPlanBundle.model_validate_json(
            plans_path.read_text(encoding="utf-8")
        )
        if plans_path is not None
        else None
    )
    followups = (
        SERSIntegratedExperimentFollowupBundle.model_validate_json(
            followups_path.read_text(encoding="utf-8")
        )
        if followups_path is not None
        else None
    )

    projector = SERSCanonicalLifecycleProjector()
    result = projector.project(
        decisions,
        review,
        experiment_plans=plans,
        followups=followups,
    )

    result_path = output_dir / "sers.canonical_lifecycle_projection.shadow.json"
    manifest_path = output_dir / "sers.canonical_lifecycle_projection_shadow.manifest.json"
    _write_json(result_path, result)

    canonical_counts = Counter(
        row.canonical_final_disposition for row in result.projections
    )
    research_counts = Counter(row.sers_research_decision for row in result.projections)
    conflict_count = sum(row.authority_conflict_detected for row in result.projections)

    sources = {
        "candidate_decisions": str(decisions_path),
        "candidate_decisions_sha256": _sha256_file(decisions_path),
        "scientific_review": str(review_path),
        "scientific_review_sha256": _sha256_file(review_path),
    }
    if plans_path is not None:
        sources.update(
            {
                "experiment_plans": str(plans_path),
                "experiment_plans_sha256": _sha256_file(plans_path),
            }
        )
    if followups_path is not None:
        sources.update(
            {
                "followups": str(followups_path),
                "followups_sha256": _sha256_file(followups_path),
            }
        )

    manifest = {
        "schema_version": "sers-canonical-lifecycle-projection-shadow-manifest-v0",
        "projection_version": projector.projection_version,
        "sources": sources,
        "projection_count": result.projection_count,
        "canonical_final_disposition_counts": dict(sorted(canonical_counts.items())),
        "sers_research_decision_counts": dict(sorted(research_counts.items())),
        "authority_conflict_count": conflict_count,
        "interpretation": (
            "This artifact joins canonical feasibility lineage with the richer SERS "
            "multi-validator research lifecycle. It never overrides canonical candidate "
            "decisions and creates no rejection, promotion, feedback, or graph authority."
        ),
        "shadow_only": True,
        "canonical_authority_changed": False,
        "canonical_feedback_permitted": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"projection={result_path}")
    print(f"manifest={manifest_path}")
    print(f"projection_count={result.projection_count}")
    print(
        "canonical_final_disposition_counts="
        + json.dumps(dict(sorted(canonical_counts.items())), sort_keys=True)
    )
    print(
        "sers_research_decision_counts="
        + json.dumps(dict(sorted(research_counts.items())), sort_keys=True)
    )
    print(f"authority_conflict_count={conflict_count}")
    print("canonical_authority_changed=false")
    print("canonical_feedback_permitted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
