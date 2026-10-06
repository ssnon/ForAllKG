#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from domains.sers.classical_em_readiness_contracts import SERSClassicalEMReadinessBundle
from domains.sers.integrated_experiment_cycle import (
    SERSIntegratedExperimentFollowupAnalyzer,
    SERSIntegratedExperimentResultCompiler,
)
from domains.sers.integrated_experiment_cycle_contracts import (
    SERSIntegratedExperimentResultSubmissionBundle,
    SERSIntegratedExperimentReviewPlanBundle,
)
from domains.sers.validation_orchestration import SERSValidationOrchestrator
from domains.sers.validation_orchestration_contracts import SERSRouteEvidenceBundle
from domains.sers.validation_review import SERSMultiValidatorScientificReviewer
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
            "Ingest reviewed integrated SERS results, merge them with existing routed "
            "evidence, rerun S4.0/S4.1, and emit shadow failure-attribution / ForAllKG "
            "feedback candidates in one result-cycle command."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--validation-plans", required=True)
    parser.add_argument("--classical-em-readiness", required=True)
    parser.add_argument("--experiment-plans", required=True)
    parser.add_argument("--result-submissions", required=True)
    parser.add_argument("--existing-route-evidence")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    portfolio_path = Path(args.portfolio).expanduser().resolve()
    plans_path = Path(args.validation_plans).expanduser().resolve()
    readiness_path = Path(args.classical_em_readiness).expanduser().resolve()
    experiment_plans_path = Path(args.experiment_plans).expanduser().resolve()
    submissions_path = Path(args.result_submissions).expanduser().resolve()
    existing_evidence_path = (
        Path(args.existing_route_evidence).expanduser().resolve()
        if args.existing_route_evidence
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()

    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    plans = SERSHypothesisValidationPlanBundle.model_validate_json(
        plans_path.read_text(encoding="utf-8")
    )
    readiness = SERSClassicalEMReadinessBundle.model_validate_json(
        readiness_path.read_text(encoding="utf-8")
    )
    experiment_plans = SERSIntegratedExperimentReviewPlanBundle.model_validate_json(
        experiment_plans_path.read_text(encoding="utf-8")
    )
    submissions = SERSIntegratedExperimentResultSubmissionBundle.model_validate_json(
        submissions_path.read_text(encoding="utf-8")
    )
    existing_evidence = (
        SERSRouteEvidenceBundle.model_validate_json(
            existing_evidence_path.read_text(encoding="utf-8")
        )
        if existing_evidence_path is not None
        else None
    )

    compiler = SERSIntegratedExperimentResultCompiler()
    result_bundle, merged_evidence = compiler.compile(
        experiment_plans,
        submissions,
        existing_route_evidence=existing_evidence,
    )

    orchestration = SERSValidationOrchestrator().build(
        portfolio,
        plans,
        classical_em_readiness=readiness,
        route_evidence=merged_evidence,
    )
    updated_review = SERSMultiValidatorScientificReviewer().review(
        portfolio,
        plans,
        orchestration,
        route_evidence=merged_evidence,
    )
    followup = SERSIntegratedExperimentFollowupAnalyzer().analyze(
        updated_review,
        result_bundle,
    )

    results_path = output_dir / "sers.integrated_experiment_results.shadow.json"
    evidence_path = output_dir / "sers.route_evidence.merged.shadow.json"
    orchestration_path = output_dir / "sers.validation_orchestration.updated.shadow.json"
    review_path = output_dir / "sers.validation_review.updated.shadow.json"
    followup_path = output_dir / "sers.integrated_experiment_followup.shadow.json"
    manifest_path = output_dir / "sers.integrated_experiment_result_cycle_shadow.manifest.json"
    _write_json(results_path, result_bundle)
    _write_json(evidence_path, merged_evidence)
    _write_json(orchestration_path, orchestration)
    _write_json(review_path, updated_review)
    _write_json(followup_path, followup)

    relation_counts = Counter(row.relation_to_claim for row in result_bundle.results)
    outcome_counts = Counter(row.outcome_verdict for row in followup.followups)
    attribution_counts = Counter(
        row.category
        for follow in followup.followups
        for row in follow.attribution_candidates
    )
    feedback_counts = Counter(
        row.feedback_kind
        for follow in followup.followups
        for row in follow.feedback_candidates
    )

    sources = {
        "portfolio": str(portfolio_path),
        "portfolio_sha256": _sha256_file(portfolio_path),
        "validation_plans": str(plans_path),
        "validation_plans_sha256": _sha256_file(plans_path),
        "classical_em_readiness": str(readiness_path),
        "classical_em_readiness_sha256": _sha256_file(readiness_path),
        "experiment_plans": str(experiment_plans_path),
        "experiment_plans_sha256": _sha256_file(experiment_plans_path),
        "result_submissions": str(submissions_path),
        "result_submissions_sha256": _sha256_file(submissions_path),
    }
    if existing_evidence_path is not None:
        sources["existing_route_evidence"] = str(existing_evidence_path)
        sources["existing_route_evidence_sha256"] = _sha256_file(existing_evidence_path)

    manifest = {
        "schema_version": "sers-integrated-experiment-result-cycle-shadow-manifest-v0",
        "result_compiler_version": compiler.compiler_version,
        "followup_analyzer_version": SERSIntegratedExperimentFollowupAnalyzer.analyzer_version,
        "sources": sources,
        "result_count": result_bundle.result_count,
        "relation_counts": dict(sorted(relation_counts.items())),
        "updated_route_evidence_count": merged_evidence.record_count,
        "updated_review_count": updated_review.review_count,
        "followup_count": followup.followup_count,
        "outcome_verdict_counts": dict(sorted(outcome_counts.items())),
        "failure_attribution_counts": dict(sorted(attribution_counts.items())),
        "feedback_candidate_counts": dict(sorted(feedback_counts.items())),
        "interpretation": (
            "Experimental results are normalized, routed, re-reviewed, and converted "
            "into non-causal failure-attribution / research-feedback candidates. No "
            "whole-hypothesis confirmation, rejection, rewrite, or canonical feedback authority is created."
        ),
        "shadow_only": True,
        "whole_hypothesis_confirmation_permitted": False,
        "automatic_hypothesis_rejection_permitted": False,
        "canonical_feedback_permitted": False,
        "feedback_generation_authority": False,
        "canonical_graph_mutated": False,
    }
    _write_json(manifest_path, manifest)

    print(f"results={results_path}")
    print(f"merged_route_evidence={evidence_path}")
    print(f"updated_orchestration={orchestration_path}")
    print(f"updated_review={review_path}")
    print(f"followup={followup_path}")
    print(f"manifest={manifest_path}")
    print(f"result_count={result_bundle.result_count}")
    print("relation_counts=" + json.dumps(dict(sorted(relation_counts.items()))))
    print("outcome_verdict_counts=" + json.dumps(dict(sorted(outcome_counts.items()))))
    print("failure_attribution_counts=" + json.dumps(dict(sorted(attribution_counts.items()))))
    print("feedback_candidate_counts=" + json.dumps(dict(sorted(feedback_counts.items()))))
    print("canonical_feedback_permitted=false")
    print("automatic_hypothesis_rejection_permitted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
