from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_core.discovery.scientific_verifier_shadow_companion import (
    build_scientific_verifier_shadow_companion_plan,
    resolve_scientific_verifier_shadow_inputs,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run(name: str, argv: list[str], expected: list[Path]) -> None:
    print()
    print("==", name, "==")
    subprocess.run([sys.executable, *argv], check=True)
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"{name} completed without expected outputs: {missing}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attach the relation-aware 0065-0078 scientific verifier as a read-only "
            "shadow companion to a completed atomic scientific N10 E2E run. Historical "
            "calibration/debug runs may explicitly opt into exact-artifact smoke mode. "
            "Atomic hypotheses and external novelty are lineage-checked and reused, "
            "while relation retrieval/adjudication is materialized independently."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--candidate-portfolio", type=Path, default=None)
    parser.add_argument("--atomic-report", type=Path, default=None)
    parser.add_argument("--atomic-portfolio", type=Path, default=None)
    parser.add_argument("--atomic-n10-manifest", type=Path, default=None)
    parser.add_argument("--external-novelty", type=Path, default=None)
    parser.add_argument("--provider-plan", type=Path, default=None)
    parser.add_argument("--context", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL")
            or os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--support-results-per-query", type=int, default=12)
    parser.add_argument("--second-pass-results-per-query", type=int, default=16)
    parser.add_argument("--max-review-works-per-claim", type=int, default=20)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument(
        "--allow-legacy-atomic-n10-artifacts",
        action="store_true",
        help=(
            "Historical calibration/debug smoke only: allow an atomic N10 manifest "
            "whose status is not complete, while retaining exact candidate/atomic/"
            "external lineage checks. This does not satisfy the prospective validation "
            "input contract and must not be used to claim a completed atomic N10 run."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    inputs = resolve_scientific_verifier_shadow_inputs(
        run_dir=args.run_dir,
        context=args.context,
        candidate_portfolio=args.candidate_portfolio,
        atomic_report=args.atomic_report,
        atomic_portfolio=args.atomic_portfolio,
        atomic_n10_manifest=args.atomic_n10_manifest,
        atomic_n10_external_report=args.external_novelty,
        atomic_n10_provider_plan=args.provider_plan,
        allow_legacy_atomic_n10_artifacts=(
            args.allow_legacy_atomic_n10_artifacts
        ),
    )
    plan = build_scientific_verifier_shadow_companion_plan(
        inputs=inputs,
        output_dir=args.output_dir,
        allow_legacy_atomic_n10_artifacts=(
            args.allow_legacy_atomic_n10_artifacts
        ),
    )
    output = plan.outputs
    output_dir = Path(output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL, "
            "GRAPHAGENTS_HYPOTHESIS_MODEL, or OPENROUTER_AGENT_MODEL is set"
        )

    context = _load(Path(inputs.context))
    domain_profile_id = str(context["domain_profile_id"])
    atomic_portfolio_id = plan.lineage.atomic_portfolio_id

    annotation = Path(output.grounded_identity_annotation)
    relation_ir = Path(output.relation_ir)
    relation_projection = Path(output.relation_projection)
    factorization = Path(output.grounded_identity_factorization)
    grounded_projection = Path(output.grounded_factor_projection)
    supporting_prefix = Path(output.supporting_prefix)
    supporting_plan = supporting_prefix.with_suffix(".supporting_query_plan.json")
    supporting_prior = supporting_prefix.with_suffix(".prior_art.json")
    supporting_report = supporting_prefix.with_suffix(".retrieval_report.json")
    counter_prefix = Path(output.counterevidence_prefix)
    counter_plan = counter_prefix.with_suffix(".counterevidence_query_plan.json")
    counter_prior = counter_prefix.with_suffix(".prior_art.json")
    counter_report = counter_prefix.with_suffix(".retrieval_report.json")
    second_prefix = Path(output.second_pass_prefix)
    second_plan = second_prefix.with_suffix(".semantic_plan.json")
    second_resolved = second_prefix.with_suffix(".resolved_prior_art.json")
    second_report = second_prefix.with_suffix(".report.json")
    adjudication_prefix = Path(output.relation_adjudication_prefix)
    adjudication_candidates = adjudication_prefix.with_suffix(".candidates.json")
    adjudication_review = adjudication_prefix.with_suffix(".review.json")
    evidence_graph = Path(output.evidence_graph)
    centrality = Path(output.centrality)
    aggregation = Path(output.aggregation)
    positive_basis = Path(output.positive_basis)
    positive_prefix = Path(output.positive_adjudication_prefix)
    positive_review = positive_prefix.with_suffix(".review.json")
    certification = Path(output.certification_report)
    manifest_path = Path(output.manifest)

    model_args: list[str] = ["--model", args.model, "--api-key-env", args.api_key_env]
    if args.base_url:
        model_args += ["--base-url", args.base_url]

    grounded_cmd = [
        "-m", "scripts.discovery.run_grounded_identity_constituent_shadow",
        "--candidate-portfolio", inputs.candidate_portfolio,
        "--atomic-report", inputs.atomic_report,
        "--annotation-output", str(annotation),
        "--annotation-only",
        *model_args,
    ]
    if args.save_prompts:
        grounded_cmd += ["--prompt-output", str(output_dir / "grounded_identity.prompt.txt")]

    relation_ir_cmd = [
        "-m", "scripts.discovery.build_scientific_relation_ir_shadow",
        "--atomic-report", inputs.atomic_report,
        "--domain-profile", domain_profile_id,
        "--output", str(relation_ir),
    ]
    relation_projection_cmd = [
        "-m", "scripts.discovery.build_scientific_relation_projection_shadow",
        "--relation-ir", str(relation_ir),
        "--output", str(relation_projection),
    ]
    factorization_cmd = [
        "-m", "scripts.discovery.build_grounded_identity_factorization_shadow",
        "--relation-ir", str(relation_ir),
        "--annotation", str(annotation),
        "--output", str(factorization),
    ]
    grounded_projection_cmd = [
        "-m", "scripts.discovery.build_grounded_factor_projection_shadow",
        "--relation-ir", str(relation_ir),
        "--factorization", str(factorization),
        "--output", str(grounded_projection),
    ]
    supporting_cmd = [
        "-m", "scripts.discovery.run_supporting_projection_retrieval_shadow",
        "--portfolio", inputs.atomic_portfolio,
        "--projection", str(grounded_projection),
        "--provider-plan", inputs.atomic_n10_provider_plan,
        "--results-per-query", str(args.support_results_per_query),
        "--output-prefix", str(supporting_prefix),
    ]
    counter_cmd = [
        "-m", "scripts.discovery.run_counterevidence_projection_retrieval_shadow",
        "--portfolio", inputs.atomic_portfolio,
        "--projection", str(grounded_projection),
        "--provider-plan", inputs.atomic_n10_provider_plan,
        "--results-per-query", str(args.support_results_per_query),
        "--output-prefix", str(counter_prefix),
    ]
    second_cmd = [
        "-m", "scripts.discovery.run_scientific_relation_second_pass_shadow",
        "--relation-ir", str(relation_ir),
        "--projection", str(grounded_projection),
        "--domain-profile", domain_profile_id,
        "--source-portfolio-id", atomic_portfolio_id,
        "--provider-plan", inputs.atomic_n10_provider_plan,
        "--results-per-query", str(args.second_pass_results_per_query),
        "--output-prefix", str(second_prefix),
        *model_args,
    ]
    if args.save_prompts:
        second_cmd += ["--save-prompts"]

    adjudication_cmd = [
        "-m", "scripts.discovery.run_projection_relation_adjudication_shadow",
        "--relation-ir", str(relation_ir),
        "--projection", str(grounded_projection),
        "--supporting-plan", str(supporting_plan),
        "--supporting-prior-art", str(supporting_prior),
        "--counterevidence-plan", str(counter_plan),
        "--counterevidence-prior-art", str(counter_prior),
        "--second-pass-plan", str(second_plan),
        "--second-pass-prior-art", str(second_resolved),
        "--domain-profile", domain_profile_id,
        "--max-review-works-per-claim", str(args.max_review_works_per_claim),
        "--output-prefix", str(adjudication_prefix),
        *model_args,
    ]
    if args.save_prompts:
        adjudication_cmd += ["--save-prompts"]

    graph_cmd = [
        "-m", "scripts.discovery.build_scientific_claim_evidence_graph_shadow",
        "--relation-ir", str(relation_ir),
        "--projection", str(grounded_projection),
        "--candidates", str(adjudication_candidates),
        "--adjudication", str(adjudication_review),
        "--output", str(evidence_graph),
    ]
    centrality_cmd = [
        "-m", "scripts.discovery.run_scientific_claim_centrality_shadow",
        "--evidence-graph", str(evidence_graph),
        "--output", str(centrality),
    ]
    aggregation_cmd = [
        "-m", "scripts.discovery.run_scientific_hypothesis_evidence_aggregation_shadow",
        "--evidence-graph", str(evidence_graph),
        "--centrality", str(centrality),
        "--output", str(aggregation),
    ]
    basis_cmd = [
        "-m", "scripts.discovery.run_positive_nonobviousness_basis_shadow",
        "--evidence-graph", str(evidence_graph),
        "--aggregation", str(aggregation),
        "--output", str(positive_basis),
    ]
    positive_cmd = [
        "-m", "scripts.discovery.run_positive_nonobviousness_adjudication_shadow",
        "--evidence-graph", str(evidence_graph),
        "--basis", str(positive_basis),
        "--output-prefix", str(positive_prefix),
        *model_args,
    ]
    if args.save_prompts:
        positive_cmd += ["--save-prompts"]

    certification_cmd = [
        "-m", "scripts.discovery.run_scientific_certification_gate_shadow",
        "--aggregation", str(aggregation),
        "--nonobviousness", str(positive_review),
        "--external-novelty", inputs.atomic_n10_external_report,
        "--output", str(certification),
    ]

    stages: list[tuple[str, list[str], list[Path]]] = [
        ("grounded_identity_annotation", grounded_cmd, [annotation]),
        ("scientific_relation_ir", relation_ir_cmd, [relation_ir]),
        ("scientific_relation_projection", relation_projection_cmd, [relation_projection]),
        ("grounded_identity_factorization", factorization_cmd, [factorization]),
        ("grounded_factor_projection", grounded_projection_cmd, [grounded_projection]),
        ("supporting_projection_retrieval", supporting_cmd, [supporting_plan, supporting_prior, supporting_report]),
        ("counterevidence_projection_retrieval", counter_cmd, [counter_plan, counter_prior, counter_report]),
        ("semantic_second_pass_resolution", second_cmd, [second_plan, second_resolved, second_report]),
        ("exhaustive_relation_adjudication", adjudication_cmd, [adjudication_candidates, adjudication_review]),
        ("claim_evidence_graph", graph_cmd, [evidence_graph]),
        ("structural_claim_centrality", centrality_cmd, [centrality]),
        ("hypothesis_evidence_aggregation", aggregation_cmd, [aggregation]),
        ("positive_nonobviousness_basis", basis_cmd, [positive_basis]),
        ("positive_nonobviousness_adjudication", positive_cmd, [positive_review]),
        ("scientific_certification_gate", certification_cmd, [certification]),
    ]

    if [name for name, _, _ in stages] != plan.stage_names:
        raise RuntimeError("runtime stage order drifted from companion plan contract")

    atomic_n10_manifest = _load(Path(inputs.atomic_n10_manifest))
    manifest: dict[str, object] = {
        "schema_version": "scientific-verifier-shadow-e2e-manifest-v1",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "status": "planned" if args.dry_run else "running",
        "run_dir": inputs.run_dir,
        "plan": plan.model_dump(mode="json"),
        "model": args.model,
        "provider_plan": inputs.atomic_n10_provider_plan,
        "existing_atomic_n10_authority_mode": plan.lineage.atomic_n10_authority_mode,
        "existing_atomic_n10_manifest_status": plan.lineage.atomic_n10_manifest_status,
        "atomic_n10_completion_verified": plan.lineage.atomic_n10_completion_verified,
        "legacy_atomic_n10_artifact_mode": plan.lineage.legacy_atomic_n10_artifact_mode,
        "legacy_atomic_artifact_autoresolution_used": (
            plan.lineage.legacy_atomic_artifact_autoresolution_used
        ),
        "manifest_source_portfolio_binding_verified": (
            plan.lineage.manifest_source_portfolio_binding_verified
        ),
        "manifest_source_hypothesis_count_verified": (
            plan.lineage.manifest_source_hypothesis_count_verified
        ),
        "legacy_mode_is_historical_smoke_only": True,
        "prospective_validation_input_contract_satisfied": (
            plan.lineage.prospective_validation_input_contract_satisfied
        ),
        "existing_atomic_n10_certified_count": atomic_n10_manifest.get("atomic_scientific_certified_count"),
        "existing_atomic_n10_unresolved_count": atomic_n10_manifest.get("atomic_scientific_unresolved_count"),
        "existing_atomic_n10_rejected_count": atomic_n10_manifest.get("atomic_scientific_rejected_count"),
        "planned_stages": [
            {"name": name, "argv": argv, "expected": [str(path) for path in expected]}
            for name, argv, expected in stages
        ],
        "execution_performed": False,
        "scientific_quality_ranking_performed": False,
        "existing_atomic_n10_reused_read_only": True,
        "external_novelty_report_reused_for_distinctness_only": True,
        "relation_retrieval_is_independent_shadow_evidence": True,
        "grounded_identity_annotation_independent_of_n10_closure": True,
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "legacy_production_selection_mutated": False,
        "canonical_graph_mutated": False,
        "failure": None,
    }
    _write(manifest_path, manifest)

    print("Scientific verifier shadow companion")
    print("Atomic hypotheses:", plan.lineage.source_hypothesis_count)
    print("Atomic claims:", len(plan.lineage.atomic_claim_ids))
    print("Existing atomic N10 authority mode:", plan.lineage.atomic_n10_authority_mode)
    print("Existing atomic N10 manifest status:", plan.lineage.atomic_n10_manifest_status)
    print("Atomic N10 completion verified:", str(plan.lineage.atomic_n10_completion_verified).lower())
    print("Legacy atomic artifact smoke mode:", str(plan.lineage.legacy_atomic_n10_artifact_mode).lower())
    print(
        "Legacy exact artifact auto-resolution used:",
        str(plan.lineage.legacy_atomic_artifact_autoresolution_used).lower(),
    )
    print("Resolved atomic report:", inputs.atomic_report)
    print("Resolved atomic portfolio:", inputs.atomic_portfolio)
    print(
        "Manifest source portfolio binding verified:",
        str(plan.lineage.manifest_source_portfolio_binding_verified).lower(),
    )
    print(
        "Manifest source hypothesis count verified:",
        str(plan.lineage.manifest_source_hypothesis_count_verified).lower(),
    )
    print(
        "Prospective validation input contract satisfied:",
        str(plan.lineage.prospective_validation_input_contract_satisfied).lower(),
    )
    print("Exact atomic/external lineage binding: true")
    print("Existing atomic N10 reused read-only: true")
    print("Grounded identity depends on N10 closure details: false")
    print("Verifier result consumed by production: false")
    print("Production selection changed: false")

    if args.dry_run:
        for name, argv, _ in stages:
            print(f"  {name}:")
            print("    $", sys.executable, *argv)
        print("Execution performed: false")
        print("Manifest:", manifest_path)
        return 0

    try:
        for name, argv, expected in stages:
            _run(name, argv, expected)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _now()
        manifest["execution_performed"] = True
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _write(manifest_path, manifest)
        raise

    annotation_payload = _load(annotation)
    relation_ir_payload = _load(relation_ir)
    adjudication_payload = _load(adjudication_review)
    positive_payload = _load(positive_review)
    certification_payload = _load(certification)
    supporting_payload = _load(supporting_report)
    counter_payload = _load(counter_report)
    second_payload = _load(second_report)

    second_pass_logical_llm_calls = int(relation_ir_payload.get("ready_count", 0) or 0)
    manifest.update(
        {
            "status": "complete",
            "finished_at_utc": _now(),
            "execution_performed": True,
            "verifier_logical_llm_calls": {
                "grounded_identity_annotation": int(annotation_payload.get("llm_calls_performed", 0) or 0),
                "semantic_second_pass": second_pass_logical_llm_calls,
                "relation_adjudication": int(adjudication_payload.get("llm_calls_performed", 0) or 0),
                "positive_nonobviousness_adjudication": int(positive_payload.get("llm_calls_performed", 0) or 0),
            },
            "relation_review": {
                "presented_work_count": adjudication_payload.get("presented_work_count"),
                "classified_work_count": adjudication_payload.get("classified_work_count"),
                "unclassified_work_count": adjudication_payload.get("unclassified_work_count"),
            },
            "retrieval_summary": {
                "supporting": {
                    "transport_query_count": supporting_payload.get("transport_query_count"),
                    "successful_provider_query_count": supporting_payload.get("successful_provider_query_count"),
                    "failed_provider_query_count": supporting_payload.get("failed_provider_query_count"),
                    "unique_work_count": supporting_payload.get("unique_work_count"),
                    "abstract_work_count": supporting_payload.get("abstract_work_count"),
                },
                "counterevidence": {
                    "transport_query_count": counter_payload.get("transport_query_count"),
                    "successful_provider_query_count": counter_payload.get("successful_provider_query_count"),
                    "failed_provider_query_count": counter_payload.get("failed_provider_query_count"),
                    "unique_work_count": counter_payload.get("unique_work_count"),
                    "abstract_work_count": counter_payload.get("abstract_work_count"),
                },
                "second_pass": {
                    "semantic_query_count": second_payload.get("semantic_query_count"),
                    "resolution_query_count": second_payload.get("resolution_query_count"),
                    "resolved_unique_work_count": second_payload.get("resolved_unique_work_count"),
                    "abstract_work_count_after_resolution": second_payload.get("abstract_work_count_after_resolution"),
                },
            },
            "certification": {
                "report_id": certification_payload.get("report_id"),
                "certified_count": certification_payload.get("certified_count", 0),
                "unresolved_count": certification_payload.get("unresolved_count", 0),
                "rejected_count": certification_payload.get("rejected_count", 0),
                "decision_counts": certification_payload.get("decision_counts", {}),
            },
            "verifier_result_consumed_by_production": False,
            "production_selection_changed": False,
            "legacy_production_selection_mutated": False,
            "canonical_graph_mutated": False,
        }
    )
    _write(manifest_path, manifest)

    print()
    print("Scientific verifier shadow companion complete")
    print("Presented works:", adjudication_payload.get("presented_work_count"))
    print("Classified works:", adjudication_payload.get("classified_work_count"))
    print("Unclassified works:", adjudication_payload.get("unclassified_work_count"))
    print("Certified:", certification_payload.get("certified_count", 0))
    print("Unresolved:", certification_payload.get("unresolved_count", 0))
    print("Rejected:", certification_payload.get("rejected_count", 0))
    print("Verifier result consumed by production: false")
    print("Production selection changed: false")
    print("Certification:", certification)
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
