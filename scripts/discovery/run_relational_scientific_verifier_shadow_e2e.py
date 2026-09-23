from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_atomic_endpoint_binding import (
    RelationalAtomicEndpointBindingReport,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    RelationalVerifierStageRecord,
    assert_fingerprints_unchanged,
    build_relational_scientific_verifier_input_freeze,
    build_relational_scientific_verifier_run_manifest,
    fingerprint_file,
    sha256_file,
    write_json_exclusive,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def _git_state() -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return head, bool(status.strip())


def _run(
    name: str,
    argv: list[str],
    expected: list[Path],
) -> None:
    print()
    print("==", name, "==")
    subprocess.run([sys.executable, *argv], check=True)
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            name + " completed without expected outputs: " + repr(missing)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen relational-only scientific verifier shadow from "
            "literal endpoint bindings through search-bounded certification. "
            "The source population and endpoint binding are frozen inputs. "
            "An input-freeze artifact is written before any verifier stage, "
            "and the completed run manifest is immutable/write-once."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--endpoint-report", required=True, type=Path)
    parser.add_argument("--provider-plan", required=True, type=Path)
    parser.add_argument(
        "--source-external-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--final-hypothesis-id", required=True)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    parser.add_argument(
        "--second-pass-results-per-query",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-review-works-per-claim",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-exhaustive-rounds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-second-pass-queries-per-claim",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-resolution-queries",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--max-alias-query-variants-per-projection",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-lower-order-factor-order",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-source-alias-variants-per-projection",
        type=int,
        default=2,
    )
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument(
        "--allow-dirty-worktree",
        action="store_true",
        help=(
            "Calibration/debug only. Prospective validation should run from "
            "a clean committed worktree so the manifest code SHA is exact."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.model:
        raise SystemExit("--model is required")

    plan_path = args.plan.expanduser().resolve()
    endpoint_path = args.endpoint_report.expanduser().resolve()
    provider_path = args.provider_plan.expanduser().resolve()
    external_path = args.source_external_report.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for path in (plan_path, endpoint_path, provider_path, external_path):
        if not path.is_file():
            raise ValueError("missing frozen verifier input: " + str(path))

    plan = RelationalAtomicBindingPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    endpoint = RelationalAtomicEndpointBindingReport.model_validate_json(
        endpoint_path.read_text(encoding="utf-8")
    )
    source_external = ExternalNoveltyReport.model_validate_json(
        external_path.read_text(encoding="utf-8")
    )

    if endpoint.source_binding_plan_id != plan.plan_id:
        raise ValueError("endpoint report/binding plan ID mismatch")
    if endpoint.source_binding_plan_sha256 != plan.plan_sha256:
        raise ValueError("endpoint report/binding plan SHA mismatch")

    lineage_rows = [
        row
        for row in plan.hypotheses
        if row.final_hypothesis_id == args.final_hypothesis_id
    ]
    if len(lineage_rows) != 1:
        raise ValueError(
            "final hypothesis must resolve exactly one binding-plan lineage"
        )
    lineage = lineage_rows[0]
    if (
        lineage.binding_status
        != "READY_FOR_LITERAL_ENDPOINT_BINDING"
    ):
        raise ValueError(
            "selected final hypothesis is not binding-ready"
        )
    if lineage.candidate_final_authority_equivalent is not True:
        raise ValueError(
            "selected lineage lacks candidate/final authority equivalence"
        )

    selected_endpoint_hypotheses = {
        row.final_hypothesis_id
        for row in endpoint.bindings
    }
    if selected_endpoint_hypotheses != {args.final_hypothesis_id}:
        raise ValueError(
            "0091 single-hypothesis runner requires endpoint report to contain "
            "only the selected final hypothesis; observed="
            + repr(sorted(selected_endpoint_hypotheses))
        )
    if endpoint.selected_claim_count < 1:
        raise ValueError("endpoint report contains no selected claims")

    source_external_cards = [
        row
        for row in source_external.cards
        if row.hypothesis_id == lineage.candidate_hypothesis_id
    ]
    if len(source_external_cards) != 1:
        raise ValueError(
            "source external novelty report must contain exactly one card "
            "for the selected candidate hypothesis"
        )

    final_portfolio_path = Path(
        plan.source_alpha6_candidate_portfolio
    ).expanduser().resolve()
    if not final_portfolio_path.is_file():
        raise ValueError(
            "missing frozen final Alpha6 portfolio: "
            + str(final_portfolio_path)
        )
    if sha256_file(final_portfolio_path) != (
        plan.source_alpha6_candidate_portfolio_sha256
    ):
        raise ValueError(
            "final Alpha6 portfolio changed after binding-plan freeze"
        )
    final_portfolio = HypothesisPortfolio.model_validate_json(
        final_portfolio_path.read_text(encoding="utf-8")
    )
    final_cards = [
        row
        for row in final_portfolio.hypotheses
        if row.hypothesis_id == args.final_hypothesis_id
    ]
    if len(final_cards) != 1:
        raise ValueError(
            "final Alpha6 portfolio must contain selected final hypothesis"
        )

    repo_head, repo_dirty = _git_state()
    if repo_dirty and not args.allow_dirty_worktree:
        raise ValueError(
            "relational verifier prospective runner requires a clean "
            "worktree; commit/stash changes or use --allow-dirty-worktree "
            "for calibration/debug only"
        )

    freeze_path = output_dir / (
        "relational_scientific_verifier.input_freeze.json"
    )
    manifest_path = output_dir / (
        "relational_scientific_verifier.run_manifest.json"
    )
    failure_path = output_dir / (
        "relational_scientific_verifier.failure.json"
    )

    if freeze_path.exists() or manifest_path.exists():
        raise ValueError(
            "immutable verifier output already exists; use a fresh output "
            "directory instead of overwriting a frozen run"
        )

    frozen_inputs = [
        fingerprint_file(plan_path),
        fingerprint_file(endpoint_path),
        fingerprint_file(provider_path),
        fingerprint_file(external_path),
        fingerprint_file(final_portfolio_path),
    ]
    freeze = build_relational_scientific_verifier_input_freeze(
        repository_head_sha=repo_head,
        repository_worktree_dirty=repo_dirty,
        final_hypothesis_id=args.final_hypothesis_id,
        candidate_hypothesis_id=lineage.candidate_hypothesis_id,
        binding_plan_id=plan.plan_id,
        endpoint_binding_report_id=endpoint.report_id,
        source_external_novelty_report_id=source_external.report_id,
        final_alpha6_portfolio_id=final_portfolio.portfolio_id,
        domain_profile_id=args.domain_profile,
        model_name=args.model,
        support_results_per_query=args.support_results_per_query,
        second_pass_results_per_query=(
            args.second_pass_results_per_query
        ),
        max_review_works_per_claim=args.max_review_works_per_claim,
        max_exhaustive_rounds=args.max_exhaustive_rounds,
        max_second_pass_queries_per_claim=(
            args.max_second_pass_queries_per_claim
        ),
        max_resolution_queries=args.max_resolution_queries,
        max_alias_query_variants_per_projection=(
            args.max_alias_query_variants_per_projection
        ),
        max_lower_order_factor_order=(
            args.max_lower_order_factor_order
        ),
        max_source_alias_variants_per_projection=(
            args.max_source_alias_variants_per_projection
        ),
        input_artifacts=frozen_inputs,
    )

    relational_projection = output_dir / "relational_atomic_projection.json"
    relation_ir = output_dir / "scientific_relation_ir.json"
    relation_projection = output_dir / "scientific_relation_projection.json"
    factorization = output_dir / (
        "relational_atomic_identity_factorization.json"
    )
    grounded_projection = output_dir / "grounded_factor_projection.json"

    supporting_prefix = output_dir / "supporting_projection"
    supporting_plan = supporting_prefix.with_suffix(
        ".supporting_query_plan.json"
    )
    supporting_prior = supporting_prefix.with_suffix(".prior_art.json")
    supporting_report = supporting_prefix.with_suffix(
        ".retrieval_report.json"
    )

    counter_prefix = output_dir / "counterevidence_projection"
    counter_plan = counter_prefix.with_suffix(
        ".counterevidence_query_plan.json"
    )
    counter_prior = counter_prefix.with_suffix(".prior_art.json")
    counter_report = counter_prefix.with_suffix(
        ".retrieval_report.json"
    )

    second_prefix = output_dir / "relation_second_pass"
    second_plan = second_prefix.with_suffix(".semantic_plan.json")
    second_resolved = second_prefix.with_suffix(
        ".resolved_prior_art.json"
    )
    second_report = second_prefix.with_suffix(".report.json")

    adjudication_prefix = output_dir / "relation_adjudication"
    adjudication_candidates = adjudication_prefix.with_suffix(
        ".candidates.json"
    )
    adjudication_review = adjudication_prefix.with_suffix(
        ".review.json"
    )

    evidence_graph = output_dir / "claim_evidence_graph.json"
    centrality = output_dir / "claim_centrality.json"
    aggregation = output_dir / "hypothesis_evidence_aggregation.json"
    positive_basis = output_dir / "positive_nonobviousness_basis.json"
    positive_prefix = output_dir / "positive_nonobviousness_adjudication"
    positive_review = positive_prefix.with_suffix(".review.json")

    projected_external = output_dir / "external_novelty.projected.json"
    external_audit = output_dir / "external_novelty.lineage_audit.json"
    certification = output_dir / "scientific_certification.report.json"

    model_args = [
        "--model", args.model,
        "--api-key-env", args.api_key_env,
    ]
    if args.base_url:
        model_args += ["--base-url", args.base_url]

    relation_ir_cmd = [
        "-m",
        "scripts.discovery.build_relational_atomic_relation_ir_shadow",
        "--plan", str(plan_path),
        "--endpoint-report", str(endpoint_path),
        "--domain-profile", args.domain_profile,
        "--projection-output", str(relational_projection),
        "--relation-ir-output", str(relation_ir),
    ]
    factor_projection_cmd = [
        "-m",
        "scripts.discovery.build_relational_atomic_factor_projection_shadow",
        "--plan", str(plan_path),
        "--relational-projection", str(relational_projection),
        "--relation-ir", str(relation_ir),
        "--relation-projection-output", str(relation_projection),
        "--factorization-output", str(factorization),
        "--grounded-projection-output", str(grounded_projection),
        "--max-alias-query-variants-per-projection",
        str(args.max_alias_query_variants_per_projection),
    ]
    supporting_cmd = [
        "-m",
        "scripts.discovery.run_supporting_projection_retrieval_shadow",
        "--portfolio", str(final_portfolio_path),
        "--projection", str(grounded_projection),
        "--provider-plan", str(provider_path),
        "--results-per-query", str(args.support_results_per_query),
        "--output-prefix", str(supporting_prefix),
    ]
    counter_cmd = [
        "-m",
        "scripts.discovery.run_counterevidence_projection_retrieval_shadow",
        "--portfolio", str(final_portfolio_path),
        "--projection", str(grounded_projection),
        "--provider-plan", str(provider_path),
        "--results-per-query", str(args.support_results_per_query),
        "--max-lower-order-factor-order",
        str(args.max_lower_order_factor_order),
        "--max-source-alias-variants-per-projection",
        str(args.max_source_alias_variants_per_projection),
        "--output-prefix", str(counter_prefix),
    ]
    second_cmd = [
        "-m",
        "scripts.discovery.run_scientific_relation_second_pass_shadow",
        "--relation-ir", str(relation_ir),
        "--projection", str(grounded_projection),
        "--domain-profile", args.domain_profile,
        "--source-portfolio-id", final_portfolio.portfolio_id,
        "--provider-plan", str(provider_path),
        "--results-per-query",
        str(args.second_pass_results_per_query),
        "--max-queries-per-claim",
        str(args.max_second_pass_queries_per_claim),
        "--max-resolution-queries",
        str(args.max_resolution_queries),
        "--output-prefix", str(second_prefix),
        *model_args,
    ]
    adjudication_cmd = [
        "-m",
        "scripts.discovery.run_projection_relation_adjudication_shadow",
        "--relation-ir", str(relation_ir),
        "--projection", str(grounded_projection),
        "--supporting-plan", str(supporting_plan),
        "--supporting-prior-art", str(supporting_prior),
        "--counterevidence-plan", str(counter_plan),
        "--counterevidence-prior-art", str(counter_prior),
        "--second-pass-plan", str(second_plan),
        "--second-pass-prior-art", str(second_resolved),
        "--domain-profile", args.domain_profile,
        "--max-review-works-per-claim",
        str(args.max_review_works_per_claim),
        "--max-exhaustive-rounds",
        str(args.max_exhaustive_rounds),
        "--output-prefix", str(adjudication_prefix),
        *model_args,
    ]
    graph_cmd = [
        "-m",
        "scripts.discovery.build_scientific_claim_evidence_graph_shadow",
        "--relation-ir", str(relation_ir),
        "--projection", str(grounded_projection),
        "--candidates", str(adjudication_candidates),
        "--adjudication", str(adjudication_review),
        "--output", str(evidence_graph),
    ]
    centrality_cmd = [
        "-m",
        "scripts.discovery.run_scientific_claim_centrality_shadow",
        "--evidence-graph", str(evidence_graph),
        "--output", str(centrality),
    ]
    aggregation_cmd = [
        "-m",
        "scripts.discovery.run_scientific_hypothesis_evidence_aggregation_shadow",
        "--evidence-graph", str(evidence_graph),
        "--centrality", str(centrality),
        "--output", str(aggregation),
    ]
    basis_cmd = [
        "-m",
        "scripts.discovery.run_positive_nonobviousness_basis_shadow",
        "--evidence-graph", str(evidence_graph),
        "--aggregation", str(aggregation),
        "--output", str(positive_basis),
    ]
    positive_cmd = [
        "-m",
        "scripts.discovery.run_positive_nonobviousness_adjudication_shadow",
        "--evidence-graph", str(evidence_graph),
        "--basis", str(positive_basis),
        "--output-prefix", str(positive_prefix),
        *model_args,
    ]
    external_cmd = [
        "-m",
        "scripts.discovery.project_external_novelty_lineage_shadow",
        "--plan", str(plan_path),
        "--source-external-report", str(external_path),
        "--final-hypothesis-id", args.final_hypothesis_id,
        "--output-report", str(projected_external),
        "--output-audit", str(external_audit),
    ]
    certification_cmd = [
        "-m",
        "scripts.discovery.run_scientific_certification_gate_shadow",
        "--aggregation", str(aggregation),
        "--nonobviousness", str(positive_review),
        "--external-novelty", str(projected_external),
        "--output", str(certification),
    ]

    if args.save_prompts:
        second_cmd += ["--save-prompts"]
        adjudication_cmd += ["--save-prompts"]
        positive_cmd += ["--save-prompts"]

    stages: list[tuple[str, list[str], list[Path]]] = [
        (
            "relational_atomic_projection_and_relation_ir",
            relation_ir_cmd,
            [relational_projection, relation_ir],
        ),
        (
            "relational_atomic_identity_and_factor_projection",
            factor_projection_cmd,
            [relation_projection, factorization, grounded_projection],
        ),
        (
            "supporting_projection_retrieval",
            supporting_cmd,
            [supporting_plan, supporting_prior, supporting_report],
        ),
        (
            "counterevidence_projection_retrieval",
            counter_cmd,
            [counter_plan, counter_prior, counter_report],
        ),
        (
            "semantic_second_pass_resolution",
            second_cmd,
            [second_plan, second_resolved, second_report],
        ),
        (
            "exhaustive_relation_adjudication",
            adjudication_cmd,
            [adjudication_candidates, adjudication_review],
        ),
        (
            "claim_evidence_graph",
            graph_cmd,
            [evidence_graph],
        ),
        (
            "structural_claim_centrality",
            centrality_cmd,
            [centrality],
        ),
        (
            "hypothesis_evidence_aggregation",
            aggregation_cmd,
            [aggregation],
        ),
        (
            "positive_nonobviousness_basis",
            basis_cmd,
            [positive_basis],
        ),
        (
            "positive_nonobviousness_adjudication",
            positive_cmd,
            [positive_review],
        ),
        (
            "external_novelty_lineage_projection",
            external_cmd,
            [projected_external, external_audit],
        ),
        (
            "scientific_certification_gate",
            certification_cmd,
            [certification],
        ),
    ]

    print("Relational scientific verifier shadow E2E")
    print("Repository HEAD:", repo_head)
    print("Worktree dirty:", str(repo_dirty).lower())
    print("Candidate:", lineage.candidate_hypothesis_id)
    print("Final:", args.final_hypothesis_id)
    print("Frozen binding plan:", plan.plan_id)
    print("Frozen endpoint report:", endpoint.report_id)
    print("Stages:", len(stages))
    print("Production authority: false")

    if args.dry_run:
        for index, (name, argv, expected) in enumerate(stages, start=1):
            print()
            print(index, name)
            print("  $", sys.executable, *argv)
            print("  outputs:", [str(path) for path in expected])
        print()
        print("Execution performed: false")
        print("Input freeze written: false")
        print("Run manifest written: false")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(freeze_path, freeze)
    print("Input freeze:", freeze_path)
    print("Freeze ID:", freeze.freeze_id)

    stage_records: list[RelationalVerifierStageRecord] = []
    try:
        for index, (name, argv, expected) in enumerate(stages, start=1):
            _run(name, argv, expected)
            stage_records.append(
                RelationalVerifierStageRecord(
                    stage_index=index,
                    stage_name=name,
                    argv=argv,
                    output_artifacts=[
                        fingerprint_file(path)
                        for path in expected
                    ],
                )
            )

        assert_fingerprints_unchanged(frozen_inputs)

        certification_payload = _load(certification)
        decisions = certification_payload.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 1:
            raise ValueError(
                "0091 single-hypothesis runner requires exactly one "
                "certification decision"
            )
        decision = decisions[0]
        if not isinstance(decision, dict):
            raise ValueError("certification decision must be an object")
        if decision.get("hypothesis_id") != args.final_hypothesis_id:
            raise ValueError(
                "certification decision/final hypothesis mismatch"
            )

        manifest = build_relational_scientific_verifier_run_manifest(
            freeze=freeze,
            stage_records=stage_records,
            certification_report_id=str(
                certification_payload.get("report_id") or ""
            ),
            certification_decision=str(
                decision.get("decision") or ""
            ),
            bounded_closure_state=str(
                decision.get("bounded_closure_state") or ""
            ),
            bounded_external_distinctness_state=str(
                decision.get("bounded_external_distinctness_state") or ""
            ),
            positive_nonobviousness_authority_state=str(
                decision.get(
                    "positive_nonobviousness_authority_state"
                )
                or ""
            ),
            fatal_blocker_state=str(
                decision.get("fatal_blocker_state") or ""
            ),
        )
        write_json_exclusive(manifest_path, manifest)

    except Exception as exc:
        if not failure_path.exists():
            write_json_exclusive(
                failure_path,
                {
                    "schema_version":
                        "relational-scientific-verifier-failure-v1",
                    "input_freeze_id": freeze.freeze_id,
                    "repository_head_sha": repo_head,
                    "failed_after_completed_stage_count":
                        len(stage_records),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "production_selection_changed": False,
                    "canonical_graph_mutated": False,
                },
            )
        raise

    print()
    print("Relational scientific verifier shadow complete")
    print("Manifest:", manifest_path)
    print("Manifest ID:", manifest.manifest_id)
    print("Certification:", manifest.certification_decision)
    print("Bounded closure:", manifest.bounded_closure_state)
    print(
        "Bounded external distinctness:",
        manifest.bounded_external_distinctness_state,
    )
    print(
        "Positive non-obviousness:",
        manifest.positive_nonobviousness_authority_state,
    )
    print("Fatal blocker:", manifest.fatal_blocker_state)
    print("Input artifacts unchanged after execution: true")
    print("External novelty reassessed: false")
    print("Verifier result consumed by production: false")
    print("Production selection changed: false")
    print("Canonical graph mutated: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
