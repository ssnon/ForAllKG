from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline_core.discovery.atomic_scientific_verifier_prospective import (
    build_atomic_scientific_verifier_prospective_cohort_freeze,
    build_atomic_scientific_verifier_prospective_comparison,
)
from pipeline_core.discovery.external_novelty_contracts import LiteratureQueryPlan
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)
from pipeline_core.discovery.reframing.live_reframing_companion import (
    extract_run_dir_from_e2e_args,
    strip_remainder_separator,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    ScientificSynthesisNoveltyCertificationReport,
)
from pipeline_core.discovery.projection_relation_adjudication import (
    ProjectionRelationCandidateReport,
)
from pipeline_core.discovery.scientific_certification_gate import (
    ScientificCertificationGateReport,
)
from pipeline_core.discovery.scientific_hypothesis_evidence_aggregation import (
    ScientificHypothesisEvidenceAggregationReport,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run(name: str, argv: list[str], expected: list[Path] | None = None) -> None:
    print()
    print(f"== {name} ==")
    subprocess.run([sys.executable, *argv], check=True)
    if expected:
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise RuntimeError(f"{name} completed without expected outputs: {missing}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a prospective atomic scientific-verifier cohort: fresh legacy E2E, "
            "production-facing candidate materialization, atomic synthesis, immutable "
            "cohort freeze, completed certification-only atomic N10, then the independent "
            "0065-0078 scientific verifier and a deterministic old-vs-new comparison."
        )
    )
    parser.add_argument("--semantic-root", action="append", required=True)
    parser.add_argument(
        "--duplicate-paper-policy",
        choices=("error", "latest_attempt"),
        default="latest_attempt",
    )
    parser.add_argument(
        "--cross-root-duplicate-policy",
        choices=("error", "prefer_last_root"),
        default="prefer_last_root",
    )
    parser.add_argument("--atomic-max-syntheses", type=int, default=2)
    parser.add_argument("--atomic-parse-retries", type=int, default=3)
    parser.add_argument(
        "--resume-existing-candidate-contract",
        action="store_true",
        help=(
            "Resume a previously selected prospective case from its existing "
            "read-only pre-N10 candidate contract instead of rerunning legacy "
            "discovery/reframing. Requires a complete candidate-contract-only "
            "companion manifest and preserves that exact upstream population."
        ),
    )
    parser.add_argument(
        "--resume-frozen-atomic-cohort",
        action="store_true",
        help=(
            "Resume from an already frozen prospective atomic cohort. Reuses the "
            "existing candidate contract, atomic report/portfolio/query plan and "
            "verifies that recomputing the freeze yields the exact stored freeze "
            "before rerunning verification stages."
        ),
    )
    parser.add_argument(
        "--resume-completed-atomic-n10",
        action="store_true",
        help=(
            "With --resume-frozen-atomic-cohort, reuse an already complete atomic "
            "N10 assessment and its exact external-novelty artifacts read-only "
            "instead of rerunning stochastic retrieval/review."
        ),
    )
    parser.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument(
        "--critic-model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_CRITIC_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--support-results-per-query", type=int, default=12)
    parser.add_argument("--second-pass-results-per-query", type=int, default=16)
    parser.add_argument("--max-review-works-per-claim", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("e2e_args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    e2e_args = strip_remainder_separator(list(args.e2e_args))
    if not e2e_args:
        raise SystemExit("legacy E2E arguments are required after '--'")
    if not args.model or not args.critic_model:
        raise SystemExit("--model and --critic-model are required")
    if args.atomic_max_syntheses < 1:
        raise SystemExit("--atomic-max-syntheses must be >= 1")
    if args.atomic_parse_retries < 1:
        raise SystemExit("--atomic-parse-retries must be >= 1")

    if (
        args.resume_completed_atomic_n10
        and not args.resume_frozen_atomic_cohort
    ):
        raise SystemExit(
            "--resume-completed-atomic-n10 requires "
            "--resume-frozen-atomic-cohort"
        )

    run = extract_run_dir_from_e2e_args(e2e_args).expanduser().resolve()
    manifest_path = run / "scientific_atomic_verifier_prospective_e2e_manifest.json"
    plan_manifest_path = run.parent / (
        run.name + ".scientific_atomic_verifier_prospective_plan.json"
    )
    manifest = {
        "schema_version": "scientific-atomic-verifier-prospective-e2e-manifest-v1",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "status": "running",
        "run_dir": str(run),
        "prospective_validation": True,
        "legacy_artifact_smoke_mode_allowed": False,
        "cohort_frozen_before_old_n10": False,
        "cohort_frozen_before_new_verifier": False,
        "old_n10_complete_required_before_verifier": True,
        "old_n10_completed_before_verifier_started": False,
        "old_n10_mutated_by_verifier": False,
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "failure": None,
    }

    context = run / "hypothesis.context.json"
    candidates = run / "scientific_pre_n10_candidate_portfolio.json"
    atomic_report = run / "scientific_atomic_cross_lane.report.json"
    atomic_portfolio = run / "scientific_atomic_cross_lane.portfolio.json"
    atomic_query_plan = run / "scientific_atomic_cross_lane.query_plan.json"
    atomic_prompt = run / "scientific_atomic_cross_lane.prompt.txt"
    freeze_path = run / "scientific_atomic_verifier_prospective_cohort.json"
    old_n10_manifest = run / "scientific_atomic_n10_e2e_manifest.json"
    old_n10_report = run / "scientific_atomic_n10_certification.report.json"
    verifier_dir = run / "scientific_verifier_shadow"
    new_verifier_report = verifier_dir / "scientific_certification.report.json"
    aggregation_path = verifier_dir / "hypothesis_evidence_aggregation.json"
    relation_candidates_path = verifier_dir / "relation_adjudication.candidates.json"
    comparison_path = run / "scientific_atomic_verifier_prospective_comparison.json"

    if args.dry_run:
        stage_order = [
            "fresh_legacy_discovery_e2e",
            "production_facing_candidate_contract",
            "fresh_atomic_cross_lane_synthesis",
            "freeze_atomic_cohort_before_verification",
            "existing_atomic_n10_certification_only",
            "require_atomic_n10_status_complete",
            "independent_scientific_verifier_0065_0078",
            "deterministic_old_vs_new_comparison",
        ]
        manifest["status"] = "planned"
        manifest["dry_run"] = True
        manifest["planned_stage_order"] = stage_order
        manifest["finished_at_utc"] = _now()
        _write(plan_manifest_path, manifest)
        print("Atomic scientific verifier prospective E2E plan")
        for index, stage in enumerate(stage_order, start=1):
            print(f"  {index}. {stage}")
        print("Legacy smoke mode allowed: false")
        print("Cohort freeze precedes both verification paths: true")
        print("Completed atomic N10 required before verifier: true")
        print("Verifier result consumed by production: false")
        print("Production selection changed: false")
        print("Execution performed: false")
        print("Manifest:", plan_manifest_path)
        return 0

    try:
        pre_manifest_path = run / "scientific_pre_n10_synthesis_companion_manifest.json"
        if args.resume_existing_candidate_contract or args.resume_frozen_atomic_cohort:
            if not context.is_file() or not candidates.is_file():
                raise RuntimeError(
                    "resume requires existing hypothesis.context.json and "
                    "scientific_pre_n10_candidate_portfolio.json"
                )
            if not pre_manifest_path.is_file():
                raise RuntimeError(
                    "resume requires scientific_pre_n10_synthesis_companion_manifest.json"
                )
            pre_manifest = _load_object(pre_manifest_path)
            if pre_manifest.get("status") != "complete_candidate_contract_only":
                raise RuntimeError(
                    "resume requires complete_candidate_contract_only; observed status="
                    + repr(pre_manifest.get("status"))
                )
            if pre_manifest.get("legacy_e2e_status") != "complete":
                raise RuntimeError(
                    "resume requires legacy_e2e_status='complete' in pre-N10 manifest"
                )
            if pre_manifest.get("candidate_contract_only") is not True:
                raise RuntimeError(
                    "resume requires candidate_contract_only=true in pre-N10 manifest"
                )
            manifest["resumed_from_existing_candidate_contract"] = True
            manifest["candidate_contract_reused_read_only"] = True
            manifest["legacy_e2e_completed"] = True
            _write(manifest_path, manifest)
            print()
            print("Reusing existing prospective candidate contract read-only")
            print("Candidate portfolio:", candidates)
            print("Pre-N10 manifest:", pre_manifest_path)
        else:
            _run(
                "fresh_legacy_discovery_e2e",
                ["-m", "scripts.discovery.run_dac_discovery_e2e", *e2e_args],
            )
            manifest["legacy_e2e_completed"] = True
            _write(manifest_path, manifest)

            pre_n10_cmd = [
                "-m",
                "scripts.discovery.run_pre_n10_scientific_synthesis_companion",
                "--run-dir",
                str(run),
                "--candidate-contract-only",
                "--duplicate-paper-policy",
                args.duplicate_paper_policy,
                "--cross-root-duplicate-policy",
                args.cross_root_duplicate_policy,
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
            ]
            for root in args.semantic_root:
                pre_n10_cmd += ["--root", root]
            if args.base_url:
                pre_n10_cmd += ["--base-url", args.base_url]
            if args.save_prompts:
                pre_n10_cmd += ["--save-prompts"]
            _run("production_facing_candidate_contract", pre_n10_cmd)
            if not candidates.is_file():
                pre_status = None
                if pre_manifest_path.is_file():
                    pre_status = _load_object(pre_manifest_path).get("status")
                if str(pre_status or "").startswith("abstained_"):
                    manifest["status"] = "abstained_upstream_before_candidate_contract"
                    manifest["upstream_pre_n10_status"] = pre_status
                    manifest["finished_at_utc"] = _now()
                    _write(manifest_path, manifest)
                    print("Disposition: abstained_upstream_before_candidate_contract")
                    return 0
                raise RuntimeError(
                    "pre-N10 companion did not materialize candidate portfolio"
                )

        candidate_portfolio = ProductionFacingScientificCandidatePortfolio.model_validate_json(
            candidates.read_text(encoding="utf-8")
        )
        if candidate_portfolio.candidate_count == 0:
            manifest["status"] = "abstained_no_production_facing_candidates"
            manifest["finished_at_utc"] = _now()
            _write(manifest_path, manifest)
            print("Disposition: abstained_no_production_facing_candidates")
            return 0

        if not args.resume_frozen_atomic_cohort:
            atomic_cmd = [
                "-m",
                "scripts.discovery.run_atomic_cross_lane_scientific_synthesis",
                "--candidate-portfolio",
                str(candidates),
                "--context",
                str(context),
                "--report-output",
                str(atomic_report),
                "--portfolio-output",
                str(atomic_portfolio),
                "--query-plan-output",
                str(atomic_query_plan),
                "--prompt-output",
                str(atomic_prompt),
                "--max-syntheses",
                str(args.atomic_max_syntheses),
                "--model",
                args.model,
                "--api-key-env",
                args.api_key_env,
                "--parse-retries",
                str(args.atomic_parse_retries),
            ]
            if args.base_url:
                atomic_cmd += ["--base-url", args.base_url]
            _run(
                "fresh_atomic_cross_lane_synthesis",
                atomic_cmd,
                [atomic_report, atomic_portfolio, atomic_query_plan],
            )
        else:
            required_frozen = [
                atomic_report,
                atomic_portfolio,
                atomic_query_plan,
                freeze_path,
            ]
            missing_frozen = [
                str(path) for path in required_frozen if not path.is_file()
            ]
            if missing_frozen:
                raise RuntimeError(
                    "frozen-cohort resume missing required artifacts: "
                    + repr(missing_frozen)
                )

        context_model = HypothesisContext.model_validate_json(
            context.read_text(encoding="utf-8")
        )
        atomic_report_model = AtomicCrossLaneSynthesisReport.model_validate_json(
            atomic_report.read_text(encoding="utf-8")
        )
        atomic_portfolio_model = HypothesisPortfolio.model_validate_json(
            atomic_portfolio.read_text(encoding="utf-8")
        )
        query_plan_model = LiteratureQueryPlan.model_validate_json(
            atomic_query_plan.read_text(encoding="utf-8")
        )
        if not atomic_portfolio_model.hypotheses:
            manifest["status"] = "abstained_no_atomic_hypotheses"
            manifest["finished_at_utc"] = _now()
            _write(manifest_path, manifest)
            print("Disposition: abstained_no_atomic_hypotheses")
            return 0

        freeze = build_atomic_scientific_verifier_prospective_cohort_freeze(
            context=context_model,
            candidate_portfolio=candidate_portfolio,
            atomic_report=atomic_report_model,
            atomic_portfolio=atomic_portfolio_model,
            query_plan=query_plan_model,
        )
        if args.resume_frozen_atomic_cohort:
            stored_freeze = _load_object(freeze_path)
            recomputed_freeze = freeze.model_dump(mode="json")
            if stored_freeze != recomputed_freeze:
                raise RuntimeError(
                    "frozen-cohort resume lineage mismatch: stored freeze does "
                    "not equal the freeze recomputed from current atomic artifacts"
                )
            manifest["resumed_from_frozen_atomic_cohort"] = True
            manifest["frozen_atomic_artifacts_reused_read_only"] = True
            print()
            print("Reusing frozen prospective atomic cohort read-only")
            print("Hypotheses:", freeze.hypothesis_count)
            print("Claims:", freeze.claim_count)
            print("Freeze:", freeze_path)
        else:
            _write(freeze_path, freeze)
        manifest["cohort_freeze"] = str(freeze_path)
        manifest["cohort_freeze_id"] = freeze.freeze_id
        manifest["frozen_hypothesis_count"] = freeze.hypothesis_count
        manifest["frozen_claim_count"] = freeze.claim_count
        manifest["cohort_frozen_before_old_n10"] = True
        manifest["cohort_frozen_before_new_verifier"] = True
        _write(manifest_path, manifest)
        print()
        print("Prospective atomic cohort frozen")
        print("Hypotheses:", freeze.hypothesis_count)
        print("Claims:", freeze.claim_count)
        print("Freeze:", freeze_path)

        n10_cmd = [
            "-m",
            "scripts.discovery.run_atomic_scientific_synthesis_n10_e2e",
            "--run-dir",
            str(run),
            "--portfolio",
            str(atomic_portfolio),
            "--query-plan",
            str(atomic_query_plan),
            "--authority-mode",
            "certification_only",
            "--model",
            args.model,
            "--critic-model",
            args.critic_model,
            "--api-key-env",
            args.api_key_env,
        ]
        if args.base_url:
            n10_cmd += ["--base-url", args.base_url]
        if args.save_prompts:
            n10_cmd += ["--save-prompts"]
        if not args.resume_completed_atomic_n10:
            _run(
                "existing_atomic_n10_certification_only",
                n10_cmd,
                [old_n10_manifest, old_n10_report],
            )
        else:
            required_old_n10 = [
                old_n10_manifest,
                old_n10_report,
                run / "scientific_atomic_n10_external.report.json",
                run / "scientific_atomic_n10_external.provider_plan.json",
            ]
            missing_old_n10 = [
                str(path)
                for path in required_old_n10
                if not path.is_file()
            ]
            if missing_old_n10:
                raise RuntimeError(
                    "completed-N10 resume missing required artifacts: "
                    + repr(missing_old_n10)
                )
            manifest["resumed_from_completed_atomic_n10"] = True
            manifest["completed_atomic_n10_reused_read_only"] = True
            print()
            print("Reusing completed atomic N10 read-only")
            print("Manifest:", old_n10_manifest)
            print("Certification report:", old_n10_report)

        old_manifest = _load_object(old_n10_manifest)
        if old_manifest.get("status") != "complete":
            raise RuntimeError(
                "prospective verifier requires completed atomic N10 E2E; observed status="
                + repr(old_manifest.get("status"))
            )
        if not bool(old_manifest.get("precomputed_atomic_query_plan_reused")):
            raise RuntimeError("atomic N10 did not reuse frozen precomputed query plan")
        if bool(old_manifest.get("external_novelty_llm_redecomposition_performed")):
            raise RuntimeError("atomic N10 unexpectedly re-decomposed frozen atomic claims")
        if old_manifest.get("authority_mode") != "certification_only":
            raise RuntimeError(
                "prospective verifier requires atomic N10 certification_only authority"
            )
        if Path(str(old_manifest.get("source_portfolio", ""))).resolve() != atomic_portfolio:
            raise RuntimeError(
                "completed atomic N10 source portfolio does not match frozen atomic portfolio"
            )
        if Path(str(old_manifest.get("source_query_plan", ""))).resolve() != atomic_query_plan:
            raise RuntimeError(
                "completed atomic N10 source query plan does not match frozen atomic query plan"
            )
        if int(old_manifest.get("source_hypothesis_count", -1)) != freeze.hypothesis_count:
            raise RuntimeError(
                "completed atomic N10 hypothesis count does not match frozen cohort"
            )
        manifest["old_n10_completed_before_verifier_started"] = True
        manifest["old_n10_manifest"] = str(old_n10_manifest)
        manifest["old_n10_report"] = str(old_n10_report)
        _write(manifest_path, manifest)

        verifier_cmd = [
            "-m",
            "scripts.discovery.run_scientific_verifier_shadow_companion",
            "--run-dir",
            str(run),
            "--candidate-portfolio",
            str(candidates),
            "--atomic-report",
            str(atomic_report),
            "--atomic-portfolio",
            str(atomic_portfolio),
            "--atomic-n10-manifest",
            str(old_n10_manifest),
            "--external-novelty",
            str(run / "scientific_atomic_n10_external.report.json"),
            "--provider-plan",
            str(run / "scientific_atomic_n10_external.provider_plan.json"),
            "--context",
            str(context),
            "--output-dir",
            str(verifier_dir),
            "--model",
            args.critic_model,
            "--api-key-env",
            args.api_key_env,
            "--support-results-per-query",
            str(args.support_results_per_query),
            "--second-pass-results-per-query",
            str(args.second_pass_results_per_query),
            "--max-review-works-per-claim",
            str(args.max_review_works_per_claim),
        ]
        if args.base_url:
            verifier_cmd += ["--base-url", args.base_url]
        if args.save_prompts:
            verifier_cmd += ["--save-prompts"]
        _run(
            "independent_scientific_verifier_0065_0078",
            verifier_cmd,
            [new_verifier_report, aggregation_path],
        )

        old_report_model = ScientificSynthesisNoveltyCertificationReport.model_validate_json(
            old_n10_report.read_text(encoding="utf-8")
        )
        new_report_model = ScientificCertificationGateReport.model_validate_json(
            new_verifier_report.read_text(encoding="utf-8")
        )
        aggregation_model = ScientificHypothesisEvidenceAggregationReport.model_validate_json(
            aggregation_path.read_text(encoding="utf-8")
        )
        relation_candidates_model = ProjectionRelationCandidateReport.model_validate_json(
            relation_candidates_path.read_text(encoding="utf-8")
        )
        comparison = build_atomic_scientific_verifier_prospective_comparison(
            cohort_freeze=freeze,
            old_n10=old_report_model,
            new_verifier=new_report_model,
            aggregation=aggregation_model,
            relation_candidates=relation_candidates_model,
        )
        _write(comparison_path, comparison)

        manifest["comparison_report"] = str(comparison_path)
        manifest["comparison_report_id"] = comparison.report_id
        manifest["agreement_count"] = comparison.agreement_count
        manifest["disagreement_count"] = comparison.disagreement_count
        manifest["comparison_cells"] = comparison.comparison_cells
        manifest["verifier_result_consumed_by_production"] = False
        manifest["production_selection_changed"] = False
        manifest["canonical_graph_mutated"] = False
        manifest["status"] = "complete"
        manifest["finished_at_utc"] = _now()
        _write(manifest_path, manifest)

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _now()
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write(manifest_path, manifest)
        raise

    print()
    print("Atomic scientific verifier prospective E2E complete")
    print("Frozen hypotheses:", freeze.hypothesis_count)
    print("Old/new agreements:", comparison.agreement_count)
    print("Old/new disagreements:", comparison.disagreement_count)
    print("Comparison cells:", comparison.comparison_cells)
    for row in comparison.rows:
        print(
            " ",
            row.hypothesis_id,
            "old=",
            row.old_n10_decision,
            "new=",
            row.new_verifier_decision,
            "factors=",
            row.diagnostic_factors,
        )
    print("Cohort changed after freeze: false")
    print("Verifier result consumed by production: false")
    print("Production selection changed: false")
    print("Freeze:", freeze_path)
    print("Comparison:", comparison_path)
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
