from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from domains.feasibility_registry import resolve_feasibility_adapter
from domains.registry import get_domain_profile
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)
from pipeline_core.discovery.reframing.scientific_synthesis_bounded_authority import (
    merge_legacy_with_bounded_scientific_survivors,
    merge_scientific_synthesis_n10_rounds,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    ScientificSynthesisN10FilterReport,
)
from pipeline_core.discovery.reframing.scientific_synthesis_specification_repair import (
    ScientificSynthesisRepairReport,
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


def _run(name: str, argv: list[str], expected: list[Path]) -> None:
    print()
    print(f"== {name} ==")
    subprocess.run([sys.executable, *argv], check=True)
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"{name} completed without expected outputs: {missing}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh depth-1 N10 over bounded specification-repaired scientific "
            "synthesis candidates, merge first-pass and repaired positive-authority "
            "survivors, then merge with strict legacy N10 survivors and run sidecar "
            "final semantic/feasibility. No further repair is allowed."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
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
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument("--max-ranked-works", type=int, default=8)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    run = args.run_dir.expanduser().resolve()
    context_path = run / "hypothesis.context.json"
    initial = run / "scientific_synthesis_pre_n10.portfolio.json"
    first_survivors = run / "scientific_synthesis_n10_survivors.portfolio.json"
    first_filter = run / "scientific_synthesis_n10_filter_report.json"
    repaired = run / "scientific_synthesis_bounded_repair.portfolio.json"
    repair_report_path = run / "scientific_synthesis_bounded_repair_report.json"

    provider_plan = run / "scientific_synthesis_n10_external.provider_plan.json"
    if not provider_plan.is_file():
        provider_plan = run / "literature_provider_plan.json"

    for path, label in [
        (context_path, "hypothesis context"),
        (initial, "initial scientific synthesis portfolio"),
        (first_survivors, "first-pass scientific synthesis survivors"),
        (first_filter, "first-pass scientific synthesis filter report"),
        (repaired, "bounded repaired synthesis portfolio"),
        (repair_report_path, "bounded repair report"),
        (provider_plan, "literature provider plan"),
    ]:
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    context = HypothesisContext.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )
    repaired_portfolio = HypothesisPortfolio.model_validate_json(
        repaired.read_text(encoding="utf-8")
    )

    manifest_path = run / "scientific_synthesis_bounded_n10_e2e_manifest.json"
    manifest = {
        "schema_version": "scientific-synthesis-bounded-n10-e2e-manifest-v1",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "status": "planned" if args.dry_run else "running",
        "continuation_depth": 1,
        "further_repair_allowed": False,
        "repaired_candidate_count": len(repaired_portfolio.hypotheses),
        "conditional_is_positive": False,
        "fresh_n10_required": True,
        "legacy_production_selection_mutated": False,
        "canonical_graph_mutated": False,
        "failure": None,
    }
    _write(manifest_path, manifest)

    if not repaired_portfolio.hypotheses:
        manifest["status"] = "abstained_no_repaired_candidates"
        manifest["finished_at_utc"] = _now()
        _write(manifest_path, manifest)
        print("Bounded scientific synthesis N10 abstained: no repaired candidates.")
        return 0

    if not args.model or not args.critic_model:
        raise SystemExit("--model and --critic-model are required")

    prefix = run / "scientific_synthesis_bounded_n10_external"
    query_plan = Path(str(prefix) + ".claims_queries.json")
    prior_art = Path(str(prefix) + ".prior_art.json")
    external_report = Path(str(prefix) + ".report.json")
    intake = run / "scientific_synthesis_bounded_n10_intake.json"
    full = run / "scientific_synthesis_bounded_n10_full.json"
    candidate_gate = run / "scientific_synthesis_bounded_n10_candidate_gate.json"
    production_gate = run / "scientific_synthesis_bounded_n10_production_gate.json"
    repaired_survivors = run / "scientific_synthesis_bounded_n10_repaired_survivors.portfolio.json"
    repaired_filter = run / "scientific_synthesis_bounded_n10_repaired_filter_report.json"
    scientific_final = run / "scientific_synthesis_bounded_n10_survivors.portfolio.json"
    scientific_round_report_path = run / "scientific_synthesis_bounded_n10_round_report.json"
    merged = run / "scientific_merged_bounded_n10_survivors.portfolio.json"
    merged_report_path = run / "scientific_merged_bounded_n10_authority_report.json"
    semantic_prefix = run / "scientific_merged_bounded_final_semantic"
    semantic_review = Path(str(semantic_prefix) + ".review.json")
    feasibility_dir = run / "scientific_merged_bounded_final_feasibility"

    external_cmd = [
        "-m", "scripts.discovery.run_external_novelty",
        "--portfolio", str(repaired),
        "--domain-profile", context.domain_profile_id,
        "--model", args.model,
        "--api-key-env", args.api_key_env,
        "--provider-plan", str(provider_plan),
        "--results-per-query", str(args.results_per_query),
        "--max-ranked-works", str(args.max_ranked_works),
        "--output-prefix", str(prefix),
    ]
    if args.base_url:
        external_cmd += ["--base-url", args.base_url]
    if args.save_prompts:
        external_cmd += ["--save-prompts"]

    intake_cmd = [
        "-m", "scripts.discovery.build_nonobviousness_shadow",
        "--query-plan", str(query_plan),
        "--external-report", str(external_report),
        "--portfolio", str(repaired),
        "--output", str(intake),
    ]
    full_cmd = [
        "-m", "scripts.discovery.run_nonobviousness_full_shadow",
        "--query-plan", str(query_plan),
        "--external-report", str(external_report),
        "--external-prior-art", str(prior_art),
        "--portfolio", str(repaired),
        "--hypothesis-context", str(context_path),
        "--intake-shadow", str(intake),
        "--provider-plan", str(provider_plan),
        "--domain-profile", context.domain_profile_id,
        "--model", args.critic_model,
        "--api-key-env", args.api_key_env,
        "--results-per-query", str(args.results_per_query),
        "--max-ranked-works", str(args.max_ranked_works),
        "--output", str(full),
    ]
    if args.base_url:
        full_cmd += ["--base-url", args.base_url]

    candidate_gate_cmd = [
        "-m", "scripts.discovery.build_nonobviousness_production_gate_v2_candidate",
        "--query-plan", str(query_plan),
        "--intake-shadow", str(intake),
        "--full-shadow", str(full),
        "--output", str(candidate_gate),
    ]
    production_gate_cmd = [
        "-m", "scripts.discovery.build_scientific_synthesis_n10_production_gate",
        "--candidate-gate", str(candidate_gate),
        "--output", str(production_gate),
    ]
    filter_cmd = [
        "-m", "scripts.discovery.build_scientific_synthesis_n10_survivors",
        "--portfolio", str(repaired),
        "--production-gate", str(production_gate),
        "--output", str(repaired_survivors),
        "--report-output", str(repaired_filter),
    ]

    commands = [
        ("fresh_external_novelty_depth1", external_cmd, [query_plan, prior_art, external_report]),
        ("fresh_n10_intake_depth1", intake_cmd, [intake]),
        ("fresh_n10_full_depth1", full_cmd, [full]),
        ("fresh_n10_candidate_gate_depth1", candidate_gate_cmd, [candidate_gate]),
        ("fresh_n10_synthesis_authority_depth1", production_gate_cmd, [production_gate]),
        ("fresh_n10_filter_depth1", filter_cmd, [repaired_survivors, repaired_filter]),
    ]
    manifest["planned_stages"] = [
        {"name": name, "argv": argv}
        for name, argv, _ in commands
    ]
    _write(manifest_path, manifest)

    print("Scientific synthesis bounded strict-N10 E2E")
    print("Repaired candidates:", len(repaired_portfolio.hypotheses))
    print("Continuation depth: 1")
    print("Further repair allowed: false")
    print("CONDITIONAL is positive: false")
    print("Fresh N10 required: true")

    if args.dry_run:
        for name, argv, _ in commands:
            print(f"  {name}:")
            print("    $", sys.executable, *argv)
        print("Execution performed: false")
        print("Manifest:", manifest_path)
        return 0

    try:
        for name, argv, expected in commands:
            _run(name, argv, expected)

        initial_portfolio = HypothesisPortfolio.model_validate_json(
            initial.read_text(encoding="utf-8")
        )
        first_survivor_portfolio = HypothesisPortfolio.model_validate_json(
            first_survivors.read_text(encoding="utf-8")
        )
        first_filter_report = ScientificSynthesisN10FilterReport.model_validate_json(
            first_filter.read_text(encoding="utf-8")
        )
        repair_report = ScientificSynthesisRepairReport.model_validate_json(
            repair_report_path.read_text(encoding="utf-8")
        )
        repaired_survivor_portfolio = HypothesisPortfolio.model_validate_json(
            repaired_survivors.read_text(encoding="utf-8")
        )
        repaired_filter_report = ScientificSynthesisN10FilterReport.model_validate_json(
            repaired_filter.read_text(encoding="utf-8")
        )

        scientific_survivors, round_report = merge_scientific_synthesis_n10_rounds(
            context=context,
            initial_portfolio=initial_portfolio,
            first_pass_survivors=first_survivor_portfolio,
            first_pass_filter=first_filter_report,
            repair_report=repair_report,
            repaired_portfolio=repaired_portfolio,
            repaired_pass_survivors=repaired_survivor_portfolio,
            repaired_pass_filter=repaired_filter_report,
        )
        _write(scientific_final, scientific_survivors)
        _write(scientific_round_report_path, round_report)

        merged_portfolio, merged_report = (
            merge_legacy_with_bounded_scientific_survivors(
                run_dir=run,
                context=context,
                scientific_survivors=scientific_survivors,
                scientific_report=round_report,
            )
        )
        _write(merged, merged_portfolio)
        _write(merged_report_path, merged_report)

        manifest["repaired_pass_survivor_count"] = (
            round_report.repaired_pass_survivor_count
        )
        manifest["scientific_final_survivor_count"] = (
            round_report.final_scientific_survivor_count
        )
        manifest["legacy_survivor_count"] = merged_report.legacy_survivor_count
        manifest["merged_survivor_count"] = merged_report.merged_survivor_count

        print()
        print("Bounded N10 authority merge")
        print("First-pass scientific survivors:", round_report.first_pass_survivor_count)
        print("Repaired-pass scientific survivors:", round_report.repaired_pass_survivor_count)
        print("Final scientific survivors:", round_report.final_scientific_survivor_count)
        print("Legacy survivors:", merged_report.legacy_survivor_count)
        print("Merged survivors:", merged_report.merged_survivor_count)

        if not merged_portfolio.hypotheses:
            manifest["status"] = "abstained_no_bounded_strict_n10_survivors"
            manifest["finished_at_utc"] = _now()
            _write(manifest_path, manifest)
            print("Disposition: abstained_no_bounded_strict_n10_survivors")
            return 0

        semantic_cmd = [
            "-m", "scripts.discovery.run_hypothesis_semantic_critic",
            "--context", str(context_path),
            "--portfolio", str(merged),
            "--model", args.critic_model,
            "--api-key-env", args.api_key_env,
            "--output-prefix", str(semantic_prefix),
            "--save-prompt",
        ]
        if args.base_url:
            semantic_cmd += ["--base-url", args.base_url]
        _run("bounded_final_semantic", semantic_cmd, [semantic_review])

        profile = get_domain_profile(context.domain_profile_id)
        feasibility = resolve_feasibility_adapter(profile)
        if feasibility is not None:
            feasibility_cmd = [
                "-m", "scripts.discovery.run_feasibility_e2e",
                "--context", str(context_path),
                "--domain-profile", context.domain_profile_id,
                "--portfolio", str(merged),
                "--semantic-review", str(semantic_review),
                "--output-dir", str(feasibility_dir),
            ]
            _run(
                "bounded_final_feasibility",
                feasibility_cmd,
                [feasibility_dir / "manifest.json"],
            )
            manifest["feasibility_status"] = "complete"
        else:
            manifest["feasibility_status"] = "not_declared_for_domain"

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _now()
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["finished_at_utc"] = _now()
    manifest["scientific_survivor_portfolio"] = str(scientific_final)
    manifest["merged_portfolio"] = str(merged)
    manifest["final_semantic_review"] = str(semantic_review)
    manifest["legacy_production_selection_mutated"] = False
    manifest["canonical_graph_mutated"] = False
    _write(manifest_path, manifest)

    print()
    print("Scientific synthesis bounded strict-N10 E2E complete")
    print("Final scientific survivors:", manifest["scientific_final_survivor_count"])
    print("Merged survivors:", manifest["merged_survivor_count"])
    print("Further repair allowed: false")
    print("Legacy production selection mutated: false")
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
