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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
            "Run fresh external novelty + role-aware N10 v2 over pre-N10 scientific "
            "synthesis candidates, merge first-pass positive synthesis survivors with "
            "strict legacy N10 survivors, then run sidecar final semantic/feasibility."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--synthesis-portfolio", type=Path, default=None)
    parser.add_argument("--provider-plan", type=Path, default=None)
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
    run_dir = args.run_dir.expanduser().resolve()
    context_path = run_dir / "hypothesis.context.json"
    synthesis_portfolio_path = (
        args.synthesis_portfolio.expanduser().resolve()
        if args.synthesis_portfolio is not None
        else run_dir / "scientific_synthesis_pre_n10.portfolio.json"
    )
    provider_plan = (
        args.provider_plan.expanduser().resolve()
        if args.provider_plan is not None
        else run_dir / "literature_provider_plan.json"
    )
    for path, label in [
        (context_path, "hypothesis context"),
        (synthesis_portfolio_path, "pre-N10 synthesis portfolio"),
        (provider_plan, "literature provider plan"),
    ]:
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    context = HypothesisContext.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )
    synthesis_portfolio = HypothesisPortfolio.model_validate_json(
        synthesis_portfolio_path.read_text(encoding="utf-8")
    )

    manifest_path = run_dir / "scientific_synthesis_n10_e2e_manifest.json"
    manifest = {
        "schema_version": "scientific-synthesis-n10-e2e-manifest-v1",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "status": "planned" if args.dry_run else "running",
        "source_synthesis_portfolio": str(synthesis_portfolio_path),
        "source_synthesis_candidate_count": len(synthesis_portfolio.hypotheses),
        "positive_authority_contract": (
            "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS"
        ),
        "conditional_is_positive": False,
        "bounded_continuation_offered_to_synthesis": False,
        "repair_opportunity_parity_with_legacy": False,
        "legacy_production_selection_mutated": False,
        "canonical_graph_mutated": False,
        "failure": None,
    }
    _write(manifest_path, manifest)

    if not synthesis_portfolio.hypotheses:
        manifest["status"] = "abstained_no_synthesis_candidates"
        manifest["finished_at_utc"] = _now()
        _write(manifest_path, manifest)
        print("Scientific synthesis N10 E2E abstained: no synthesis candidates.")
        return 0

    if not args.model or not args.critic_model:
        raise SystemExit(
            "--model and --critic-model are required unless configured by environment"
        )

    prefix = run_dir / "scientific_synthesis_n10_external"
    query_plan = Path(str(prefix) + ".claims_queries.json")
    prior_art = Path(str(prefix) + ".prior_art.json")
    external_report = Path(str(prefix) + ".report.json")
    intake = run_dir / "scientific_synthesis_n10_intake.json"
    full = run_dir / "scientific_synthesis_n10_full.json"
    candidate_gate = run_dir / "scientific_synthesis_n10_candidate_gate.json"
    production_gate = run_dir / "scientific_synthesis_n10_production_gate.json"
    survivors = run_dir / "scientific_synthesis_n10_survivors.portfolio.json"
    filter_report = run_dir / "scientific_synthesis_n10_filter_report.json"
    merged = run_dir / "scientific_merged_n10_survivors.portfolio.json"
    merge_report = run_dir / "scientific_merged_n10_authority_report.json"
    semantic_prefix = run_dir / "scientific_merged_final_semantic"
    semantic_review = Path(str(semantic_prefix) + ".review.json")
    feasibility_dir = run_dir / "scientific_merged_final_feasibility"

    external_cmd = [
        "-m", "scripts.discovery.run_external_novelty",
        "--portfolio", str(synthesis_portfolio_path),
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
        "--portfolio", str(synthesis_portfolio_path),
        "--output", str(intake),
    ]
    full_cmd = [
        "-m", "scripts.discovery.run_nonobviousness_full_shadow",
        "--query-plan", str(query_plan),
        "--external-report", str(external_report),
        "--external-prior-art", str(prior_art),
        "--portfolio", str(synthesis_portfolio_path),
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
        "--portfolio", str(synthesis_portfolio_path),
        "--production-gate", str(production_gate),
        "--output", str(survivors),
        "--report-output", str(filter_report),
    ]
    merge_cmd = [
        "-m", "scripts.discovery.merge_legacy_and_scientific_n10_survivors",
        "--run-dir", str(run_dir),
        "--context", str(context_path),
        "--scientific-source-portfolio", str(synthesis_portfolio_path),
        "--scientific-survivors", str(survivors),
        "--output", str(merged),
        "--report-output", str(merge_report),
    ]

    commands = [
        ("external_novelty", external_cmd, [query_plan, prior_art, external_report]),
        ("n10_intake", intake_cmd, [intake]),
        ("n10_full_shadow", full_cmd, [full]),
        ("n10_candidate_gate", candidate_gate_cmd, [candidate_gate]),
        ("n10_synthesis_authority", production_gate_cmd, [production_gate]),
        ("n10_synthesis_filter", filter_cmd, [survivors, filter_report]),
        ("strict_survivor_merge", merge_cmd, [merged, merge_report]),
    ]
    manifest["planned_stages"] = [
        {"name": name, "argv": argv}
        for name, argv, _ in commands
    ]
    _write(manifest_path, manifest)

    print("Scientific synthesis strict-N10 E2E")
    print("Synthesis candidates:", len(synthesis_portfolio.hypotheses))
    print("Synthesis bounded continuation offered: false")
    print("CONDITIONAL is positive: false")
    print("Legacy production selection mutated: false")

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

        merged_portfolio = HypothesisPortfolio.model_validate_json(
            merged.read_text(encoding="utf-8")
        )
        filter_payload = _load(filter_report)
        merge_payload = _load(merge_report)
        manifest["scientific_synthesis_survivor_count"] = int(
            filter_payload.get("survivor_count", 0)
        )
        manifest["legacy_survivor_count"] = int(
            merge_payload.get("legacy_survivor_count", 0)
        )
        manifest["merged_survivor_count"] = len(merged_portfolio.hypotheses)

        if not merged_portfolio.hypotheses:
            manifest["status"] = "abstained_no_strict_n10_survivors"
            manifest["finished_at_utc"] = _now()
            _write(manifest_path, manifest)
            print()
            print("No legacy or scientific synthesis hypotheses survived strict N10.")
            print("Disposition: abstained_no_strict_n10_survivors")
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
        _run("final_semantic", semantic_cmd, [semantic_review])

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
                "final_feasibility",
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
    manifest["merged_portfolio"] = str(merged)
    manifest["final_semantic_review"] = str(semantic_review)
    manifest["legacy_production_selection_mutated"] = False
    manifest["canonical_graph_mutated"] = False
    _write(manifest_path, manifest)

    print()
    print("Scientific synthesis strict-N10 E2E complete")
    print("Scientific synthesis survivors:", manifest["scientific_synthesis_survivor_count"])
    print("Legacy survivors:", manifest["legacy_survivor_count"])
    print("Merged survivors:", manifest["merged_survivor_count"])
    print("Final semantic review:", semantic_review)
    print("Feasibility:", manifest["feasibility_status"])
    print("Legacy production selection mutated: false")
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
