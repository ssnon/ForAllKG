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
from pipeline_core.discovery.reframing.atomic_synthesis_n10_smoke import (
    ready_for_closure_count,
    summarize_n9_states,
)
from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY,
    SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_HARD_FILTER,
    SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODES,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
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
        raise RuntimeError(
            f"{name} completed without expected outputs: {missing}"
        )


def _existing_provider_plan(run: Path) -> Path | None:
    for name in (
        "scientific_synthesis_bounded_n10_external.provider_plan.json",
        "scientific_synthesis_n10_external.provider_plan.json",
        "literature_provider_plan.json",
    ):
        path = run / name
        if path.is_file():
            return path
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh prior-art retrieval/review and the existing strict N9/N10 "
            "chain over an atomic cross-lane synthesis query plan. The query plan "
            "is reused exactly, so external-novelty LLM atomic re-decomposition is "
            "bypassed while retrieval, review, closure, adjudication and production "
            "gate semantics remain unchanged. Authority can operate as a "
            "legacy hard filter or as certification-only, which preserves all "
            "scientific candidates while emitting a separate certified subset."
        )
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--portfolio", type=Path, default=None)
    parser.add_argument("--query-plan", type=Path, default=None)
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
    parser.add_argument(
        "--authority-mode",
        choices=SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODES,
        default=SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY,
        help=(
            "certification_only preserves all scientific candidates and "
            "separates novelty certification; hard_filter preserves the "
            "legacy deletion semantics."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    run = args.run_dir.expanduser().resolve()
    context_path = run / "hypothesis.context.json"
    portfolio_path = (
        args.portfolio.expanduser().resolve()
        if args.portfolio is not None
        else run / "scientific_atomic_cross_lane.portfolio.json"
    )
    query_plan_path = (
        args.query_plan.expanduser().resolve()
        if args.query_plan is not None
        else run / "scientific_atomic_cross_lane.query_plan.json"
    )

    for path, label in (
        (context_path, "hypothesis context"),
        (portfolio_path, "atomic synthesis portfolio"),
        (query_plan_path, "atomic synthesis query plan"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    context = HypothesisContext.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )

    if not portfolio.hypotheses:
        print("Atomic synthesis strict-N10 smoke abstained: no atomic hypotheses.")
        return 0

    if not args.model or not args.critic_model:
        raise SystemExit("--model and --critic-model are required")

    provider_plan = (
        args.provider_plan.expanduser().resolve()
        if args.provider_plan is not None
        else _existing_provider_plan(run)
    )

    manifest_path = run / "scientific_atomic_n10_e2e_manifest.json"
    manifest = {
        "schema_version": "scientific-atomic-n10-e2e-manifest-v1",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "status": "planned" if args.dry_run else "running",
        "source_portfolio": str(portfolio_path),
        "source_query_plan": str(query_plan_path),
        "source_hypothesis_count": len(portfolio.hypotheses),
        "external_novelty_llm_redecomposition_performed": False,
        "precomputed_atomic_query_plan_reused": True,
        "n9_contract_changed": False,
        "n10_contract_changed": False,
        "authority_mode": args.authority_mode,
        "scientific_candidate_retention_is_not_novelty_authority": True,
        "semantic_or_feasibility_can_upgrade_novelty": False,
        "conditional_is_positive": False,
        "bounded_repair_allowed": False,
        "legacy_production_selection_mutated": False,
        "canonical_graph_mutated": False,
        "failure": None,
    }
    _write(manifest_path, manifest)

    prefix = run / "scientific_atomic_n10_external"
    external_plan = Path(str(prefix) + ".claims_queries.json")
    prior_art = Path(str(prefix) + ".prior_art.json")
    external_report = Path(str(prefix) + ".report.json")
    generated_provider_plan = Path(str(prefix) + ".provider_plan.json")
    intake = run / "scientific_atomic_n10_intake.json"
    full = run / "scientific_atomic_n10_full.json"
    candidate_gate = run / "scientific_atomic_n10_candidate_gate.json"
    production_gate = run / "scientific_atomic_n10_production_gate.json"
    survivors = run / "scientific_atomic_n10_survivors.portfolio.json"
    filter_report = run / "scientific_atomic_n10_filter_report.json"
    merged = run / "scientific_atomic_merged_n10_survivors.portfolio.json"
    merge_report = run / "scientific_atomic_merged_n10_authority_report.json"

    candidate_portfolio = run / "scientific_atomic_n10_candidates.portfolio.json"
    certification_report = run / "scientific_atomic_n10_certification.report.json"
    certified_portfolio = run / "scientific_atomic_n10_certified.portfolio.json"
    merged_candidates = run / "scientific_atomic_merged_candidates.portfolio.json"
    merged_certified = run / "scientific_atomic_merged_certified.portfolio.json"
    certification_merge_report = (
        run / "scientific_atomic_merged_certification_authority_report.json"
    )

    semantic_prefix = run / "scientific_atomic_merged_final_semantic"
    semantic_review = Path(str(semantic_prefix) + ".review.json")
    feasibility_dir = run / "scientific_atomic_merged_final_feasibility"

    external_cmd = [
        "-m",
        "scripts.discovery.run_external_novelty",
        "--portfolio",
        str(portfolio_path),
        "--domain-profile",
        context.domain_profile_id,
        "--model",
        args.model,
        "--api-key-env",
        args.api_key_env,
        "--reuse-query-plan",
        str(query_plan_path),
        "--results-per-query",
        str(args.results_per_query),
        "--max-ranked-works",
        str(args.max_ranked_works),
        "--output-prefix",
        str(prefix),
    ]
    if provider_plan is not None:
        external_cmd += ["--provider-plan", str(provider_plan)]
    else:
        external_cmd += ["--providers", "auto"]
    if args.base_url:
        external_cmd += ["--base-url", args.base_url]
    if args.save_prompts:
        external_cmd += ["--save-prompts"]

    intake_cmd = [
        "-m",
        "scripts.discovery.build_nonobviousness_shadow",
        "--query-plan",
        str(external_plan),
        "--external-report",
        str(external_report),
        "--portfolio",
        str(portfolio_path),
        "--output",
        str(intake),
    ]

    # run_external_novelty always materializes the provider plan for a fresh
    # retrieval even when the query plan itself is reused.
    full_cmd = [
        "-m",
        "scripts.discovery.run_nonobviousness_full_shadow",
        "--query-plan",
        str(external_plan),
        "--external-report",
        str(external_report),
        "--external-prior-art",
        str(prior_art),
        "--portfolio",
        str(portfolio_path),
        "--hypothesis-context",
        str(context_path),
        "--intake-shadow",
        str(intake),
        "--provider-plan",
        str(generated_provider_plan),
        "--domain-profile",
        context.domain_profile_id,
        "--model",
        args.critic_model,
        "--api-key-env",
        args.api_key_env,
        "--results-per-query",
        str(args.results_per_query),
        "--max-ranked-works",
        str(args.max_ranked_works),
        "--output",
        str(full),
    ]
    if args.base_url:
        full_cmd += ["--base-url", args.base_url]

    candidate_gate_cmd = [
        "-m",
        "scripts.discovery.build_nonobviousness_production_gate_v2_candidate",
        "--query-plan",
        str(external_plan),
        "--intake-shadow",
        str(intake),
        "--full-shadow",
        str(full),
        "--output",
        str(candidate_gate),
    ]
    production_gate_cmd = [
        "-m",
        "scripts.discovery.build_scientific_synthesis_n10_production_gate",
        "--candidate-gate",
        str(candidate_gate),
        "--output",
        str(production_gate),
    ]
    filter_cmd = [
        "-m",
        "scripts.discovery.build_scientific_synthesis_n10_survivors",
        "--portfolio",
        str(portfolio_path),
        "--production-gate",
        str(production_gate),
        "--output",
        str(survivors),
        "--report-output",
        str(filter_report),
    ]
    merge_cmd = [
        "-m",
        "scripts.discovery.merge_legacy_and_scientific_n10_survivors",
        "--run-dir",
        str(run),
        "--context",
        str(context_path),
        "--scientific-source-portfolio",
        str(portfolio_path),
        "--scientific-survivors",
        str(survivors),
        "--output",
        str(merged),
        "--report-output",
        str(merge_report),
    ]

    certification_cmd = [
        "-m",
        "scripts.discovery.build_scientific_synthesis_n10_certification",
        "--portfolio",
        str(portfolio_path),
        "--production-gate",
        str(production_gate),
        "--output-candidate-portfolio",
        str(candidate_portfolio),
        "--output-certification-report",
        str(certification_report),
        "--output-certified-portfolio",
        str(certified_portfolio),
    ]
    certification_merge_cmd = [
        "-m",
        "scripts.discovery.merge_legacy_and_scientific_n10_certification",
        "--run-dir",
        str(run),
        "--context",
        str(context_path),
        "--scientific-candidates",
        str(candidate_portfolio),
        "--scientific-certified",
        str(certified_portfolio),
        "--certification-report",
        str(certification_report),
        "--output-candidate-portfolio",
        str(merged_candidates),
        "--output-certified-portfolio",
        str(merged_certified),
        "--report-output",
        str(certification_merge_report),
    ]

    commands = [
        (
            "fresh_external_novelty_precomputed_atomic_plan",
            external_cmd,
            [
                external_plan,
                prior_art,
                external_report,
                generated_provider_plan,
            ],
        ),
        ("strict_n9_intake", intake_cmd, [intake]),
        ("strict_n10_full", full_cmd, [full]),
        ("strict_n10_candidate_gate", candidate_gate_cmd, [candidate_gate]),
        ("strict_n10_scientific_authority", production_gate_cmd, [production_gate]),
    ]
    if args.authority_mode == SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_HARD_FILTER:
        commands.extend(
            [
                ("strict_n10_filter", filter_cmd, [survivors, filter_report]),
                (
                    "strict_legacy_scientific_merge",
                    merge_cmd,
                    [merged, merge_report],
                ),
            ]
        )
    else:
        commands.extend(
            [
                (
                    "n10_certification_only_split",
                    certification_cmd,
                    [
                        candidate_portfolio,
                        certification_report,
                        certified_portfolio,
                    ],
                ),
                (
                    "certification_only_legacy_scientific_merge",
                    certification_merge_cmd,
                    [
                        merged_candidates,
                        merged_certified,
                        certification_merge_report,
                    ],
                ),
            ]
        )
    manifest["planned_stages"] = [
        {"name": name, "argv": argv}
        for name, argv, _ in commands
    ]
    _write(manifest_path, manifest)

    print("Atomic scientific synthesis strict-N10 E2E")
    print("Hypotheses:", len(portfolio.hypotheses))
    print("External-novelty LLM re-decomposition performed: false")
    print("Precomputed atomic query plan reused: true")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Authority mode:", args.authority_mode)
    print("Bounded repair allowed: false")

    if args.dry_run:
        for name, argv, _ in commands:
            print(f"  {name}:")
            print("    $", sys.executable, *argv)
        print("Execution performed: false")
        print("Manifest:", manifest_path)
        return 0

    try:
        _run(
            "fresh_external_novelty_precomputed_atomic_plan",
            external_cmd,
            [external_plan, prior_art, external_report, generated_provider_plan],
        )
        _run("strict_n9_intake", intake_cmd, [intake])

        intake_payload = _load(intake)
        state_counts = summarize_n9_states(intake_payload)
        ready_count = ready_for_closure_count(intake_payload)
        manifest["n9_state_counts"] = state_counts
        manifest["n9_ready_for_closure_count"] = ready_count

        print()
        print("Atomic N9 specification result")
        print("States:", state_counts)
        print("READY_FOR_CLOSURE:", ready_count)

        _run("strict_n10_full", full_cmd, [full])
        _run("strict_n10_candidate_gate", candidate_gate_cmd, [candidate_gate])
        _run(
            "strict_n10_scientific_authority",
            production_gate_cmd,
            [production_gate],
        )
        if (
            args.authority_mode
            == SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_HARD_FILTER
        ):
            _run("strict_n10_filter", filter_cmd, [survivors, filter_report])
            _run(
                "strict_legacy_scientific_merge",
                merge_cmd,
                [merged, merge_report],
            )

            survivor_portfolio = HypothesisPortfolio.model_validate_json(
                survivors.read_text(encoding="utf-8")
            )
            merged_portfolio = HypothesisPortfolio.model_validate_json(
                merged.read_text(encoding="utf-8")
            )
            merge_payload = _load(merge_report)

            manifest["atomic_scientific_survivor_count"] = len(
                survivor_portfolio.hypotheses
            )
            manifest["legacy_survivor_count"] = int(
                merge_payload.get("legacy_survivor_count", 0)
            )
            manifest["merged_survivor_count"] = len(
                merged_portfolio.hypotheses
            )
            downstream_portfolio_path = merged

            print()
            print("Strict N10 hard-filter result")
            print(
                "Atomic scientific survivors:",
                manifest["atomic_scientific_survivor_count"],
            )
            print("Legacy survivors:", manifest["legacy_survivor_count"])
            print("Merged survivors:", manifest["merged_survivor_count"])

            if not merged_portfolio.hypotheses:
                manifest["status"] = "abstained_no_strict_n10_survivors"
                manifest["finished_at_utc"] = _now()
                _write(manifest_path, manifest)
                print("Disposition: abstained_no_strict_n10_survivors")
                return 0

        else:
            _run(
                "n10_certification_only_split",
                certification_cmd,
                [
                    candidate_portfolio,
                    certification_report,
                    certified_portfolio,
                ],
            )
            _run(
                "certification_only_legacy_scientific_merge",
                certification_merge_cmd,
                [
                    merged_candidates,
                    merged_certified,
                    certification_merge_report,
                ],
            )

            certification_payload = _load(certification_report)
            merge_payload = _load(certification_merge_report)
            merged_portfolio = HypothesisPortfolio.model_validate_json(
                merged_candidates.read_text(encoding="utf-8")
            )
            merged_certified_portfolio = HypothesisPortfolio.model_validate_json(
                merged_certified.read_text(encoding="utf-8")
            )

            manifest["atomic_scientific_candidate_count"] = int(
                certification_payload.get("candidate_count", 0)
            )
            manifest["atomic_scientific_certified_count"] = int(
                certification_payload.get("certified_count", 0)
            )
            manifest["atomic_scientific_unresolved_count"] = int(
                certification_payload.get("unresolved_count", 0)
            )
            manifest["atomic_scientific_rejected_count"] = int(
                certification_payload.get("rejected_count", 0)
            )
            manifest["legacy_survivor_count"] = int(
                merge_payload.get("legacy_strict_authority_count", 0)
            )
            manifest["merged_candidate_count"] = len(
                merged_portfolio.hypotheses
            )
            manifest["merged_certified_count"] = len(
                merged_certified_portfolio.hypotheses
            )
            downstream_portfolio_path = merged_candidates

            print()
            print("N10 certification-only result")
            print(
                "Atomic scientific candidates:",
                manifest["atomic_scientific_candidate_count"],
            )
            print(
                "Novelty-certified:",
                manifest["atomic_scientific_certified_count"],
            )
            print(
                "Novelty-unresolved:",
                manifest["atomic_scientific_unresolved_count"],
            )
            print(
                "Novelty-rejected:",
                manifest["atomic_scientific_rejected_count"],
            )
            print("Legacy strict hypotheses:", manifest["legacy_survivor_count"])
            print("Merged discovery candidates:", manifest["merged_candidate_count"])
            print("Merged novelty-certified:", manifest["merged_certified_count"])
            print("Candidate retention is novelty authority: false")

            if not merged_portfolio.hypotheses:
                manifest["status"] = "abstained_no_discovery_candidates"
                manifest["finished_at_utc"] = _now()
                _write(manifest_path, manifest)
                print("Disposition: abstained_no_discovery_candidates")
                return 0

        semantic_cmd = [
            "-m",
            "scripts.discovery.run_hypothesis_semantic_critic",
            "--context",
            str(context_path),
            "--portfolio",
            str(downstream_portfolio_path),
            "--model",
            args.critic_model,
            "--api-key-env",
            args.api_key_env,
            "--output-prefix",
            str(semantic_prefix),
            "--save-prompt",
        ]
        if args.base_url:
            semantic_cmd += ["--base-url", args.base_url]
        _run("final_semantic", semantic_cmd, [semantic_review])

        profile = get_domain_profile(context.domain_profile_id)
        feasibility = resolve_feasibility_adapter(profile)
        if feasibility is not None:
            feasibility_cmd = [
                "-m",
                "scripts.discovery.run_feasibility_e2e",
                "--context",
                str(context_path),
                "--domain-profile",
                context.domain_profile_id,
                "--portfolio",
                str(downstream_portfolio_path),
                "--semantic-review",
                str(semantic_review),
                "--output-dir",
                str(feasibility_dir),
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
    manifest["downstream_candidate_portfolio"] = str(
        downstream_portfolio_path
    )
    if (
        args.authority_mode
        == SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY
    ):
        manifest["novelty_certified_portfolio"] = str(merged_certified)
        manifest["novelty_certification_report"] = str(
            certification_report
        )
    else:
        manifest["novelty_certified_portfolio"] = str(merged)
    manifest["final_semantic_review"] = str(semantic_review)
    manifest["legacy_production_selection_mutated"] = False
    manifest["canonical_graph_mutated"] = False
    _write(manifest_path, manifest)

    print()
    print("Atomic scientific synthesis N10 E2E complete")
    print("Authority mode:", args.authority_mode)
    print("N9 READY_FOR_CLOSURE:", manifest["n9_ready_for_closure_count"])
    if (
        args.authority_mode
        == SCIENTIFIC_SYNTHESIS_N10_AUTHORITY_MODE_CERTIFICATION_ONLY
    ):
        print(
            "Scientific candidates preserved:",
            manifest["atomic_scientific_candidate_count"],
        )
        print(
            "Novelty-certified scientific candidates:",
            manifest["atomic_scientific_certified_count"],
        )
        print(
            "Novelty-unresolved scientific candidates:",
            manifest["atomic_scientific_unresolved_count"],
        )
        print(
            "Novelty-rejected scientific candidates:",
            manifest["atomic_scientific_rejected_count"],
        )
        print("Merged discovery candidates:", manifest["merged_candidate_count"])
        print("Merged novelty-certified:", manifest["merged_certified_count"])
    else:
        print(
            "Atomic scientific survivors:",
            manifest["atomic_scientific_survivor_count"],
        )
        print("Merged survivors:", manifest["merged_survivor_count"])
    print("Legacy production selection mutated: false")
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
