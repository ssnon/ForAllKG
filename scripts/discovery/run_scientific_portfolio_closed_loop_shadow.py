
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.scientific_portfolio_closed_loop import (
    build_effective_portfolio,
    compile_residual_epistemic_state,
    summarize_closed_loop,
)
from pipeline_core.discovery.sers_closed_loop_decision_consolidation_v2 import (
    consolidate_post_verification_decisions,
)
from pipeline_core.discovery.sers_novelty_feedback_closed_loop import (
    build_feedback_plan,
    load_context_source,
    run_feedback_generation,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(
    label: str,
    cmd: list[str],
    out: Path,
    *,
    allow_code_2: bool = False,
) -> int:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    safe = "_".join(label.lower().split())
    (out / f"{safe}.stdout.txt").write_text(
        result.stdout or "", encoding="utf-8"
    )
    (out / f"{safe}.stderr.txt").write_text(
        result.stderr or "", encoding="utf-8"
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0 and not (
        allow_code_2 and result.returncode == 2
    ):
        raise RuntimeError(
            f"{label} failed with return code {result.returncode}"
        )
    return result.returncode


def model_args(args) -> list[str]:
    result = [
        "--model", args.critic_model,
        "--api-key-env", args.api_key_env,
    ]
    if args.base_url:
        result += ["--base-url", args.base_url]
    return result


def lower_order(
    *,
    args,
    portfolio: Path,
    query_plan: Path,
    external_report: Path,
    prefix: Path,
    results_per_query: int | None = None,
):
    run(
        f"{prefix.parent.name} lower-order saturation",
        [
            sys.executable,
            "-m",
            "scripts.discovery.run_lower_order_prior_art_saturation_shadow",
            "--portfolio", str(portfolio),
            "--query-plan", str(query_plan),
            "--external-report", str(external_report),
            "--provider-plan", str(args.provider_plan),
            "--domain-profile", args.domain_profile,
            *model_args(args),
            "--results-per-query",
            str(
                args.results_per_query
                if results_per_query is None
                else results_per_query
            ),
            "--max-ranked-works", str(args.max_ranked_works),
            "--parse-retries", str(args.parse_retries),
            "--output-prefix", str(prefix),
            "--save-prompts",
        ],
        args.output_dir,
    )
    return (
        prefix.parent / f"{prefix.name}.report.json",
        prefix.parent / f"{prefix.name}.claim_reviews.json",
        prefix.parent / f"{prefix.name}.prior_art.json",
    )


def run_topology(
    *,
    args,
    query_plan: Path,
    saturation_report: Path,
    external_report: Path,
    source_binding: Path,
    output: Path,
):
    run(
        f"{output.parent.name} source-bound topology",
        [
            sys.executable,
            "-m",
            "scripts.discovery.run_generic_source_bound_topology_shadow",
            "--query-plan", str(query_plan),
            "--saturation-report", str(saturation_report),
            "--external-report", str(external_report),
            "--source-binding", str(source_binding),
            "--output", str(output),
        ],
        args.output_dir,
    )


def run_fulltext(
    *,
    args,
    query_plan: Path,
    reviews: Path,
    packet: Path,
    topology_report: Path,
    output: Path,
):
    run(
        f"{output.parent.name} bounded full-text escalation",
        [
            sys.executable,
            "-m",
            "scripts.discovery.run_fulltext_lower_order_escalation_shadow",
            "--query-plan", str(query_plan),
            "--claim-reviews", str(reviews),
            "--prior-art-packet", str(packet),
            "--topology-report", str(topology_report),
            "--output", str(output),
            "--acquisition-dir", str(output.parent / "fulltext_acquisition"),
            *model_args(args),
            "--max-candidates-per-claim",
            str(args.max_fulltext_candidates),
            "--max-excerpt-chars", str(args.max_excerpt_chars),
            "--parse-retries", str(args.parse_retries),
        ],
        args.output_dir,
    )


def run_aggregation(
    *,
    args,
    topology_report: Path,
    reviews: Path,
    fulltext_report: Path,
    output: Path,
):
    run(
        f"{output.parent.name} residual aggregation",
        [
            sys.executable,
            "-m",
            "scripts.discovery.run_residual_novelty_aggregation_shadow",
            "--topology-report", str(topology_report),
            "--claim-reviews", str(reviews),
            "--fulltext-escalation", str(fulltext_report),
            "--output", str(output),
        ],
        args.output_dir,
    )


def _derive_external_topology_completion_path(
    external_report: Path,
) -> Path:
    name = external_report.name
    suffix = ".report.json"
    if not name.endswith(suffix):
        raise ValueError(
            "cannot derive topology-completion artifact from external report: "
            + str(external_report)
        )
    return external_report.with_name(
        name[: -len(suffix)] + ".topology_completion_shadow.json"
    )


def _empty_gen1_seed_generation_report(
    gen0_portfolio: HypothesisPortfolio,
) -> dict[str, Any]:
    records = []
    for card in gen0_portfolio.hypotheses:
        hid = str(card.hypothesis_id)
        records.append(
            {
                "source_hypothesis_id": hid,
                "route": "EMPTY_GEN1_ADAPTIVE_SEED",
                "decision": "CARRIED_UNRESOLVED_GEN0_SEED",
                "generated_hypothesis_id": None,
                "reason_codes": [
                    "stage_775_generated_no_gen1_candidate",
                    "unresolved_gen0_preserved_for_adaptive_search",
                    "no_scientific_authority_created",
                ],
            }
        )
    return {
        "schema_version": "adaptive-discovery-generation-report-v1",
        "source_portfolio_id": str(gen0_portfolio.portfolio_id),
        "output_portfolio_id": str(gen0_portfolio.portfolio_id),
        "round_index": 1,
        "records": records,
        "decision_counts": {
            "CARRIED_UNRESOLVED_GEN0_SEED": len(records),
        },
        "route_counts": {
            "EMPTY_GEN1_ADAPTIVE_SEED": len(records),
        },
        "external_prior_art_as_positive_premise": False,
        "generation_authority_created": False,
        "empty_gen1_fallback_seed": True,
    }


def materialize_empty_gen1_adaptive_seed(
    *,
    output_dir: Path,
    gen0_portfolio: HypothesisPortfolio,
    external_query_plan: Path,
    external_report: Path,
    external_prior_art: Path,
    external_source_binding: Path,
    gen0_aggregation: Path,
) -> Path:
    completion = _derive_external_topology_completion_path(
        external_report
    )
    required = (
        external_query_plan,
        external_report,
        external_prior_art,
        external_source_binding,
        completion,
        gen0_aggregation,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "cannot materialize empty-Gen1 adaptive seed; missing "
            + repr(missing)
        )

    seed = output_dir / "adaptive_seed"
    gen1 = seed / "gen1"
    gen1.mkdir(parents=True, exist_ok=True)

    write(seed / "gen1.portfolio.json", gen0_portfolio)
    write(
        seed / "generation.report.json",
        _empty_gen1_seed_generation_report(gen0_portfolio),
    )

    empty_effective = gen0_portfolio.model_copy(
        update={
            "hypotheses": [],
            "abstention_reason": (
                "Stage-7.75 produced no effective Gen1 candidate; "
                "unresolved Gen0 is preserved only as adaptive-search seed."
            ),
        }
    )
    write(seed / "effective_gen1.portfolio.json", empty_effective)
    write(
        seed / "post_verification_decisions.json",
        {
            "schema_version": "empty-gen1-adaptive-seed-decisions-v1",
            "decision_counts": {},
            "rows": [],
            "empty_gen1_fallback_seed": True,
            "production_selection_changed": False,
        },
    )

    copies = {
        external_query_plan: gen1 / "external.claims_queries.json",
        external_report: gen1 / "external.report.json",
        external_prior_art: gen1 / "external.prior_art.json",
        external_source_binding: gen1 / "external.atomic_source_binding.json",
        completion: gen1 / "external.topology_completion_shadow.json",
        gen0_aggregation: gen1 / "residual_aggregation.report.json",
    }
    for source, target in copies.items():
        shutil.copy2(source, target)

    write(
        seed / "adaptive_seed.summary.json",
        {
            "schema_version": "empty-gen1-adaptive-seed-v1",
            "source_stage": "7.75",
            "source_population": "GEN0_UNRESOLVED",
            "stage_775_gen1_empty": True,
            "seed_hypothesis_count": len(gen0_portfolio.hypotheses),
            "seed_hypothesis_ids": [
                str(card.hypothesis_id)
                for card in gen0_portfolio.hypotheses
            ],
            "external_prior_art_as_positive_premise": False,
            "generation_authority_created": False,
            "production_selection_changed": False,
        },
    )
    return seed


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--materialization-report", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--external-query-plan", required=True, type=Path)
    p.add_argument("--external-report", required=True, type=Path)
    p.add_argument("--external-prior-art", required=True, type=Path)
    p.add_argument("--external-source-binding", required=True, type=Path)
    p.add_argument("--provider-plan", required=True, type=Path)
    p.add_argument("--domain-profile", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--critic-model", required=True)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--max-ranked-works", type=int, default=12)
    p.add_argument("--parse-retries", type=int, default=3)
    p.add_argument("--max-fulltext-candidates", type=int, default=4)
    p.add_argument("--max-excerpt-chars", type=int, default=24000)
    p.add_argument("--output-dir", required=True, type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.provider_plan = args.provider_plan.expanduser().resolve()

    context = load_context_source(args.context)
    gen0_portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    gen0_external = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(encoding="utf-8")
    )
    materialization = load(args.materialization_report)

    gen0 = args.output_dir / "gen0"
    gen0.mkdir(parents=True, exist_ok=True)
    lower_report, lower_reviews, lower_packet = lower_order(
        args=args,
        portfolio=args.portfolio,
        query_plan=args.external_query_plan,
        external_report=args.external_report,
        prefix=gen0 / "lower",
    )
    gen0_topology = gen0 / "topology.report.json"
    run_topology(
        args=args,
        query_plan=args.external_query_plan,
        saturation_report=lower_report,
        external_report=args.external_report,
        source_binding=args.external_source_binding,
        output=gen0_topology,
    )
    gen0_fulltext = gen0 / "fulltext.report.json"
    run_fulltext(
        args=args,
        query_plan=args.external_query_plan,
        reviews=lower_reviews,
        packet=lower_packet,
        topology_report=gen0_topology,
        output=gen0_fulltext,
    )
    gen0_aggregation = gen0 / "residual_aggregation.report.json"
    run_aggregation(
        args=args,
        topology_report=gen0_topology,
        reviews=lower_reviews,
        fulltext_report=gen0_fulltext,
        output=gen0_aggregation,
    )

    gen0_state = compile_residual_epistemic_state(
        portfolio=gen0_portfolio.model_dump(mode="json"),
        query_plan=load(args.external_query_plan),
        external_report=load(args.external_report),
        aggregation=load(gen0_aggregation),
    )
    write(gen0 / "epistemic_state.json", gen0_state)

    feedback_plan = build_feedback_plan(
        context=context,
        portfolio=gen0_portfolio,
        closeout=gen0_state,
        external=gen0_external,
        materialization_report=materialization,
    )
    write(args.output_dir / "feedback.plan.json", feedback_plan)

    generation_report, gen1_portfolio = run_feedback_generation(
        context=context,
        portfolio=gen0_portfolio,
        external=gen0_external,
        plan=feedback_plan,
        model=args.model,
        critic_model=args.critic_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        output_dir=args.output_dir / "generation",
    )
    write(args.output_dir / "generation.report.json", generation_report)
    gen1_portfolio_path = args.output_dir / "gen1.portfolio.json"
    write(gen1_portfolio_path, gen1_portfolio)

    if not gen1_portfolio.hypotheses:
        effective = gen1_portfolio
        write(args.output_dir / "effective_gen1.portfolio.json", effective)

        adaptive_seed = materialize_empty_gen1_adaptive_seed(
            output_dir=args.output_dir,
            gen0_portfolio=gen0_portfolio,
            external_query_plan=args.external_query_plan,
            external_report=args.external_report,
            external_prior_art=args.external_prior_art,
            external_source_binding=args.external_source_binding,
            gen0_aggregation=gen0_aggregation,
        )

        summary = summarize_closed_loop(
            gen0_state=gen0_state,
            feedback_plan=feedback_plan,
            generation_report=generation_report,
            gen1_aggregation={
                "residual_state_counts": {},
                "disposition_counts": {},
            },
            gen1_cohort_audit={"pass": True},
            decisions={"decision_counts": {}, "rows": []},
            effective_portfolio=effective,
        )
        summary["status"] = "COMPLETE_EMPTY_GEN1"
        summary["adaptive_seed_available"] = True
        summary["adaptive_seed_source"] = "GEN0_UNRESOLVED"
        summary["adaptive_seed_dir"] = str(adaptive_seed)
        write(args.output_dir / "closed_loop.summary.json", summary)
        print("Closed-loop complete: no Gen1 candidate available.")
        print("Adaptive seed preserved from unresolved Gen0:", adaptive_seed)
        return 0

    gen1 = args.output_dir / "gen1"
    gen1.mkdir(parents=True, exist_ok=True)
    ext_prefix = gen1 / "external"
    ext_cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_external_novelty",
        "--portfolio", str(gen1_portfolio_path),
        "--domain-profile", args.domain_profile,
        "--model", args.critic_model,
        "--api-key-env", args.api_key_env,
        "--provider-plan", str(args.provider_plan),
        "--results-per-query", str(args.results_per_query),
        "--max-ranked-works", str(args.max_ranked_works),
        "--parse-retries", str(args.parse_retries),
        "--pre-review-metadata-resolution",
        "--pre-review-coverage-shadow",
        "--downstream-gate-shadow",
        "--source-bound-topology-shadow",
        "--prior-art-memory-query-plan", str(args.external_query_plan),
        "--prior-art-memory-report", str(args.external_report),
        "--prior-art-memory-packet", str(args.external_prior_art),
        "--output-prefix", str(ext_prefix),
        "--save-prompts",
    ]
    if args.base_url:
        ext_cmd += ["--base-url", args.base_url]
    run("Gen1 fresh external novelty", ext_cmd, args.output_dir)

    gen1_query = gen1 / "external.claims_queries.json"
    gen1_external = gen1 / "external.report.json"
    gen1_binding = gen1 / "external.atomic_source_binding.json"
    gen1_completion = gen1 / "external.topology_completion_shadow.json"

    g1_lower_report, g1_lower_reviews, g1_lower_packet = lower_order(
        args=args,
        portfolio=gen1_portfolio_path,
        query_plan=gen1_query,
        external_report=gen1_external,
        prefix=gen1 / "lower",
    )
    g1_topology = gen1 / "topology.report.json"
    run_topology(
        args=args,
        query_plan=gen1_query,
        saturation_report=g1_lower_report,
        external_report=gen1_external,
        source_binding=gen1_binding,
        output=g1_topology,
    )
    g1_fulltext = gen1 / "fulltext.report.json"
    run_fulltext(
        args=args,
        query_plan=gen1_query,
        reviews=g1_lower_reviews,
        packet=g1_lower_packet,
        topology_report=g1_topology,
        output=g1_fulltext,
    )
    g1_aggregation = gen1 / "residual_aggregation.report.json"
    run_aggregation(
        args=args,
        topology_report=g1_topology,
        reviews=g1_lower_reviews,
        fulltext_report=g1_fulltext,
        output=g1_aggregation,
    )

    cohort = gen1 / "cohort_audit.report.json"
    cohort_code = run(
        "Gen1 residual cohort audit",
        [
            sys.executable,
            "-m",
            "scripts.discovery.run_residual_novelty_cohort_audit_shadow",
            "--aggregation", str(g1_aggregation),
            "--query-plan", str(gen1_query),
            "--topology-completion", str(gen1_completion),
            "--source-binding", str(gen1_binding),
            "--topology-report", str(g1_topology),
            "--output", str(cohort),
        ],
        args.output_dir,
        allow_code_2=True,
    )

    decisions = consolidate_post_verification_decisions(
        gen1_portfolio=gen1_portfolio.model_dump(mode="json"),
        generation_report=generation_report,
        query_plan=load(gen1_query),
        external_report=load(gen1_external),
        aggregation=load(g1_aggregation),
        cohort_audit=load(cohort),
    )
    write(args.output_dir / "post_verification_decisions.json", decisions)

    effective = build_effective_portfolio(
        gen1_portfolio=gen1_portfolio,
        decisions=decisions,
    )
    write(args.output_dir / "effective_gen1.portfolio.json", effective)

    summary = summarize_closed_loop(
        gen0_state=gen0_state,
        feedback_plan=feedback_plan,
        generation_report=generation_report,
        gen1_aggregation=load(g1_aggregation),
        gen1_cohort_audit=load(cohort),
        decisions=decisions,
        effective_portfolio=effective,
    )
    summary["status"] = (
        "COMPLETE"
        if cohort_code == 0
        else "COMPLETE_FAIL_CLOSED_COHORT_AUDIT"
    )
    write(args.output_dir / "closed_loop.summary.json", summary)

    print()
    print("===== SCIENTIFIC PORTFOLIO CLOSED LOOP SHADOW =====")
    print("Gen0 states:", summary["gen0_state_counts"])
    print("Routes:", summary["feedback_route_counts"])
    print("Generation:", summary["generation_decision_counts"])
    print("Gen1 residual:", summary["gen1_residual_disposition_counts"])
    print("Decisions:", summary["post_verification_decision_counts"])
    print("Effective Gen1:", summary["effective_gen1_hypothesis_count"])
    print("IDs:", summary["effective_gen1_hypothesis_ids"])
    print("STAGE8_INPUT_CHANGED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    print("N10_RESEARCH_SELECTION_AUTHORITY=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
