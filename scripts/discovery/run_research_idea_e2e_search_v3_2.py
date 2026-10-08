from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.idea_evolution import IdeaEvolutionReport
from pipeline_core.discovery.research_idea_e2e_integration import (
    ResearchIdeaE2ESeedReport,
    build_e2e_research_idea_seed,
    build_e2e_seed_execution,
)
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioSelectionReport,
)


_MODES = ("none", "v3_0", "v3_1", "all")


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _load(path: Path, model):
    if not path.is_file():
        raise FileNotFoundError(f"Required E2E/SIS artifact is missing: {path}")
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required E2E/SIS artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _header(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--header values must use KEY=VALUE")
    key, item = value.split("=", 1)
    if not key.strip():
        raise argparse.ArgumentTypeError("header key may not be empty")
    return key.strip(), item


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "SIS-v3.2 E2E ResearchIdea search integration. Freeze the exact retained "
            "Stage-7.73 Scientific Portfolio as one P0 ResearchIdea population, then "
            "optionally continue that identical P0 through SIS-v3.0 or SIS-v3.1. "
            "The legacy Stage-7.73 materialized portfolio is the NO_SEARCH arm. "
            "This runner never changes E2E production selection and never grants N10 "
            "candidate-survival authority."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--mode", choices=_MODES, default="v3_1")
    p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help=(
            "Number of child-generation steps before terminal strict realization. "
            "Ignored for mode=none. Default: 1."
        ),
    )
    p.add_argument(
        "--seed-only",
        action="store_true",
        help="Freeze/validate P0 and write its compatibility execution without LLM calls.",
    )
    p.add_argument("--output-dir", type=Path, default=None)

    p.add_argument("--max-ideas", type=int, default=None)
    p.add_argument("--max-realizations-per-idea", type=int, default=2)
    p.add_argument("--max-realizations-per-parent", type=int, default=2)
    p.add_argument("--realization-repairs", type=int, default=1)
    p.add_argument("--acquisition-persistence-threshold", type=int, default=2)
    p.add_argument("--max-next-parents", type=int, default=None)
    p.add_argument("--max-next-outputs-per-parent", type=int, default=1, choices=(1, 2))
    p.add_argument("--next-semantic-retries", type=int, default=1)
    p.add_argument("--population-growth-budget", type=int, default=0)
    p.add_argument("--run-prospective", action="store_true")
    p.add_argument("--max-prospective-audits", type=int, default=8)
    p.add_argument("--save-prompts", action="store_true")

    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument("--realization-model", default=None)
    p.add_argument("--generation-model", default=None)
    p.add_argument("--prospective-model", default=None)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--realization-temperature", type=float, default=0.0)
    p.add_argument("--generation-temperature", type=float, default=0.2)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    return p


def _paths(run: Path) -> dict[str, Path]:
    portfolio = run / "scientific_portfolio_shadow"
    return {
        "context": run / "hypothesis.context.json",
        "population": run / "frontier_idea_population.shadow.json",
        "evolution": run / "idea_evolution.shadow.json",
        "pool": portfolio / "candidate_pool.json",
        "selection": portfolio / "selection.json",
        "baseline_portfolio": portfolio / "materialized.shadow.portfolio.json",
    }


def _run_command(label: str, cmd: list[str], *, log_dir: Path) -> None:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)
    print("$", *cmd)
    result = subprocess.run(cmd, text=True, capture_output=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = label.lower().replace(" ", "_").replace("/", "_")
    (log_dir / f"{safe}.stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / f"{safe}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _cycle_command(
    *,
    args: argparse.Namespace,
    mode: str,
    case_root: Path,
    context: Path,
    execution: Path,
    output: Path,
    max_ideas: int,
    max_next_parents: int,
    generate_next: bool,
) -> list[str]:
    module = (
        "scripts.discovery.run_research_idea_adaptive_fertility_cycle_v3_1"
        if mode == "v3_1"
        else "scripts.discovery.run_research_idea_closed_generation_cycle_v3_0"
    )
    cmd = [
        sys.executable,
        "-m",
        module,
        "--case",
        f"E2E={case_root}",
        "--execution",
        f"E2E={execution}",
        "--context",
        f"E2E={context}",
        "--max-ideas",
        str(max_ideas),
        "--max-realizations-per-idea",
        str(args.max_realizations_per_idea),
        "--max-realizations-per-parent",
        str(args.max_realizations_per_parent),
        "--realization-repairs",
        str(args.realization_repairs),
        "--acquisition-persistence-threshold",
        str(args.acquisition_persistence_threshold),
        "--max-next-parents",
        str(max_next_parents),
        "--max-next-outputs-per-parent",
        str(args.max_next_outputs_per_parent),
        "--next-semantic-retries",
        str(args.next_semantic_retries),
        "--max-prospective-audits",
        str(args.max_prospective_audits),
        "--api-key-env",
        str(args.api_key_env),
        "--instructor-mode",
        str(args.instructor_mode),
        "--realization-temperature",
        str(args.realization_temperature),
        "--generation-temperature",
        str(args.generation_temperature),
        "--parse-retries",
        str(args.parse_retries),
        "--timeout",
        str(args.timeout),
        "--output",
        str(output),
    ]
    if args.model:
        cmd += ["--model", str(args.model)]
    if args.realization_model:
        cmd += ["--realization-model", str(args.realization_model)]
    if args.generation_model:
        cmd += ["--generation-model", str(args.generation_model)]
    if args.prospective_model:
        cmd += ["--prospective-model", str(args.prospective_model)]
    if args.base_url:
        cmd += ["--base-url", str(args.base_url)]
    for key, value in args.header:
        cmd += ["--header", f"{key}={value}"]
    if args.run_prospective:
        cmd.append("--run-prospective")
    if args.save_prompts:
        cmd.append("--save-prompts")
    if generate_next:
        cmd.append("--generate-next")
    if mode == "v3_1":
        cmd += ["--population-growth-budget", str(args.population_growth_budget)]
    return cmd


def _portfolio_copy(source: Path, target: Path) -> HypothesisPortfolio:
    portfolio = _load(source, HypothesisPortfolio)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return portfolio


def _run_no_search_arm(
    *,
    args: argparse.Namespace,
    source_run: Path,
    context_path: Path,
    seed_execution_path: Path,
    seed: ResearchIdeaE2ESeedReport,
    arm_dir: Path,
) -> dict[str, Any]:
    """Realize the frozen P0 once with the common SIS lifecycle, then stop.

    This is the primary no-search control. It deliberately uses the same
    strict-realization machinery as the v3.0/v3.1 search arms but does not
    execute child generation. The legacy Stage-7.73 materialized portfolio is
    retained separately as a diagnostic baseline and is not this arm.
    """

    max_ideas = args.max_ideas or seed.selected_candidate_count
    max_next_parents = args.max_next_parents or seed.selected_candidate_count
    case_root = arm_dir / "case_root"
    (case_root / "scientific_portfolio_shadow").mkdir(parents=True, exist_ok=True)

    terminal_output = arm_dir / "terminal_realization.cohort.json"
    cmd = _cycle_command(
        args=args,
        mode="v3_0",
        case_root=case_root,
        context=context_path,
        execution=seed_execution_path,
        output=terminal_output,
        max_ideas=max_ideas,
        max_next_parents=max_next_parents,
        generate_next=False,
    )
    _run_command(
        "none terminal strict realization",
        cmd,
        log_dir=arm_dir / "logs",
    )

    terminal = _load_json(terminal_output)
    terminal_case = terminal.get("cases", {}).get("E2E", {})
    generation_index = int(terminal_case.get("current_generation_index"))
    aggregate = terminal.get("aggregate", {})
    realization_calls = int(aggregate.get("realization_llm_call_count", 0) or 0)
    generation_calls = int(aggregate.get("next_generation_llm_call_count", 0) or 0)
    if generation_calls != 0:
        raise RuntimeError("NO_SEARCH control unexpectedly executed child generation")

    source_final = (
        case_root
        / "scientific_portfolio_shadow"
        / f"sis_v3_0.g{generation_index}_materialized_portfolio.json"
    )
    final_path = arm_dir / "final.materialized.portfolio.json"
    portfolio = _portfolio_copy(source_final, final_path)

    summary = {
        "schema_version": "sis-v3-2-e2e-search-arm-v1",
        "mode": "none",
        "source_p0_seed_id": seed.seed_id,
        "source_p0_seed_sha256": seed.seed_sha256,
        "source_e2e_run": str(source_run),
        "isolated_case_root": str(case_root),
        "child_generation_steps": 0,
        "terminal_output": str(terminal_output),
        "source_final_portfolio": str(source_final),
        "final_portfolio": str(final_path),
        "final_hypothesis_count": len(portfolio.hypotheses),
        "continuation_realization_llm_calls": realization_calls,
        "continuation_generation_llm_calls": 0,
        "uses_existing_stage773_materialization": False,
        "common_sis_strict_realization_used": True,
        "child_generation_executed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    _write(arm_dir / "arm.summary.json", summary)
    return summary


def _run_search_arm(
    *,
    args: argparse.Namespace,
    mode: str,
    source_run: Path,
    context_path: Path,
    seed_execution_path: Path,
    seed: ResearchIdeaE2ESeedReport,
    arm_dir: Path,
) -> dict[str, Any]:
    max_ideas = args.max_ideas or seed.selected_candidate_count
    max_next_parents = args.max_next_parents or seed.selected_candidate_count
    current_execution = seed_execution_path
    case_root = arm_dir / "case_root"
    (case_root / "scientific_portfolio_shadow").mkdir(parents=True, exist_ok=True)
    cycle_outputs: list[str] = []
    realization_calls = 0
    generation_calls = 0

    for step in range(1, args.cycles + 1):
        output = arm_dir / f"cycle_{step:02d}.cohort.json"
        cmd = _cycle_command(
            args=args,
            mode=mode,
            case_root=case_root,
            context=context_path,
            execution=current_execution,
            output=output,
            max_ideas=max_ideas,
            max_next_parents=max_next_parents,
            generate_next=True,
        )
        _run_command(
            f"{mode} child-generation step {step}",
            cmd,
            log_dir=arm_dir / "logs",
        )
        payload = _load_json(output)
        case = payload.get("cases", {}).get("E2E", {})
        next_execution = case.get("next_generation_execution")
        if not next_execution:
            raise RuntimeError(
                f"{mode} step {step} produced no next-generation execution"
            )
        current_execution = Path(str(next_execution)).expanduser().resolve()
        if not current_execution.is_file():
            raise FileNotFoundError(
                f"{mode} step {step} next execution is missing: {current_execution}"
            )
        aggregate = payload.get("aggregate", {})
        realization_calls += int(aggregate.get("realization_llm_call_count", 0) or 0)
        generation_calls += int(aggregate.get("next_generation_llm_call_count", 0) or 0)
        cycle_outputs.append(str(output))

    terminal_output = arm_dir / "terminal_realization.cohort.json"
    terminal_cmd = _cycle_command(
        args=args,
        mode=mode,
        case_root=case_root,
        context=context_path,
        execution=current_execution,
        output=terminal_output,
        max_ideas=max_ideas,
        max_next_parents=max_next_parents,
        generate_next=False,
    )
    _run_command(
        f"{mode} terminal strict realization",
        terminal_cmd,
        log_dir=arm_dir / "logs",
    )
    terminal = _load_json(terminal_output)
    terminal_case = terminal.get("cases", {}).get("E2E", {})
    generation_index = int(terminal_case.get("current_generation_index"))
    aggregate = terminal.get("aggregate", {})
    realization_calls += int(aggregate.get("realization_llm_call_count", 0) or 0)

    cycle_tag = "sis_v3_1" if mode == "v3_1" else "sis_v3_0"
    source_final = (
        case_root
        / "scientific_portfolio_shadow"
        / f"{cycle_tag}.g{generation_index}_materialized_portfolio.json"
    )
    final_path = arm_dir / "final.materialized.portfolio.json"
    portfolio = _portfolio_copy(source_final, final_path)
    final_execution = _load(current_execution, EpistemicG4ExecutionReport)

    summary = {
        "schema_version": "sis-v3-2-e2e-search-arm-v1",
        "mode": mode,
        "source_p0_seed_id": seed.seed_id,
        "source_p0_seed_sha256": seed.seed_sha256,
        "source_e2e_run": str(source_run),
        "isolated_case_root": str(case_root),
        "child_generation_steps": args.cycles,
        "cycle_outputs": cycle_outputs,
        "terminal_output": str(terminal_output),
        "final_execution": str(current_execution),
        "final_execution_report_id": final_execution.report_id,
        "final_execution_generation_index": final_execution.generation_index,
        "source_final_portfolio": str(source_final),
        "final_portfolio": str(final_path),
        "final_hypothesis_count": len(portfolio.hypotheses),
        "continuation_realization_llm_calls": realization_calls,
        "continuation_generation_llm_calls": generation_calls,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    _write(arm_dir / "arm.summary.json", summary)
    return summary


def main() -> int:
    args = parser().parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")
    for name in (
        "max_realizations_per_idea",
        "max_realizations_per_parent",
        "acquisition_persistence_threshold",
        "max_next_outputs_per_parent",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_ideas is not None and args.max_ideas < 1:
        raise ValueError("--max-ideas must be >= 1 when supplied")
    if args.max_next_parents is not None and args.max_next_parents < 1:
        raise ValueError("--max-next-parents must be >= 1 when supplied")
    if args.realization_repairs < 0 or args.next_semantic_retries < 0:
        raise ValueError("repair/retry counts must be >= 0")
    if args.population_growth_budget < 0:
        raise ValueError("--population-growth-budget must be >= 0")
    if args.max_prospective_audits < 0:
        raise ValueError("--max-prospective-audits must be >= 0")

    run = args.run_dir.expanduser().resolve()
    source = _paths(run)
    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run / "scientific_portfolio_shadow" / "sis_v3_2_e2e"
    )
    out.mkdir(parents=True, exist_ok=True)

    context = _load(source["context"], HypothesisContext)
    population = _load(source["population"], FrontierIdeaPopulation)
    evolution = _load(source["evolution"], IdeaEvolutionReport)
    pool = _load(source["pool"], ScientificPortfolioCandidatePool)
    selection = _load(source["selection"], ScientificPortfolioSelectionReport)
    baseline = _load(source["baseline_portfolio"], HypothesisPortfolio)

    seed = build_e2e_research_idea_seed(
        population=population,
        evolution_report=evolution,
        pool=pool,
        selection=selection,
    )
    if context.context_id != seed.source_context_id:
        raise ValueError("E2E HypothesisContext / SIS P0 context ID mismatch")
    if context.context_sha256 != seed.source_context_sha256:
        raise ValueError("E2E HypothesisContext / SIS P0 context SHA mismatch")
    if baseline.source_context_id != seed.source_context_id:
        raise ValueError("Stage-7.73 baseline portfolio / SIS P0 context ID mismatch")
    if baseline.source_context_sha256 != seed.source_context_sha256:
        raise ValueError("Stage-7.73 baseline portfolio / SIS P0 context SHA mismatch")

    seed_execution = build_e2e_seed_execution(seed)
    seed_report_path = out / "p0.seed_report.json"
    seed_execution_path = out / "p0.execution.json"
    seed_population_path = out / "p0.population.json"
    _write(seed_report_path, seed)
    _write(seed_execution_path, seed_execution)
    _write(
        seed_population_path,
        {
            "schema_version": "sis-v3-2-e2e-p0-population-v1",
            "seed_id": seed.seed_id,
            "seed_sha256": seed.seed_sha256,
            "research_idea_count": seed.selected_candidate_count,
            "research_ideas": [row.model_dump(mode="json") for row in seed.research_ideas],
            "epistemic_status": "INSPIRATION_ONLY",
            "positive_premise_authority": False,
            "production_selection_authority": False,
        },
    )

    manifest: dict[str, Any] = {
        "schema_version": "sis-v3-2-e2e-search-integration-v1",
        "run_dir": str(run),
        "mode_requested": args.mode,
        "p0_seed_report": str(seed_report_path),
        "p0_seed_execution": str(seed_execution_path),
        "p0_population": str(seed_population_path),
        "p0_seed_id": seed.seed_id,
        "p0_seed_sha256": seed.seed_sha256,
        "p0_research_idea_count": seed.selected_candidate_count,
        "p0_research_idea_ids": list(seed.selected_research_idea_ids),
        "source_selection_id": seed.source_selection_id,
        "source_context_id": seed.source_context_id,
        "source_context_sha256": seed.source_context_sha256,
        "arms": {},
        "identical_p0_required_across_arms": True,
        "common_final_portfolio_contract": "hypothesis-portfolio-v1",
        "legacy_stage773_portfolio": str(source["baseline_portfolio"]),
        "legacy_stage773_hypothesis_count": len(baseline.hypotheses),
        "none_arm_uses_common_sis_strict_realization": True,
        "downstream_n10_authority_mode_required": "certification_only",
        "n10_candidate_survival_authority": False,
        "n10_pass_fail_is_primary_success_metric": False,
        "n10_executed_by_this_runner": False,
        "common_downstream_verification_executed_by_this_runner": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }

    if args.seed_only:
        manifest["status"] = "P0_FROZEN"
        _write(out / "integration.manifest.json", manifest)
        print("SIS-v3.2 E2E P0 seed frozen")
        print("P0 ideas:", seed.selected_candidate_count)
        print("P0 SHA256:", seed.seed_sha256)
        print("LLM_CALLS=0")
        print("PRODUCTION_SELECTION_CHANGED=False")
        print("N10_CANDIDATE_SURVIVAL_AUTHORITY=False")
        print("manifest:", out / "integration.manifest.json")
        return 0

    requested_modes = (
        ["none", "v3_0", "v3_1"]
        if args.mode == "all"
        else [args.mode]
    )
    if any(mode != "none" for mode in requested_modes) and not (
        args.model or args.realization_model
    ):
        raise SystemExit(
            "SIS search requires --model/--realization-model or a configured model env"
        )
    if any(mode in {"v3_0", "v3_1"} for mode in requested_modes) and not (
        args.model or args.generation_model
    ):
        raise SystemExit(
            "SIS child generation requires --model/--generation-model or a configured model env"
        )

    for mode in requested_modes:
        arm_dir = out / "arms" / mode
        if mode == "none":
            summary = _run_no_search_arm(
                args=args,
                source_run=run,
                context_path=source["context"],
                seed_execution_path=seed_execution_path,
                seed=seed,
                arm_dir=arm_dir,
            )
        else:
            summary = _run_search_arm(
                args=args,
                mode=mode,
                source_run=run,
                context_path=source["context"],
                seed_execution_path=seed_execution_path,
                seed=seed,
                arm_dir=arm_dir,
            )
        if summary["source_p0_seed_sha256"] != seed.seed_sha256:
            raise RuntimeError("arm did not preserve the frozen P0 fingerprint")
        manifest["arms"][mode] = summary

    manifest["status"] = "COMPLETE"
    _write(out / "integration.manifest.json", manifest)

    print()
    print("SIS-v3.2 E2E ResearchIdea search integration complete")
    print("P0 ideas:", seed.selected_candidate_count)
    print("P0 SHA256:", seed.seed_sha256)
    for mode, summary in manifest["arms"].items():
        print(
            f"{mode}: final hypotheses={summary['final_hypothesis_count']}; "
            f"realization calls={summary['continuation_realization_llm_calls']}; "
            f"generation calls={summary['continuation_generation_llm_calls']}"
        )
    print("IDENTICAL_P0_ACROSS_ARMS=True")
    print("N10_AUTHORITY_MODE_REQUIRED=certification_only")
    print("N10_CANDIDATE_SURVIVAL_AUTHORITY=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("manifest:", out / "integration.manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
