from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.research_idea_adaptive_fertility import (
    build_adaptive_fertility_report,
    build_fertility_gated_parallel_view,
    compose_persistent_population_execution,
)
from pipeline_core.discovery.research_idea_closed_generation_cycle import population_nodes
from pipeline_core.discovery.research_idea_closed_loop import RealizationLifecycleReport
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
    build_epistemic_generation_plan,
    execute_epistemic_generation,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    InstructorOpenAICompatibleOffspringBackend,
)
from pipeline_core.discovery.research_idea_parallel_partial_search import (
    ParallelPartialSearchReport,
)


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


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
        raise FileNotFoundError(f"Required SIS-v3.2 artifact is missing: {path}")
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required SIS-v3.2 artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


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
            "SIS-v3.2 common-R0 E2E ablation harness. Freeze P0, execute exactly one "
            "common strict-realization/epistemic R0, then fork that exact frozen state "
            "into NO_SEARCH, ALWAYS_EVOLVE(v3.0), and ADAPTIVE(v3.1). This runner is "
            "benchmark wiring only: it does not alter SIS search policy, N10 authority, "
            "production selection, or the canonical graph."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--common-r0-source",
        type=Path,
        default=None,
        help=(
            "Optional existing corrected SIS-v3.2 output root containing p0.execution.json "
            "and arms/none/case_root/.../sis_v3_0.g2_* artifacts. When supplied, reuse "
            "that frozen R0 exactly and make no new pre-policy realization calls."
        ),
    )
    p.add_argument("--max-ideas", type=int, default=None)
    p.add_argument("--max-realizations-per-idea", type=int, default=2)
    p.add_argument("--max-realizations-per-parent", type=int, default=2)
    p.add_argument("--realization-repairs", type=int, default=1)
    p.add_argument("--acquisition-persistence-threshold", type=int, default=2)
    p.add_argument("--max-next-parents", type=int, default=None)
    p.add_argument("--max-next-outputs-per-parent", type=int, default=1, choices=(1, 2))
    p.add_argument("--next-semantic-retries", type=int, default=1)
    p.add_argument("--population-growth-budget", type=int, default=0)
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
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--realization-temperature", type=float, default=0.0)
    p.add_argument("--generation-temperature", type=float, default=0.2)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    p.add_argument("--force-common-r0", action="store_true")
    p.add_argument("--save-prompts", action="store_true")
    return p


def _run(label: str, cmd: list[str], *, log_dir: Path) -> None:
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


def _common_paths_from_root(root: Path) -> dict[str, Path]:
    case_out = root / "arms" / "none" / "case_root" / "scientific_portfolio_shadow"
    return {
        "root": root,
        "p0_execution": root / "p0.execution.json",
        "p0_seed": root / "p0.seed_report.json",
        "manifest": root / "integration.manifest.json",
        "none_summary": root / "arms" / "none" / "arm.summary.json",
        "lifecycle": case_out / "sis_v3_0.g2_realization_lifecycle.json",
        "portfolio": case_out / "sis_v3_0.g2_materialized_portfolio.json",
        "decomposition": case_out / "sis_v3_0.g2_epistemic_decomposition.json",
        "archive": case_out / "sis_v3_0.g2_multi_realization_archive.json",
        "parallel": case_out / "sis_v3_0.g2_parallel_partial_search.json",
    }


def _common_paths(out: Path) -> dict[str, Path]:
    return _common_paths_from_root(out / "common_r0")


def _common_complete(paths: dict[str, Path]) -> bool:
    return all(
        paths[key].is_file()
        for key in (
            "p0_execution",
            "p0_seed",
            "manifest",
            "none_summary",
            "lifecycle",
            "portfolio",
            "decomposition",
            "archive",
            "parallel",
        )
    )


def _base_model_args(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = []
    if args.model:
        cmd += ["--model", str(args.model)]
    if args.realization_model:
        cmd += ["--realization-model", str(args.realization_model)]
    if args.generation_model:
        cmd += ["--generation-model", str(args.generation_model)]
    if args.base_url:
        cmd += ["--base-url", str(args.base_url)]
    cmd += [
        "--api-key-env", str(args.api_key_env),
        "--instructor-mode", str(args.instructor_mode),
        "--realization-temperature", str(args.realization_temperature),
        "--generation-temperature", str(args.generation_temperature),
        "--parse-retries", str(args.parse_retries),
        "--timeout", str(args.timeout),
    ]
    for key, value in args.header:
        cmd += ["--header", f"{key}={value}"]
    return cmd


def _ensure_common_r0(
    *,
    args: argparse.Namespace,
    run: Path,
    out: Path,
    paths: dict[str, Path],
) -> None:
    if _common_complete(paths) and not args.force_common_r0:
        print("Reusing frozen common R0:", paths["root"])
        return
    if paths["root"].exists() and args.force_common_r0:
        shutil.rmtree(paths["root"])
    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_research_idea_e2e_search_v3_2",
        "--run-dir", str(run),
        "--output-dir", str(paths["root"]),
        "--mode", "none",
        "--cycles", "1",
        "--max-realizations-per-idea", str(args.max_realizations_per_idea),
        "--max-realizations-per-parent", str(args.max_realizations_per_parent),
        "--realization-repairs", str(args.realization_repairs),
        "--acquisition-persistence-threshold", str(args.acquisition_persistence_threshold),
        "--max-next-outputs-per-parent", str(args.max_next_outputs_per_parent),
        "--next-semantic-retries", str(args.next_semantic_retries),
        "--population-growth-budget", str(args.population_growth_budget),
        *_base_model_args(args),
    ]
    if args.max_ideas is not None:
        cmd += ["--max-ideas", str(args.max_ideas)]
    if args.max_next_parents is not None:
        cmd += ["--max-next-parents", str(args.max_next_parents)]
    if args.save_prompts:
        cmd.append("--save-prompts")
    _run("common R0 strict realization", cmd, log_dir=out / "logs")
    if not _common_complete(paths):
        missing = [key for key, path in paths.items() if key != "root" and not path.is_file()]
        raise RuntimeError("common R0 did not produce required artifacts: " + ", ".join(missing))


def _parent_by_id(execution: EpistemicG4ExecutionReport) -> dict[str, ResearchIdeaNode]:
    rows = [ResearchIdeaNode.model_validate(row) for row in population_nodes(execution)]
    if not rows:
        raise ValueError("frozen P0 has no ResearchIdea population")
    return {row.idea_id: row for row in rows}


def _write_prompts(directory: Path, prompts: tuple[Any, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts, start=1):
        suffix = str(prompt.task_id).split(":")[-1]
        (directory / f"generation_{index:02d}_{suffix}.txt").write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )


def _generation_backend(
    *, args: argparse.Namespace, arm: str, output_dir: Path
) -> InstructorOpenAICompatibleOffspringBackend:
    model = args.generation_model or args.model
    if not model:
        raise SystemExit("generation requires --model/--generation-model or configured env")
    return InstructorOpenAICompatibleOffspringBackend(
        model=model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.generation_temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        extra_headers=dict(args.header),
        telemetry_path=output_dir / "generation.telemetry.jsonl",
        telemetry_context={
            "pipeline": "sis_v3_2_common_r0_policy_fork",
            "arm": arm,
            "generation": 3,
        },
    )


def _terminal_realization(
    *,
    args: argparse.Namespace,
    run: Path,
    context: Path,
    arm: str,
    execution_path: Path,
    arm_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    case_root = arm_dir / "case_root"
    (case_root / "scientific_portfolio_shadow").mkdir(parents=True, exist_ok=True)
    module = (
        "scripts.discovery.run_research_idea_adaptive_fertility_cycle_v3_1"
        if arm == "v3_1"
        else "scripts.discovery.run_research_idea_closed_generation_cycle_v3_0"
    )
    output = arm_dir / "terminal_realization.cohort.json"
    cmd = [
        sys.executable,
        "-m",
        module,
        "--case", f"E2E={case_root}",
        "--execution", f"E2E={execution_path}",
        "--context", f"E2E={context}",
        "--max-realizations-per-idea", str(args.max_realizations_per_idea),
        "--max-realizations-per-parent", str(args.max_realizations_per_parent),
        "--realization-repairs", str(args.realization_repairs),
        "--acquisition-persistence-threshold", str(args.acquisition_persistence_threshold),
        "--max-next-outputs-per-parent", str(args.max_next_outputs_per_parent),
        "--next-semantic-retries", str(args.next_semantic_retries),
        "--population-growth-budget", str(args.population_growth_budget),
        "--output", str(output),
        *_base_model_args(args),
    ]
    if args.max_ideas is not None:
        cmd += ["--max-ideas", str(args.max_ideas)]
    if args.max_next_parents is not None:
        cmd += ["--max-next-parents", str(args.max_next_parents)]
    if args.save_prompts:
        cmd.append("--save-prompts")
    _run(f"{arm} terminal strict realization", cmd, log_dir=arm_dir / "logs")
    payload = _load_json(output)
    case = payload.get("cases", {}).get("E2E", {})
    generation = int(case.get("current_generation_index"))
    cycle_tag = "sis_v3_1" if arm == "v3_1" else "sis_v3_0"
    portfolio = case_root / "scientific_portfolio_shadow" / f"{cycle_tag}.g{generation}_materialized_portfolio.json"
    if not portfolio.is_file():
        raise FileNotFoundError(f"terminal materialized portfolio missing: {portfolio}")
    return portfolio, payload


def _fork_generation(
    *,
    args: argparse.Namespace,
    arm: str,
    run: Path,
    out: Path,
    current_execution: EpistemicG4ExecutionReport,
    lifecycle: RealizationLifecycleReport,
    parallel: ParallelPartialSearchReport,
    context: HypothesisContext,
) -> dict[str, Any]:
    if arm not in {"v3_0", "v3_1"}:
        raise ValueError(f"unsupported policy arm: {arm}")
    arm_dir = out / "arms" / arm
    arm_out = arm_dir / "case_root" / "scientific_portfolio_shadow"
    arm_out.mkdir(parents=True, exist_ok=True)

    parent_by_id = _parent_by_id(current_execution)
    max_parents = args.max_next_parents or len(parent_by_id)
    generation_parallel: Any = parallel
    fertility = None
    persistence = None

    if arm == "v3_1":
        fertility = build_adaptive_fertility_report(
            parallel_report=parallel,
            lifecycle=lifecycle,
            cycle_generation_index=current_execution.generation_index,
        )
        _write(arm_out / "sis_v3_1.g2_adaptive_fertility.json", fertility)
        generation_parallel = build_fertility_gated_parallel_view(
            parallel_report=parallel,
            fertility_report=fertility,
        )
        _write(
            arm_out / "sis_v3_1.g2_fertile_handoffs.json",
            {
                "schema_version": "sis-v3-2-common-r0-fertility-gated-handoffs-v1",
                "source_common_parallel_report_id": parallel.report_id,
                "source_fertility_report_id": fertility.report_id,
                "handoff_count": len(generation_parallel.evolution_handoffs),
                "handoffs": [
                    row.model_dump(mode="json") if hasattr(row, "model_dump") else vars(row)
                    for row in generation_parallel.evolution_handoffs
                ],
                "positive_premise_authority": False,
                "production_generation_authority": False,
            },
        )

    next_plan = build_epistemic_generation_plan(
        parallel_report=generation_parallel,
        parent_by_id=parent_by_id,
        generation_index=current_execution.generation_index + 1,
        max_parents=max_parents,
        max_outputs_per_parent=args.max_next_outputs_per_parent,
    )
    prefix = "sis_v3_1" if arm == "v3_1" else "sis_v3_0"
    _write(arm_out / f"{prefix}.g3_generation_plan.json", next_plan)

    backend = _generation_backend(args=args, arm=arm, output_dir=arm_dir)
    raw_execution, prompts = execute_epistemic_generation(
        plan=next_plan,
        parallel_report=generation_parallel,
        parent_by_id=parent_by_id,
        research_question=context.question,
        backend=backend,
        semantic_retry_limit=args.next_semantic_retries,
    )
    if args.save_prompts:
        _write_prompts(arm_dir / "generation_prompts", prompts)

    if arm == "v3_1":
        _write(arm_out / "sis_v3_1.g3_raw_generation_execution.json", raw_execution)
        next_execution, persistence = compose_persistent_population_execution(
            current_execution=current_execution,
            fertility_report=fertility,
            next_generation_index=current_execution.generation_index + 1,
            next_plan=next_plan,
            raw_next_execution=raw_execution,
            population_growth_budget=args.population_growth_budget,
        )
        _write(arm_out / "sis_v3_1.g2_population_persistence.json", persistence)
        execution_path = arm_out / "sis_v3_1.g3_adaptive_population_execution.json"
    else:
        next_execution = raw_execution
        execution_path = arm_out / "sis_v3_0.g3_generation_execution.json"

    _write(execution_path, next_execution)
    _write(
        arm_out / ("sis_v3_1.g3_adaptive_population.json" if arm == "v3_1" else "sis_v3_0.g3_population.json"),
        {
            "schema_version": "sis-v3-2-common-r0-next-population-v1",
            "arm": arm,
            "generation_index": next_execution.generation_index,
            "source_execution_report_id": next_execution.report_id,
            "research_idea_count": next_execution.g4_population_count,
            "research_ideas": [
                row.model_dump(mode="json") if hasattr(row, "model_dump") else row
                for row in next_execution.g4_population_nodes
            ],
            "positive_premise_authority": False,
            "production_generation_authority": False,
        },
    )

    if next_execution.g4_population_count < 1:
        raise RuntimeError(f"{arm}: policy fork produced an empty next ResearchIdea population")

    source_final, terminal = _terminal_realization(
        args=args,
        run=run,
        context=run / "hypothesis.context.json",
        arm=arm,
        execution_path=execution_path,
        arm_dir=arm_dir,
    )
    final_path = arm_dir / "final.materialized.portfolio.json"
    shutil.copy2(source_final, final_path)
    portfolio = _load(final_path, HypothesisPortfolio)
    terminal_agg = terminal.get("aggregate", {})

    summary = {
        "schema_version": "sis-v3-2-common-r0-arm-v1",
        "arm": arm,
        "common_r0_reused": True,
        "common_r0_parallel_report_id": parallel.report_id,
        "common_r0_lifecycle_report_id": lifecycle.report_id,
        "generation_plan_id": next_plan.plan_id,
        "selected_parent_count": next_plan.task_count,
        "generation_llm_calls": next_execution.llm_call_count,
        "terminal_realization_llm_calls": int(
            terminal_agg.get("realization_llm_call_count", 0) or 0
        ),
        "final_hypothesis_count": len(portfolio.hypotheses),
        "final_portfolio": str(final_path),
        "next_execution": str(execution_path),
        "fertility_report_id": fertility.report_id if fertility is not None else None,
        "fertility_dispositions": (
            fertility.disposition_counts if fertility is not None else None
        ),
        "carried_forward_count": (
            persistence.carried_forward_count if persistence is not None else 0
        ),
        "replaced_parent_count": (
            persistence.replaced_parent_count if persistence is not None else 0
        ),
        "canonical_graph_mutated": False,
        "production_selection_authority": False,
    }
    _write(arm_dir / "arm.summary.json", summary)
    return summary


def main() -> int:
    args = parser().parse_args()
    if not (args.model or args.realization_model):
        raise SystemExit("common R0 requires --model/--realization-model or configured env")
    if not (args.model or args.generation_model):
        raise SystemExit("policy forks require --model/--generation-model or configured env")
    if args.max_realizations_per_idea < 1 or args.max_realizations_per_parent < 1:
        raise ValueError("realization maxima must be >= 1")
    if args.realization_repairs < 0 or args.next_semantic_retries < 0:
        raise ValueError("repair/retry counts must be >= 0")
    if args.acquisition_persistence_threshold < 1:
        raise ValueError("--acquisition-persistence-threshold must be >= 1")
    if args.population_growth_budget < 0:
        raise ValueError("--population-growth-budget must be >= 0")

    run = args.run_dir.expanduser().resolve()
    if not (run / "hypothesis.context.json").is_file():
        raise FileNotFoundError(f"not an E2E run directory: {run}")
    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run / "scientific_portfolio_shadow" / "sis_v3_2_common_r0_ablation"
    )
    out.mkdir(parents=True, exist_ok=True)
    if args.common_r0_source is not None:
        common_source = args.common_r0_source.expanduser().resolve()
        paths = _common_paths_from_root(common_source)
        if not _common_complete(paths):
            missing = [
                key for key, path in paths.items()
                if key != "root" and not path.is_file()
            ]
            raise FileNotFoundError(
                "--common-r0-source is incomplete; missing: " + ", ".join(missing)
            )
        print("Reusing externally frozen common R0:", common_source)
    else:
        paths = _common_paths(out)
        _ensure_common_r0(args=args, run=run, out=out, paths=paths)

    current_execution = _load(paths["p0_execution"], EpistemicG4ExecutionReport)
    lifecycle = _load(paths["lifecycle"], RealizationLifecycleReport)
    parallel = _load(paths["parallel"], ParallelPartialSearchReport)
    common_portfolio = _load(paths["portfolio"], HypothesisPortfolio)
    context = _load(run / "hypothesis.context.json", HypothesisContext)
    seed = _load_json(paths["p0_seed"])
    common_none = _load_json(paths["none_summary"])

    if current_execution.generation_index != lifecycle.generation_index:
        raise ValueError("common R0 lifecycle / P0 execution generation mismatch")
    if lifecycle.source_context_id != context.context_id:
        raise ValueError("common R0 lifecycle / HypothesisContext ID mismatch")
    if lifecycle.source_context_sha256 != context.context_sha256:
        raise ValueError("common R0 lifecycle / HypothesisContext SHA mismatch")

    common_fingerprint = _sha(
        {
            "p0_seed_sha256": seed.get("seed_sha256"),
            "lifecycle": lifecycle,
            "parallel": parallel,
            "portfolio": common_portfolio,
        }
    )

    none_dir = out / "arms" / "none"
    none_dir.mkdir(parents=True, exist_ok=True)
    none_final = none_dir / "final.materialized.portfolio.json"
    shutil.copy2(paths["portfolio"], none_final)
    none_summary = {
        "schema_version": "sis-v3-2-common-r0-arm-v1",
        "arm": "none",
        "common_r0_reused": True,
        "common_r0_fingerprint": common_fingerprint,
        "generation_llm_calls": 0,
        "terminal_realization_llm_calls": 0,
        "shared_common_r0_realization_llm_calls": int(
            common_none.get("continuation_realization_llm_calls", 0) or 0
        ),
        "final_hypothesis_count": len(common_portfolio.hypotheses),
        "final_portfolio": str(none_final),
        "child_generation_executed": False,
        "canonical_graph_mutated": False,
        "production_selection_authority": False,
    }
    _write(none_dir / "arm.summary.json", none_summary)

    v30 = _fork_generation(
        args=args,
        arm="v3_0",
        run=run,
        out=out,
        current_execution=current_execution,
        lifecycle=lifecycle,
        parallel=parallel,
        context=context,
    )
    v31 = _fork_generation(
        args=args,
        arm="v3_1",
        run=run,
        out=out,
        current_execution=current_execution,
        lifecycle=lifecycle,
        parallel=parallel,
        context=context,
    )

    legacy_path = run / "scientific_portfolio_shadow" / "materialized.shadow.portfolio.json"
    legacy_count = None
    if legacy_path.is_file():
        legacy_count = len(_load(legacy_path, HypothesisPortfolio).hypotheses)

    manifest = {
        "schema_version": "sis-v3-2-common-r0-ablation-v1",
        "status": "COMPLETE",
        "source_e2e_run": str(run),
        "common_r0_source": str(paths["root"]),
        "common_r0_source_reused": args.common_r0_source is not None,
        "p0_seed_id": seed.get("seed_id"),
        "p0_seed_sha256": seed.get("seed_sha256"),
        "p0_research_idea_count": seed.get("selected_candidate_count"),
        "common_r0_fingerprint": common_fingerprint,
        "common_r0_lifecycle_report_id": lifecycle.report_id,
        "common_r0_parallel_report_id": parallel.report_id,
        "common_r0_hypothesis_count": len(common_portfolio.hypotheses),
        "common_r0_realization_llm_calls": int(
            common_none.get("continuation_realization_llm_calls", 0) or 0
        ),
        "policy_fork_occurs_after_common_r0": True,
        "independent_pre_policy_realization_per_arm": False,
        "arms": {
            "none": none_summary,
            "v3_0": v30,
            "v3_1": v31,
        },
        "legacy_stage773_portfolio": str(legacy_path) if legacy_path.is_file() else None,
        "legacy_stage773_hypothesis_count": legacy_count,
        "n10_authority_mode_required": "certification_only",
        "n10_candidate_survival_authority": False,
        "n10_pass_fail_is_primary_success_metric": False,
        "common_downstream_verification_executed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    _write(out / "ablation.manifest.json", manifest)

    print()
    print("SIS-v3.2 common-R0 E2E ablation complete")
    print("P0 ideas:", seed.get("selected_candidate_count"))
    print("P0 SHA256:", seed.get("seed_sha256"))
    print("common R0 fingerprint:", common_fingerprint)
    print(
        "common R0 hypotheses / realization calls:",
        len(common_portfolio.hypotheses),
        "/",
        manifest["common_r0_realization_llm_calls"],
    )
    print("none: final hypotheses=", none_summary["final_hypothesis_count"], sep="")
    print(
        "v3_0: final hypotheses=",
        v30["final_hypothesis_count"],
        "; generation calls=",
        v30["generation_llm_calls"],
        sep="",
    )
    print(
        "v3_1: final hypotheses=",
        v31["final_hypothesis_count"],
        "; generation calls=",
        v31["generation_llm_calls"],
        "; fertility=",
        v31["fertility_dispositions"],
        sep="",
    )
    print("POLICY_FORK_AFTER_COMMON_R0=True")
    print("INDEPENDENT_PRE_POLICY_REALIZATION_PER_ARM=False")
    print("N10_CANDIDATE_SURVIVAL_AUTHORITY=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("manifest:", out / "ablation.manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
