from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
)


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load(path: Path, model):
    if not path.is_file():
        raise FileNotFoundError(path)
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _run(label: str, cmd: list[str], *, log_dir: Path) -> None:
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, text=True, capture_output=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = label.lower().replace(" ", "_").replace("/", "_")
    (log_dir / f"{safe}.stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (log_dir / f"{safe}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "SIS-v3.4 E2E continuation over an existing frozen SIS-v3.2 P0 execution. "
            "Runs contemporaneous prior-art guided bounded reproduction for N cycles, "
            "then performs terminal strict realization."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--p0-execution", type=Path, default=None)

    p.add_argument("--max-ideas", type=int, default=8)
    p.add_argument("--max-realizations-per-idea", type=int, default=2)
    p.add_argument("--max-realizations-per-parent", type=int, default=2)
    p.add_argument("--realization-repairs", type=int, default=1)
    p.add_argument("--acquisition-persistence-threshold", type=int, default=2)
    p.add_argument("--max-selected-parents", type=int, default=3)
    p.add_argument("--max-selected-per-same-program", type=int, default=1)
    p.add_argument("--max-next-outputs-per-parent", type=int, default=1, choices=(1, 2))
    p.add_argument("--next-semantic-retries", type=int, default=1)
    p.add_argument("--population-growth-budget", type=int, default=0)

    p.add_argument("--model", default=os.getenv("OPENROUTER_AGENT_MODEL") or "")
    p.add_argument("--realization-model", default=None)
    p.add_argument("--generation-model", default=None)
    p.add_argument("--novelty-model", default=None)
    p.add_argument("--program-family-model", default=None)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--realization-temperature", type=float, default=0.0)
    p.add_argument("--generation-temperature", type=float, default=0.2)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)

    p.add_argument("--providers", default="auto")
    p.add_argument("--novelty-results-per-query", type=int, default=6)
    p.add_argument("--provider-plan", type=Path, default=None)
    p.add_argument("--save-prompts", action="store_true")
    p.add_argument("--feedback-cache-dir", type=Path, default=None)
    p.add_argument("--incremental-program-family", action="store_true")
    p.add_argument("--exact-prior-art-cache", action="store_true")
    p.add_argument("--prior-art-cache-max-age-hours", type=float, default=24.0)
    return p


def _cycle_cmd(
    *,
    args: argparse.Namespace,
    case_root: Path,
    execution: Path,
    context: Path,
    output: Path,
    generate_next: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_research_idea_novelty_aware_cycle_v3_4",
        "--case-root",
        str(case_root),
        "--execution",
        str(execution),
        "--context",
        str(context),
        "--output",
        str(output),
        "--max-ideas",
        str(args.max_ideas),
        "--max-realizations-per-idea",
        str(args.max_realizations_per_idea),
        "--max-realizations-per-parent",
        str(args.max_realizations_per_parent),
        "--realization-repairs",
        str(args.realization_repairs),
        "--acquisition-persistence-threshold",
        str(args.acquisition_persistence_threshold),
        "--max-selected-parents",
        str(args.max_selected_parents),
        "--max-selected-per-same-program",
        str(args.max_selected_per_same_program),
        "--max-next-outputs-per-parent",
        str(args.max_next_outputs_per_parent),
        "--next-semantic-retries",
        str(args.next_semantic_retries),
        "--population-growth-budget",
        str(args.population_growth_budget),
        "--model",
        args.model,
        "--api-key-env",
        args.api_key_env,
        "--instructor-mode",
        args.instructor_mode,
        "--realization-temperature",
        str(args.realization_temperature),
        "--generation-temperature",
        str(args.generation_temperature),
        "--parse-retries",
        str(args.parse_retries),
        "--timeout",
        str(args.timeout),
        "--providers",
        args.providers,
        "--novelty-results-per-query",
        str(args.novelty_results_per_query),
    ]
    if generate_next:
        cmd.append("--generate-next")
    if args.realization_model:
        cmd += ["--realization-model", args.realization_model]
    if args.generation_model:
        cmd += ["--generation-model", args.generation_model]
    if args.novelty_model:
        cmd += ["--novelty-model", args.novelty_model]
    if args.program_family_model:
        cmd += ["--program-family-model", args.program_family_model]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.provider_plan is not None:
        cmd += ["--provider-plan", str(args.provider_plan.expanduser().resolve())]
    if args.save_prompts:
        cmd.append("--save-prompts")
    if args.feedback_cache_dir is not None:
        cmd += ["--feedback-cache-dir", str(args.feedback_cache_dir.expanduser().resolve())]
    if args.incremental_program_family:
        cmd.append("--incremental-program-family")
    if args.exact_prior_art_cache:
        cmd.append("--exact-prior-art-cache")
    cmd += ["--prior-art-cache-max-age-hours", str(args.prior_art_cache_max_age_hours)]
    return cmd


def main() -> int:
    args = parser().parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be >= 1")
    if not args.model:
        raise SystemExit("--model or OPENROUTER_AGENT_MODEL is required")

    run = args.run_dir.expanduser().resolve()
    context = run / "hypothesis.context.json"
    if not context.is_file():
        raise FileNotFoundError(context)

    p0 = (
        args.p0_execution.expanduser().resolve()
        if args.p0_execution is not None
        else run
        / "scientific_portfolio_shadow"
        / "sis_v3_2_e2e"
        / "p0.execution.json"
    )
    if not p0.is_file():
        raise FileNotFoundError(
            f"missing frozen P0 execution: {p0}; run SIS-v3.2 seed creation first or pass --p0-execution"
        )
    p0_execution = _load(p0, EpistemicG4ExecutionReport)

    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run / "scientific_portfolio_shadow" / "sis_v3_4_e2e"
    )
    out.mkdir(parents=True, exist_ok=True)
    case_root = out / "case_root"
    (case_root / "scientific_portfolio_shadow").mkdir(parents=True, exist_ok=True)

    current_execution = p0
    cycle_outputs: list[str] = []
    total_realization_calls = 0
    total_generation_calls = 0
    total_genuine_children = 0
    selected_parent_sets: list[list[str]] = []

    for step in range(1, args.cycles + 1):
        output = out / f"cycle_{step:02d}.summary.json"
        cmd = _cycle_cmd(
            args=args,
            case_root=case_root,
            execution=current_execution,
            context=context,
            output=output,
            generate_next=True,
        )
        _run(f"v3.4 novelty-aware child-generation step {step}", cmd, log_dir=out / "logs")
        payload = _load_json(output)
        next_execution = payload.get("next_generation_execution")
        if not next_execution:
            raise RuntimeError(f"v3.4 step {step} produced no next-generation execution")
        current_execution = Path(str(next_execution)).expanduser().resolve()
        if not current_execution.is_file():
            raise FileNotFoundError(current_execution)
        total_realization_calls += int(payload.get("realization_llm_call_count", 0) or 0)
        total_generation_calls += int(payload.get("next_generation_llm_call_count", 0) or 0)
        total_genuine_children += int(payload.get("next_generation_genuine_child_count", 0) or 0)
        selected_parent_sets.append(list(payload.get("selected_parent_idea_ids") or []))
        cycle_outputs.append(str(output))

    terminal_output = out / "terminal_realization.summary.json"
    terminal_cmd = _cycle_cmd(
        args=args,
        case_root=case_root,
        execution=current_execution,
        context=context,
        output=terminal_output,
        generate_next=False,
    )
    _run("v3.4 terminal strict realization", terminal_cmd, log_dir=out / "logs")
    terminal = _load_json(terminal_output)
    total_realization_calls += int(terminal.get("realization_llm_call_count", 0) or 0)
    final_generation = int(terminal["current_generation_index"])
    source_final = (
        case_root
        / "scientific_portfolio_shadow"
        / f"sis_v3_4.g{final_generation}_materialized_portfolio.json"
    )
    final_path = out / "final.materialized.portfolio.json"
    portfolio = _load(source_final, HypothesisPortfolio)
    shutil.copy2(source_final, final_path)
    final_execution = _load(current_execution, EpistemicG4ExecutionReport)

    parent_set_changed_across_cycles = (
        len({tuple(rows) for rows in selected_parent_sets}) > 1
        if len(selected_parent_sets) > 1
        else False
    )
    summary = {
        "schema_version": "sis-v3-4-e2e-search-arm-v1",
        "mode": "v3_4_novelty_aware_bounded_reproduction",
        "source_run": str(run),
        "source_p0_execution": str(p0),
        "source_p0_report_id": p0_execution.report_id,
        "child_generation_steps": args.cycles,
        "cycle_outputs": cycle_outputs,
        "terminal_output": str(terminal_output),
        "final_execution": str(current_execution),
        "final_execution_report_id": final_execution.report_id,
        "final_execution_generation_index": final_execution.generation_index,
        "final_portfolio": str(final_path),
        "final_hypothesis_count": len(portfolio.hypotheses),
        "continuation_realization_llm_calls": total_realization_calls,
        "continuation_generation_llm_calls": total_generation_calls,
        "genuine_child_count_across_cycles": total_genuine_children,
        "selected_parent_sets": selected_parent_sets,
        "selected_parent_set_changed_across_cycles": parent_set_changed_across_cycles,
        "external_prior_art_promoted_to_positive_premise": False,
        "grounding_required_before_scientific_claim": True,
        "canonical_graph_mutated": False,
        "production_selection_changed": False,
        "n10_executed": False,
    }
    _write(out / "arm.summary.json", summary)

    print()
    print("=== SIS-v3.4 E2E COMPLETE ===")
    print("cycles:", args.cycles)
    print("generation calls:", total_generation_calls)
    print("genuine children:", total_genuine_children)
    print("selected parent sets:", selected_parent_sets)
    print("parent set changed:", parent_set_changed_across_cycles)
    print("final hypotheses:", len(portfolio.hypotheses))
    print("PRIOR_ART_POSITIVE_PREMISE_AUTHORITY=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("summary:", out / "arm.summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
