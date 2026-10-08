"""M2-C: directly seed SIS ResearchIdeas without Stage7 HypothesisCard materialization.

Opt-in path: Stage7 Frontier + Evolution -> evaluated Scientific Portfolio -> P0.
The old Full Current runner, Stage7 runtime and scientific claim boundary are unchanged.
External/novelty, hypothesis materialization and child generation are *not* run here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.frontier_exploration_audit import FrontierExplorationAudit
from pipeline_core.discovery.frontier_idea_population import FrontierIdeaPopulation
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.idea_evolution import IdeaEvolutionReport
from pipeline_core.discovery.research_idea_e2e_integration import (
    build_e2e_research_idea_seed,
    build_e2e_seed_execution,
)
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
)
from pipeline_core.discovery.scientific_portfolio_prompt import build_evaluation_prompt
from pipeline_core.discovery.scientific_portfolio_runtime import (
    InstructorOpenAICompatibleScientificPortfolioBackend,
)
from pipeline_core.discovery.scientific_portfolio_selection import (
    ScientificPortfolioCandidatePool,
    ScientificPortfolioEvaluationReport,
    ScientificPortfolioSelectionReport,
    build_scientific_portfolio_candidate_pool,
    build_scientific_portfolio_selection,
    compile_scientific_portfolio_evaluation,
)


def _load(path: Path, model: type) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _read_dict(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError(f"unsupported JSON value: {type(value)}")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_exact(label: str, actual: Any, expected: Any) -> None:
    """Exact scientific/lineage parity, including list order and identities."""
    if _json(actual) != _json(expected):
        raise ValueError(f"{label}: exact parity FAILED (no output written)")


def _check_manifest_pin(manifest_path: Path, key: str, path: Path) -> None:
    manifest = _read_dict(manifest_path)
    rows = [x for x in manifest.get("artifacts", []) if x.get("key") == key]
    if len(rows) != 1 or not rows[0].get("present"):
        raise ValueError(f"M1 manifest lacks {key} freeze pin")
    expected = rows[0]["source_sha256"]
    got = _sha(path)
    if got != expected:
        raise ValueError(f"M1 {key} SHA mismatch: expected {expected}, got {got}")


def _validate_pool_evaluation(pool: Any, evaluation: Any) -> None:
    if evaluation.source_pool_id != pool.pool_id:
        raise ValueError("pool/evaluation lineage ID mismatch")
    if evaluation.source_pool_sha256 != pool.pool_sha256:
        raise ValueError("pool/evaluation lineage SHA mismatch")
    pids = [x.candidate_id for x in pool.candidates]
    eids = [x.candidate_id for x in evaluation.evaluations]
    if len(pids) != len(set(pids)) or len(eids) != len(set(eids)):
        raise ValueError("duplicate candidate or evaluation IDs")
    if set(pids) != set(eids):
        raise ValueError("evaluation must cover every candidate exactly once")


def _make_p0_population(seed: Any) -> dict[str, Any]:
    # Same contract and ordering used by run_research_idea_e2e_search_v3_2.
    return {
        "schema_version": "sis-v3-2-e2e-p0-population-v1",
        "seed_id": seed.seed_id,
        "seed_sha256": seed.seed_sha256,
        "research_idea_count": seed.selected_candidate_count,
        "research_ideas": [n.model_dump(mode="json") for n in seed.research_ideas],
        "epistemic_status": "INSPIRATION_ONLY",
        "positive_premise_authority": False,
        "production_selection_authority": False,
    }


def _check_prefix(candidate: Path, another: Path) -> bool:
    return candidate == another or another in candidate.parents


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Opt-in Stage7 -> P0 ResearchIdea seed, NO early HypothesisCard materialization; "
            "reuse frozen evaluation or make exactly one portfolio evaluation LLM call."
        )
    )
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--population", required=True, type=Path)
    p.add_argument("--evolution-report", required=True, type=Path)
    p.add_argument("--candidate-pool", type=Path, default=None,
                   help="Reuse previously projected candidate pool; otherwise provide --frontier-audit")
    p.add_argument("--frontier-audit", type=Path, default=None)
    p.add_argument("--evaluation", type=Path, default=None,
                   help="Reuse frozen portfolio evaluation (0 LLM calls); omit for new LLM evaluation")
    p.add_argument("--allow-evaluation-llm", action="store_true",
                   help="Explicitly authorize one new LLM evaluation call when --evaluation is absent")
    p.add_argument("--task-source", default=None)
    p.add_argument("--task-target", default=None)
    p.add_argument("--model", default=os.getenv("OPENROUTER_AGENT_MODEL") or "")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max-evaluation-candidates", type=int, default=48)
    p.add_argument("--max-retained", type=int, default=8)
    p.add_argument("--max-retained-per-profile", type=int, default=2)
    p.add_argument("--expected-selection", type=Path, default=None)
    p.add_argument("--expected-p0-execution", type=Path, default=None)
    p.add_argument("--m1-freeze-manifest", type=Path, default=None)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--save-prompt", action="store_true")
    return p


def run(args: argparse.Namespace) -> dict[str, Any]:
    ctx_path = args.context.expanduser().resolve()
    pop_path = args.population.expanduser().resolve()
    evo_path = args.evolution_report.expanduser().resolve()
    pool_path = args.candidate_pool.expanduser().resolve() if args.candidate_pool else None
    eval_path = args.evaluation.expanduser().resolve() if args.evaluation else None
    out = args.output_dir.expanduser().resolve()

    if out.exists():
        raise FileExistsError(f"refusing to overwrite M2-C output: {out}")
    if eval_path is None and not args.allow_evaluation_llm:
        raise ValueError("fresh LLM evaluation is disabled by default; pass --evaluation or --allow-evaluation-llm")
    protected = [ctx_path, pop_path, evo_path]
    for p in (pool_path, eval_path, args.frontier_audit, args.expected_selection, args.expected_p0_execution, args.m1_freeze_manifest):
        if p is not None:
            protected.append(Path(p).expanduser().resolve())
    if any(_check_prefix(out, f.parent) for f in protected):
        raise ValueError("output directory must be outside all source-artifact directories")
    if args.max_evaluation_candidates < 1 or args.max_retained < 1 or args.max_retained_per_profile < 1:
        raise ValueError("budget caps must be >= 1")
    # Preflight comparison inputs *before* potentially paying for new evaluation.
    for opt in (args.expected_selection, args.expected_p0_execution, args.m1_freeze_manifest):
        if opt is not None and not Path(opt).expanduser().resolve().is_file():
            raise FileNotFoundError(Path(opt))

    context = _load(ctx_path, HypothesisContext)
    population = _load(pop_path, FrontierIdeaPopulation)
    evolution = _load(evo_path, IdeaEvolutionReport)
    if context.context_id != population.source_context_id or context.context_sha256 != population.source_context_sha256:
        raise ValueError("context/Frontier ID or SHA mismatch")
    if evolution.source_population_id != population.population_id or evolution.source_population_sha256 != population.population_sha256:
        raise ValueError("Evolution/Frontier ID or SHA mismatch")

    pins = {"task_context": ctx_path, "frontier": pop_path, "stage7_idea_evolution": evo_path}
    if pool_path:
        pins["stage7_candidate_pool"] = pool_path
    if args.m1_freeze_manifest:
        mf = Path(args.m1_freeze_manifest).expanduser().resolve()
        manifest = _read_dict(mf)
        if Path(manifest["source_case"]).expanduser().resolve() != ctx_path.parent:
            raise ValueError("M1 freeze manifest source_case does not match context case")
        if args.expected_selection is not None:
            pins["stage7_selection"] = Path(args.expected_selection).expanduser().resolve()
        if args.expected_p0_execution is not None:
            pins["frozen_p0_execution"] = Path(args.expected_p0_execution).expanduser().resolve()
        for key, path in pins.items():
            _check_manifest_pin(mf, key, path)

    if pool_path:
        pool = _load(pool_path, ScientificPortfolioCandidatePool)
    else:
        if args.frontier_audit is None:
            raise ValueError("--frontier-audit is required when --candidate-pool is omitted")
        frontier_audit = _load(Path(args.frontier_audit).expanduser().resolve(), FrontierExplorationAudit)
        pool = build_scientific_portfolio_candidate_pool(
            population=population,
            exploration_audit=frontier_audit,
            evolution_report=evolution,
            max_evaluation_candidates=args.max_evaluation_candidates,
        )

    # Strict cross-artifact checks even for reused pool: do not build a P0 from
    # a different Frontier/Evolution snapshot or from orphan candidate IDs.
    if pool.source_population_id != population.population_id or pool.source_population_sha256 != population.population_sha256:
        raise ValueError("candidate pool/Frontier mismatch")
    if pool.source_evolution_report_id != evolution.report_id or pool.source_evolution_report_sha256 != evolution.report_sha256:
        raise ValueError("candidate pool/Evolution mismatch")
    if pool.source_context_id != context.context_id or pool.source_context_sha256 != context.context_sha256:
        raise ValueError("candidate pool/context mismatch")

    prompt = None
    llm_calls = 0
    if eval_path:
        evaluation = _load(eval_path, ScientificPortfolioEvaluationReport)
    else:
        if not args.model or not args.task_source or not args.task_target:
            raise ValueError("fresh evaluation requires --model, --task-source and --task-target")
        prompt = build_evaluation_prompt(pool=pool, task_source=args.task_source, task_target=args.task_target)
        backend = InstructorOpenAICompatibleScientificPortfolioBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
        )
        generation = backend.evaluate(prompt)
        llm_calls = 1
        evaluation = compile_scientific_portfolio_evaluation(
            pool=pool,
            draft=generation.draft,
            evaluator_model=backend.model_name,
        )
    _validate_pool_evaluation(pool, evaluation)

    selection = build_scientific_portfolio_selection(
        pool=pool,
        evaluation=evaluation,
        max_retained_candidates=args.max_retained,
        max_retained_per_profile=args.max_retained_per_profile,
    )
    if args.expected_selection:
        baseline_selection = _load(Path(args.expected_selection).expanduser().resolve(), ScientificPortfolioSelectionReport)
        _assert_exact("Stage7 scientific portfolio selection", selection, baseline_selection)

    seed = build_e2e_research_idea_seed(
        population=population,
        evolution_report=evolution,
        pool=pool,
        selection=selection,
    )
    execution = build_e2e_seed_execution(seed)
    if args.expected_p0_execution:
        old_execution = _load(Path(args.expected_p0_execution).expanduser().resolve(), EpistemicG4ExecutionReport)
        _assert_exact("P0 execution scientific identity / genealogy", execution, old_execution)

    # Output is only committed after all validation/parity gates pass.
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "candidate_pool.json", pool)
    _write(out / "evaluation.json", evaluation)
    _write(out / "selection.json", selection)
    _write(out / "p0.seed_report.json", seed)
    _write(out / "p0.execution.json", execution)
    _write(out / "p0.population.json", _make_p0_population(seed))
    inputs = {
        "context": ctx_path,
        "population": pop_path,
        "evolution": evo_path,
    }
    if pool_path:
        inputs["candidate_pool"] = pool_path
    if eval_path:
        inputs["evaluation"] = eval_path
    summary = {
        "schema_version": "ai-scientist-m2c-direct-research-idea-seed-v1",
        "status": "DIRECT_P0_READY",
        "candidate_pool_count": len(pool.candidates),
        "selected_research_idea_count": seed.selected_candidate_count,
        "selection_id": selection.selection_id,
        "seed_id": seed.seed_id,
        "seed_sha256": seed.seed_sha256,
        "p0_execution_report_id": execution.report_id,
        "p0_execution": str(out / "p0.execution.json"),
        "early_hypothesis_materialization_executed": False,
        "early_materialization_llm_calls": 0,
        "portfolio_evaluation_llm_calls": llm_calls,
        "old_selection_parity_checked": bool(args.expected_selection),
        "old_p0_exact_parity_checked": bool(args.expected_p0_execution),
        "strict_hypothesis_grounding_relaxed": False,
        "scientific_truth_or_novelty_certified": False,
        "common_final_verifier_executed": False,
        "source_sha256": {key: _sha(path) for key, path in inputs.items()},
    }
    _write(out / "direct_seed.summary.json", summary)
    if args.save_prompt and prompt is not None:
        (out / "evaluation.prompt.txt").write_text(
            "SYSTEM\n======\n" + prompt.system_prompt + "\n\nUSER\n====\n" + prompt.user_prompt + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> int:
    result = run(parser().parse_args())
    print("M2-C direct ResearchIdea P0 ready:", result["p0_execution"])
    print("P0 count:", result["selected_research_idea_count"])
    print("Portfolio evaluation LLM calls:", result["portfolio_evaluation_llm_calls"])
    print("Early HypothesisCard materialization calls: 0")
    print("Exact P0 parity checked:", result["old_p0_exact_parity_checked"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
