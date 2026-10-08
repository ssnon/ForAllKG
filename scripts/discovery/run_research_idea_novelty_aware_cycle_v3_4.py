from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext, HypothesisPortfolio
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.research_idea_adaptive_fertility import (
    build_adaptive_fertility_report,
    build_fertility_gated_parallel_view,
    compose_persistent_population_execution,
)
from pipeline_core.discovery.research_idea_closed_generation_cycle import (
    adapt_epistemic_execution_for_realization,
    build_cycle_epistemic_artifacts,
    population_nodes,
)
from pipeline_core.discovery.research_idea_closed_loop import run_realization_lifecycle
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
    build_epistemic_generation_plan,
    execute_epistemic_generation,
)
from pipeline_core.discovery.research_idea_feedback_cache_m2d import (
    incremental_program_pairs,
    load_prior_art_cache,
    prior_art_cache_key,
    save_prior_art_cache,
)
from pipeline_core.discovery.research_idea_novelty_aware_reproduction_v3_4 import (
    NoveltyAwareOffspringBackendAdapter,
    adapt_fertility_report_v3_4,
    build_mutation_search_contexts,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    InstructorOpenAICompatibleOffspringBackend,
)
from pipeline_core.discovery.research_idea_program_family_v3_3_1 import (
    PROGRAM_FAMILY_SYSTEM_PROMPT,
    ScientificProgramPairBatch,
    build_scientific_program_family_report,
    program_family_prompt_payload,
    validate_program_pair_batch,
)
from pipeline_core.discovery.research_idea_search_assessment import (
    build_research_idea_search_assessment,
    build_terminal_novelty_signals_by_idea,
)
from pipeline_core.discovery.research_idea_search_pressure_v3_3_2 import (
    build_bounded_parent_schedule,
    build_novelty_depth_signals_by_idea,
    build_research_idea_search_pressure_report,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _load(path: Path, model):
    if not path.is_file():
        raise FileNotFoundError(path)
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _run(label: str, cmd: list[str], *, log_dir: Path) -> subprocess.CompletedProcess[str]:
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
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result


def _program_pairs(
    *,
    nodes: list[ResearchIdeaNode],
    output_dir: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    parse_retries: int,
) -> ScientificProgramPairBatch:
    if len(nodes) < 2:
        return ScientificProgramPairBatch(pairs=[])

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"No API key available. Set {api_key_env}.")
    try:
        import instructor
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("program-family audit requires openai and instructor") from exc

    payload = program_family_prompt_payload(nodes)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "program_family.prompt.txt").write_text(
        "SYSTEM\n======\n"
        + PROGRAM_FAMILY_SYSTEM_PROMPT
        + "\n\nUSER\n====\nAssess every unordered pair exactly once.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = instructor.from_openai(OpenAI(**kwargs), mode=instructor.Mode.JSON)
    batch, _event = run_instructor_structured_call(
        client.chat.completions,
        model=model,
        response_model=ScientificProgramPairBatch,
        messages=[
            {"role": "system", "content": PROGRAM_FAMILY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Assess every unordered pair exactly once. Preserve idea IDs verbatim.\n\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        temperature=0.0,
        max_retries=parse_retries,
        telemetry_path=output_dir / "program_family.telemetry.jsonl",
        telemetry_context={
            "pipeline": "sis_v3_4_novelty_aware_cycle",
            "stage": "semantic_scientific_program_family",
        },
        semantic_components={
            "authority": "SEARCH_PROGRAM_FAMILY_ONLY",
            "truth_authority": False,
            "novelty_authority": False,
        },
    )
    if not isinstance(batch, ScientificProgramPairBatch):
        batch = ScientificProgramPairBatch.model_validate(batch)
    validate_program_pair_batch(
        idea_ids=[row.idea_id for row in nodes],
        batch=batch,
    )
    _write(output_dir / "program_family.semantic_pairs.json", batch)
    return batch


def _incremental_program_pairs(
    *,
    nodes: list[ResearchIdeaNode],
    output_dir: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    parse_retries: int,
    cache_dir: Path,
) -> ScientificProgramPairBatch:
    """Optional pair-local semantics, evaluated only for changed/unseen pairs."""
    if len(nodes) < 2:
        return ScientificProgramPairBatch(pairs=[])
    payload = program_family_prompt_payload(nodes)
    scoped_prompt = PROGRAM_FAMILY_SYSTEM_PROMPT.replace(
        "Assess EVERY unordered pair exactly once.",
        "Assess ONLY target_pairs given in the user message, exactly once each. "
        "Do not produce any other pairs.",
    )
    scoped_prompt += (
        "\nAssess each pair using ONLY its two idea records, independent of the other "
        "ideas in the batch. Contextual population membership must not determine "
        "the pair's relation.\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def infer_missing(missing: list[tuple[str, str]]) -> list[dict[str, Any]]:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"No API key available. Set {api_key_env}.")
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("program-family audit requires openai and instructor") from exc
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = instructor.from_openai(OpenAI(**kwargs), mode=instructor.Mode.JSON)
        task = {**payload, "target_pairs": [list(pair) for pair in missing]}
        user_text = (
            "Evaluate ONLY target_pairs. Each unordered target pair must appear once. "
            "Return no other pairs. Preserve IDs verbatim.\n\n"
            + json.dumps(task, ensure_ascii=False, indent=2)
        )
        (output_dir / "program_family.incremental.prompt.txt").write_text(
            "SYSTEM\n======\n" + scoped_prompt + "\n\nUSER\n====\n" + user_text + "\n",
            encoding="utf-8",
        )
        batch, _ = run_instructor_structured_call(
            client.chat.completions,
            model=model,
            response_model=ScientificProgramPairBatch,
            messages=[
                {"role": "system", "content": scoped_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            max_retries=parse_retries,
            telemetry_path=output_dir / "program_family.telemetry.jsonl",
            telemetry_context={
                "pipeline": "sis_m2d_incremental_program_pairs",
                "stage": "semantic_scientific_program_family",
            },
            semantic_components={
                "authority": "SEARCH_PROGRAM_FAMILY_ONLY",
                "truth_authority": False,
                "novelty_authority": False,
            },
        )
        if not isinstance(batch, ScientificProgramPairBatch):
            batch = ScientificProgramPairBatch.model_validate(batch)
        return [row.model_dump(mode="json") for row in batch.pairs]

    rows, cache_stats = incremental_program_pairs(
        idea_payloads=payload["ideas"],
        source_contexts={
            row.idea_id: row.source_context_sha256 for row in nodes
        },
        model=model,
        base_url=base_url,
        system_prompt=scoped_prompt,
        cache_dir=cache_dir,
        infer_missing=infer_missing,
    )
    batch = ScientificProgramPairBatch.model_validate({"pairs": rows})
    validate_program_pair_batch(idea_ids=[row.idea_id for row in nodes], batch=batch)
    _write(output_dir / "program_family.semantic_pairs.json", batch)
    _write(output_dir / "program_family.cache_stats.json", cache_stats)
    return batch


def _run_prior_art_probe(
    *,
    portfolio_path: Path,
    context: HypothesisContext,
    output_dir: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    providers: str,
    results_per_query: int,
    provider_plan: Path | None,
    cache_dir: Path | None = None,
    cache_max_age_hours: float = 24.0,
) -> tuple[LiteratureQueryPlan, ExternalNoveltyReport]:
    prefix = output_dir / "contemporaneous_prior_art"
    plan_path = Path(str(prefix) + ".claims_queries.json")
    report_path = Path(str(prefix) + ".report.json")
    cache_fingerprint = None
    source_portfolio_id = None
    if cache_dir is not None:
        source_portfolio_id = _load(portfolio_path, HypothesisPortfolio).portfolio_id
        cache_fingerprint = prior_art_cache_key(
            portfolio_bytes=portfolio_path.read_bytes(),
            context_id=context.context_id,
            context_sha256=context.context_sha256,
            model=model,
            base_url=base_url,
            providers=providers,
            results_per_query=results_per_query,
            provider_plan_bytes=provider_plan.read_bytes() if provider_plan is not None else None,
            runner_bytes=Path(__file__).with_name("run_external_novelty.py").read_bytes(),
        )
        cached = load_prior_art_cache(
            cache_dir=cache_dir,
            fingerprint=cache_fingerprint,
            max_age_hours=cache_max_age_hours,
            source_portfolio_id=source_portfolio_id,
        )
        if cached is not None:
            try:
                cached_plan = LiteratureQueryPlan.model_validate(cached[0])
                cached_report = ExternalNoveltyReport.model_validate(cached[1])
            except Exception:
                pass  # Invalid structured report => fresh prior-art probe.
            else:
                _write(plan_path, cached_plan)
                _write(report_path, cached_report)
                _write(output_dir / "prior_art.cache_stats.json", {
                    "status": "EXACT_INPUT_CACHE_HIT",
                    "freshness_hours_max": cache_max_age_hours,
                    "fingerprint": cache_fingerprint,
                    "source_portfolio_id": source_portfolio_id,
                    "truth_authority": False,
                })
                return cached_plan, cached_report
    cmd = [
        sys.executable,
        "-m",
        "scripts.discovery.run_external_novelty",
        "--portfolio",
        str(portfolio_path),
        "--domain-profile",
        context.domain_profile_id,
        "--model",
        model,
        "--api-key-env",
        api_key_env,
        "--providers",
        providers,
        "--results-per-query",
        str(results_per_query),
        "--output-prefix",
        str(prefix),
        "--pre-review-coverage-shadow",
        "--downstream-gate-shadow",
        "--save-prompts",
    ]
    if base_url:
        cmd += ["--base-url", base_url]
    if provider_plan is not None:
        cmd += ["--provider-plan", str(provider_plan)]

    result = _run("contemporaneous lightweight prior-art probe", cmd, log_dir=output_dir / "logs")
    if result.returncode != 0:
        raise RuntimeError(
            f"contemporaneous prior-art probe failed with return code {result.returncode}"
        )
    plan = _load(plan_path, LiteratureQueryPlan)
    report = _load(report_path, ExternalNoveltyReport)
    if cache_dir is not None:
        if report.source_portfolio_id != source_portfolio_id:
            raise ValueError("External novelty report does not match exact source portfolio")
        save_prior_art_cache(
            cache_dir=cache_dir,
            fingerprint=cache_fingerprint,
            plan=plan.model_dump(mode="json"),
            report=report.model_dump(mode="json"),
        )
        _write(output_dir / "prior_art.cache_stats.json", {
            "status": "FRESH_PROBE_CACHE_STORED",
            "freshness_hours_max": cache_max_age_hours,
            "fingerprint": cache_fingerprint,
            "source_portfolio_id": source_portfolio_id,
            "truth_authority": False,
        })
    return plan, report


def _write_augmented_prompts(path: Path, backend: NoveltyAwareOffspringBackendAdapter) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for task_id, prompt in sorted(backend.augmented_prompts.items()):
        safe = task_id.replace(":", "_").replace("/", "_")
        (path / f"{safe}.txt").write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run one SIS-v3.4 novelty-aware closed generation cycle. Current ResearchIdeas "
            "are strictly realized, probed against bounded external prior art, assessed by "
            "v3.3.2 search pressure, and only bounded selected parents may generate children."
        )
    )
    p.add_argument("--case-root", required=True, type=Path)
    p.add_argument("--execution", required=True, type=Path)
    p.add_argument("--context", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--generate-next", action="store_true")

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
    p.add_argument("--feedback-cache-dir", type=Path, default=None,
                   help="Opt-in M2-D cache root; default unchanged v3.4 behavior")
    p.add_argument("--incremental-program-family", action="store_true",
                   help="Opt-in pair-local semantic family evaluation with reuse")
    p.add_argument("--exact-prior-art-cache", action="store_true",
                   help="Opt-in exact same portfolio/query cache, freshness <=24 hours")
    p.add_argument("--prior-art-cache-max-age-hours", type=float, default=24.0)
    return p


def main() -> int:
    args = parser().parse_args()
    if not args.model:
        raise SystemExit("--model or OPENROUTER_AGENT_MODEL is required")
    if args.max_selected_parents < 1:
        raise ValueError("--max-selected-parents must be >= 1")
    if args.max_selected_per_same_program < 1:
        raise ValueError("--max-selected-per-same-program must be >= 1")
    if args.population_growth_budget < 0:
        raise ValueError("--population-growth-budget must be >= 0")
    if (args.incremental_program_family or args.exact_prior_art_cache) and args.feedback_cache_dir is None:
        raise ValueError("M2-D caching requires --feedback-cache-dir")
    if args.exact_prior_art_cache and not 0 < args.prior_art_cache_max_age_hours <= 24:
        raise ValueError("prior-art cache age must be (0, 24] hours")
    feedback_cache = args.feedback_cache_dir.expanduser().resolve() if args.feedback_cache_dir else None

    case_root = args.case_root.expanduser().resolve()
    sp = case_root / "scientific_portfolio_shadow"
    sp.mkdir(parents=True, exist_ok=True)
    execution = _load(args.execution.expanduser().resolve(), EpistemicG4ExecutionReport)
    context = _load(args.context.expanduser().resolve(), HypothesisContext)
    generation = execution.generation_index
    nodes = [ResearchIdeaNode.model_validate(row) for row in population_nodes(execution)]
    if not nodes:
        raise ValueError("current execution has no retained ResearchIdea population")
    if any(row.source_context_id != context.context_id for row in nodes):
        raise ValueError("execution/context ID mismatch")
    if any(row.source_context_sha256 != context.context_sha256 for row in nodes):
        raise ValueError("execution/context SHA mismatch")

    tag = f"sis_v3_4.g{generation}"
    work = sp / "sis_v3_4_novelty_aware_cycle" / f"g{generation}"
    work.mkdir(parents=True, exist_ok=True)

    realization_model = args.realization_model or args.model
    realization_backend = InstructorOpenAICompatibleHypothesisBackend(
        model=realization_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        instructor_mode=args.instructor_mode,
        temperature=args.realization_temperature,
        parse_retries=args.parse_retries,
        timeout=args.timeout,
        telemetry_path=work / "realization.telemetry.jsonl",
        telemetry_context={
            "pipeline": "sis_v3_4_novelty_aware_cycle_realization",
            "generation": generation,
        },
    )
    legacy_execution = adapt_epistemic_execution_for_realization(execution)
    lifecycle, portfolio = run_realization_lifecycle(
        execution=legacy_execution,
        context=context,
        backend=realization_backend,
        output_dir=work / "realization",
        max_ideas=min(args.max_ideas, len(nodes)),
        max_realizations_per_idea=args.max_realizations_per_idea,
        max_per_parent=args.max_realizations_per_parent,
        max_repair_attempts=args.realization_repairs,
    )
    lifecycle_path = sp / f"{tag}_realization_lifecycle.json"
    portfolio_path = sp / f"{tag}_materialized_portfolio.json"
    _write(lifecycle_path, lifecycle)
    _write(portfolio_path, portfolio)

    decomposition, archive, parallel = build_cycle_epistemic_artifacts(
        execution=execution,
        lifecycle=lifecycle,
        portfolio=portfolio,
        acquisition_persistence_threshold=args.acquisition_persistence_threshold,
    )
    _write(sp / f"{tag}_epistemic_decomposition.json", decomposition)
    _write(sp / f"{tag}_multi_realization_archive.json", archive)
    _write(sp / f"{tag}_parallel_partial_search.json", parallel)

    summary: dict[str, Any] = {
        "schema_version": "sis-v3-4-novelty-aware-cycle-summary-v1",
        "current_generation_index": generation,
        "next_generation_index": generation + 1,
        "current_population_count": len(nodes),
        "materialized_hypothesis_count": len(portfolio.hypotheses),
        "usable_grounded_realization_count": lifecycle.usable_grounded_realization_count,
        "realization_llm_call_count": lifecycle.realization_llm_call_count,
        "child_generation_requested": bool(args.generate_next),
        "next_generation_llm_call_count": 0,
        "next_generation_genuine_child_count": 0,
        "next_generation_execution": None,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "external_prior_art_promoted_to_positive_premise": False,
    }

    if not args.generate_next:
        summary["status"] = "TERMINAL_REALIZATION_COMPLETE"
        _write(args.output.expanduser().resolve(), summary)
        print("SIS-v3.4 terminal realization complete")
        print("generation:", generation)
        print("hypotheses:", len(portfolio.hypotheses))
        print("output:", args.output.expanduser().resolve())
        return 0

    query_plan = None
    external_report = None
    novelty_signals = {}
    depth_signals = {}
    prior_art_dir = work / "prior_art_probe"
    if portfolio.hypotheses:
        query_plan, external_report = _run_prior_art_probe(
            portfolio_path=portfolio_path,
            context=context,
            output_dir=prior_art_dir,
            model=args.novelty_model or args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            providers=args.providers,
            results_per_query=args.novelty_results_per_query,
            provider_plan=(
                args.provider_plan.expanduser().resolve()
                if args.provider_plan is not None
                else None
            ),
            cache_dir=feedback_cache if args.exact_prior_art_cache else None,
            cache_max_age_hours=args.prior_art_cache_max_age_hours,
        )
        novelty_signals = build_terminal_novelty_signals_by_idea(
            terminal_parallel_report=parallel,
            external_novelty_report=external_report,
            source_mode="CONTEMPORANEOUS_SEARCH_PROBE",
        )
        depth_signals = build_novelty_depth_signals_by_idea(
            terminal_parallel_report=parallel,
            external_novelty_report=external_report,
            source_mode="CONTEMPORANEOUS_SEARCH_PROBE",
        )

    assessment = build_research_idea_search_assessment(
        parallel_report=parallel,
        lifecycle=lifecycle,
        population_nodes=nodes,
        novelty_signals_by_idea=novelty_signals,
    )
    _write(sp / f"{tag}_search_assessment.json", assessment)

    program_args = dict(
        nodes=nodes,
        output_dir=work / "program_family",
        model=args.program_family_model or args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        parse_retries=args.parse_retries,
    )
    if args.incremental_program_family:
        pair_batch = _incremental_program_pairs(**program_args, cache_dir=feedback_cache)
    else:
        pair_batch = _program_pairs(**program_args)
    family = build_scientific_program_family_report(nodes=nodes, pair_batch=pair_batch)
    program_cache_stats_path = work / "program_family" / "program_family.cache_stats.json"
    if program_cache_stats_path.is_file() and args.incremental_program_family:
        summary["m2d_program_family_cache"] = json.loads(
            program_cache_stats_path.read_text(encoding="utf-8")
        )
    prior_cache_stats_path = prior_art_dir / "prior_art.cache_stats.json"
    if prior_cache_stats_path.is_file() and args.exact_prior_art_cache:
        summary["m2d_prior_art_cache"] = json.loads(
            prior_cache_stats_path.read_text(encoding="utf-8")
        )
    _write(sp / f"{tag}_scientific_program_family.json", family)

    pressure = build_research_idea_search_pressure_report(
        search_assessment=assessment,
        program_family=family,
        novelty_by_idea=depth_signals,
    )
    _write(sp / f"{tag}_search_pressure.json", pressure)
    schedule = build_bounded_parent_schedule(
        pressure_report=pressure,
        program_family=family,
        max_selected_parents=args.max_selected_parents,
        max_selected_per_same_program=args.max_selected_per_same_program,
    )
    _write(sp / f"{tag}_bounded_parent_schedule.json", schedule)

    base_fertility = build_adaptive_fertility_report(
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=generation,
    )
    compat_fertility, control = adapt_fertility_report_v3_4(
        base_fertility=base_fertility,
        pressure_report=pressure,
        schedule=schedule,
    )
    _write(sp / f"{tag}_base_v3_1_fertility.json", base_fertility)
    _write(sp / f"{tag}_reproduction_compat_fertility.json", compat_fertility)
    _write(sp / f"{tag}_reproduction_control.json", control)

    generation_parallel = build_fertility_gated_parallel_view(
        parallel_report=parallel,
        fertility_report=compat_fertility,
    )
    current_by_id = {row.idea_id: row for row in nodes}
    next_plan = build_epistemic_generation_plan(
        parallel_report=generation_parallel,
        parent_by_id=current_by_id,
        generation_index=generation + 1,
        max_parents=max(1, schedule.selected_parent_count),
        max_outputs_per_parent=args.max_next_outputs_per_parent,
    )
    scheduled_ids = {row.idea_id for row in schedule.selected_parents}
    if set(next_plan.selected_parent_idea_ids) != scheduled_ids:
        raise RuntimeError(
            "v3.4 generation plan did not preserve the bounded scheduler parent set"
        )
    _write(sp / f"sis_v3_4.g{generation + 1}_generation_plan.json", next_plan)

    mutation_contexts = {}
    if query_plan is not None and external_report is not None:
        mutation_contexts = build_mutation_search_contexts(
            parallel_report=parallel,
            query_plan=query_plan,
            external_novelty_report=external_report,
            novelty_by_idea=depth_signals,
            pressure_report=pressure,
            schedule=schedule,
        )
    _write(
        sp / f"{tag}_mutation_search_context.json",
        {
            "schema_version": "sis-v3-4-mutation-search-context-bundle-v1",
            "generation_index": generation,
            "contexts": {
                key: value.model_dump(mode="json")
                for key, value in sorted(mutation_contexts.items())
            },
            "prior_art_only_not_positive_premise": True,
            "scientific_truth_authority": False,
        },
    )

    raw_next = None
    prompts = ()
    if next_plan.task_count:
        base_generation_backend = InstructorOpenAICompatibleOffspringBackend(
            model=args.generation_model or args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.generation_temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            telemetry_path=work / "next_generation.telemetry.jsonl",
            telemetry_context={
                "pipeline": "sis_v3_4_novelty_aware_bounded_reproduction",
                "generation": generation + 1,
            },
        )
        adapter = NoveltyAwareOffspringBackendAdapter(
            backend=base_generation_backend,
            context_by_idea=mutation_contexts,
            task_to_idea={
                task.task_id: task.primary_parent_idea_id for task in next_plan.tasks
            },
        )
        raw_next, prompts = execute_epistemic_generation(
            plan=next_plan,
            parallel_report=generation_parallel,
            parent_by_id=current_by_id,
            research_question=context.question,
            backend=adapter,
            semantic_retry_limit=args.next_semantic_retries,
        )
        _write(sp / f"sis_v3_4.g{generation + 1}_raw_generation_execution.json", raw_next)
        if args.save_prompts:
            _write_augmented_prompts(work / "augmented_generation_prompts", adapter)

    next_execution, persistence = compose_persistent_population_execution(
        current_execution=execution,
        fertility_report=compat_fertility,
        next_generation_index=generation + 1,
        next_plan=next_plan,
        raw_next_execution=raw_next,
        population_growth_budget=args.population_growth_budget,
    )
    next_execution_path = sp / f"sis_v3_4.g{generation + 1}_adaptive_population_execution.json"
    _write(next_execution_path, next_execution)
    _write(sp / f"{tag}_population_persistence.json", persistence)
    _write(
        sp / f"sis_v3_4.g{generation + 1}_adaptive_population.json",
        {
            "schema_version": "sis-v3-4-novelty-aware-persistent-population-v1",
            "generation_index": generation + 1,
            "source_execution_report_id": next_execution.report_id,
            "research_ideas": [
                row.model_dump(mode="json") if hasattr(row, "model_dump") else row
                for row in next_execution.g4_population_nodes
            ],
            "research_idea_count": next_execution.g4_population_count,
            "selected_parent_idea_ids": list(next_plan.selected_parent_idea_ids),
            "epistemic_status": "INSPIRATION_ONLY_UNTIL_STRICT_REALIZATION",
            "grounding_required_before_scientific_claim": True,
            "prior_art_only_not_positive_premise": True,
            "production_selection_authority": False,
        },
    )

    summary.update(
        {
            "status": "NOVELTY_AWARE_REPRODUCTION_COMPLETE",
            "prior_art_probe_executed": bool(portfolio.hypotheses),
            "program_count": family.program_count,
            "m2d_feedback_cache_enabled": feedback_cache is not None,
            "m2d_pair_local_semantics_enabled": bool(args.incremental_program_family),
            "m2d_exact_prior_art_cache_enabled": bool(args.exact_prior_art_cache),
            "search_priority_counts": pressure.search_priority_counts,
            "preferred_action_counts": pressure.preferred_action_counts,
            "selected_parent_idea_ids": list(next_plan.selected_parent_idea_ids),
            "selected_parent_count": len(next_plan.selected_parent_idea_ids),
            "deferred_search_worthy_idea_ids": list(schedule.deferred_search_worthy_idea_ids),
            "next_generation_llm_call_count": (
                raw_next.llm_call_count if raw_next is not None else 0
            ),
            "next_generation_genuine_child_count": (
                raw_next.genuine_child_count if raw_next is not None else 0
            ),
            "next_generation_indeterminate_probe_count": (
                raw_next.indeterminate_probe_count if raw_next is not None else 0
            ),
            "next_generation_same_idea_refinement_count": (
                raw_next.same_idea_refinement_count if raw_next is not None else 0
            ),
            "carried_forward_idea_count": persistence.carried_forward_count,
            "replaced_parent_idea_count": persistence.replaced_parent_count,
            "generated_child_idea_count": persistence.generated_child_count,
            "next_generation_population_count": persistence.final_population_count,
            "next_generation_execution": str(next_execution_path),
        }
    )
    _write(args.output.expanduser().resolve(), summary)

    print()
    print("=== SIS-v3.4 NOVELTY-AWARE BOUNDED REPRODUCTION ===")
    print("generation:", generation, "->", generation + 1)
    print("current population:", len(nodes))
    print("materialized hypotheses:", len(portfolio.hypotheses))
    print("search priority:", pressure.search_priority_counts)
    print("selected parents:", list(next_plan.selected_parent_idea_ids))
    print("generation LLM calls:", summary["next_generation_llm_call_count"])
    print("genuine children:", summary["next_generation_genuine_child_count"])
    print("carried forward:", persistence.carried_forward_count)
    print("replaced parents:", persistence.replaced_parent_count)
    print("next population:", persistence.final_population_count)
    print("PRIOR_ART_POSITIVE_PREMISE_AUTHORITY=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("output:", args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
