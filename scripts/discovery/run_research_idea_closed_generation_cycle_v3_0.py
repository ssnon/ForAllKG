from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.prospective_identification_materialization_shadow import (
    run_prospective_identification_shadow,
)
from pipeline_core.discovery.research_idea_closed_generation_cycle import (
    adapt_epistemic_execution_for_realization,
    build_cycle_epistemic_artifacts,
    build_cycle_report,
    population_nodes,
)
from pipeline_core.discovery.research_idea_adaptive_fertility import (
    build_adaptive_fertility_report,
    build_fertility_gated_parallel_view,
    compose_persistent_population_execution,
)
from pipeline_core.discovery.research_idea_closed_loop import run_realization_lifecycle
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
    build_epistemic_generation_plan,
    execute_epistemic_generation,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    InstructorOpenAICompatibleOffspringBackend,
)


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
        raise FileNotFoundError(f"Required SIS artifact is missing: {path}")
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    label, raw = value.split("=", 1)
    if not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label.strip(), Path(raw).expanduser().resolve()


def _execution_override(value: str) -> tuple[str, Path]:
    return _case(value)


def _context_override(value: str) -> tuple[str, Path]:
    return _case(value)


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
            "Run SIS-v3.0 Closed Generational Cycle. A retained epistemic generation "
            "(G4 by default, or a later v3.0 generation via --execution) is sent through "
            "the existing strict ResearchIdea realization lifecycle, decomposed into "
            "grounded/partial/speculative epistemic realizations, archived, and returned "
            "to parallel same-idea/child-idea search. The resulting handoffs can then "
            "optionally generate the next ResearchIdea generation. No evidence acquisition "
            "or canonical KG mutation is performed."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument(
        "--execution",
        action="append",
        type=_execution_override,
        default=[],
        help=(
            "Optional LABEL=/path/to/epistemic-generation-execution.json. "
            "Without this, each case uses sis_v2_9.g4_epistemic_evolution_execution.json."
        ),
    )
    p.add_argument("--context", action="append", type=_context_override, default=[])
    p.add_argument("--max-ideas", type=int, default=8)
    p.add_argument("--max-realizations-per-idea", type=int, default=2)
    p.add_argument("--max-realizations-per-parent", type=int, default=2)
    p.add_argument("--realization-repairs", type=int, default=1)
    p.add_argument("--acquisition-persistence-threshold", type=int, default=2)
    p.add_argument(
        "--adaptive-fertility",
        action="store_true",
        help=(
            "Enable SIS-v3.1 ACTIVE != FERTILE policy. Stable or still-local-searchable "
            "ResearchIdeas persist without automatic child generation; only typed "
            "fertility decisions may enter the next-generation offspring plan."
        ),
    )
    p.add_argument(
        "--population-growth-budget",
        type=int,
        default=0,
        help=(
            "Extra active-population slots allowed beyond the current active population "
            "when adaptive fertility is enabled. Zero means one retained child replaces "
            "its fertile parent while nonfertile active ideas persist."
        ),
    )
    p.add_argument("--run-prospective", action="store_true")
    p.add_argument("--max-prospective-audits", type=int, default=8)
    p.add_argument("--generate-next", action="store_true")
    p.add_argument("--max-next-parents", type=int, default=4)
    p.add_argument("--max-next-outputs-per-parent", type=int, default=1, choices=(1, 2))
    p.add_argument("--next-semantic-retries", type=int, default=1)
    p.add_argument("--save-prompts", action="store_true")
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
        help="Default model for realization and next-generation idea generation.",
    )
    p.add_argument("--realization-model", default=None)
    p.add_argument("--generation-model", default=None)
    p.add_argument("--prospective-model", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--realization-temperature", type=float, default=0.0)
    p.add_argument("--generation-temperature", type=float, default=0.2)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    p.add_argument("--output", required=True, type=Path)
    return p


def _find_context(
    *,
    root: Path,
    context_id: str,
    context_sha256: str,
    override: Path | None,
) -> tuple[HypothesisContext, list[str]]:
    if override is not None:
        path = override.expanduser().resolve()
        context = _load(path, HypothesisContext)
        if context.context_id != context_id or context.context_sha256 != context_sha256:
            raise ValueError("context override does not match ResearchIdea source lineage")
        return context, [str(path)]

    matches: list[tuple[Path, HypothesisContext]] = []
    scanned: set[Path] = set()
    preferred = [
        root / "hypothesis.context.json",
        root / "scientific_portfolio_shadow" / "hypothesis.context.json",
    ]
    for path in [*preferred, *root.rglob("*.json")]:
        path = path.resolve()
        if path in scanned or not path.is_file():
            continue
        scanned.add(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                prefix = handle.read(16384)
            if '"hypothesis-context-v1"' not in prefix:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != "hypothesis-context-v1":
            continue
        if str(payload.get("context_id") or "") != context_id:
            continue
        if str(payload.get("context_sha256") or "") != context_sha256:
            continue
        try:
            context = HypothesisContext.model_validate(payload)
        except Exception:
            continue
        matches.append((path, context))
    if not matches:
        raise FileNotFoundError(
            "No exact-lineage HypothesisContext found. Use --context LABEL=/path/to/context.json."
        )
    matches.sort(key=lambda item: (len(str(item[0])), str(item[0])))
    return matches[0][1], [str(path) for path, _ in matches]


def _typed_population(execution: EpistemicG4ExecutionReport) -> list[ResearchIdeaNode]:
    return [ResearchIdeaNode.model_validate(row) for row in population_nodes(execution)]


def _collect_history(root: Path, current: EpistemicG4ExecutionReport) -> dict[str, ResearchIdeaNode]:
    out = root / "scientific_portfolio_shadow"
    paths = [
        out / "sis_v2_2.source_v2_1_1.generational_shadow.json",
        out / "sis_v2_1_1.feedback_aware.generational_shadow.json",
        out / "sis_v2_1.generational_shadow.json",
        out / "sis_v2_3.offspring_execution.json",
        out / "sis_v2_4.g3_offspring_execution.json",
        out / "sis_v2_9.g4_epistemic_evolution_execution.json",
        *sorted(out.glob("sis_v3_0.g*_generation_execution.json")),
        *sorted(out.glob("sis_v3_1.g*_adaptive_population_execution.json")),
    ]
    nodes: dict[str, ResearchIdeaNode] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        rows = []
        rows.extend(payload.get("research_ideas") or [])
        rows.extend(payload.get("offspring_nodes") or [])
        rows.extend(payload.get("g4_population_nodes") or [])
        for row in rows:
            try:
                node = ResearchIdeaNode.model_validate(row)
            except Exception:
                continue
            nodes[node.idea_id] = node
    for node in _typed_population(current):
        nodes[node.idea_id] = node
    return nodes


def _write_prompts(directory: Path, prompts: tuple[Any, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts, start=1):
        suffix = prompt.task_id.split(":")[-1]
        (directory / f"next_{index:02d}_{suffix}.txt").write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )


def main() -> int:
    args = parser().parse_args()
    for name in (
        "max_ideas",
        "max_realizations_per_idea",
        "max_realizations_per_parent",
        "acquisition_persistence_threshold",
        "max_next_parents",
        "max_next_outputs_per_parent",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.realization_repairs < 0 or args.next_semantic_retries < 0:
        raise ValueError("repair/retry counts must be >= 0")
    if args.max_prospective_audits < 0:
        raise ValueError("--max-prospective-audits must be >= 0")
    if args.population_growth_budget < 0:
        raise ValueError("--population-growth-budget must be >= 0")
    if not args.model and not args.realization_model:
        raise SystemExit("A realization model is required via --model/--realization-model or env")
    if args.generate_next and not (args.generation_model or args.model):
        raise SystemExit("--generate-next requires --model/--generation-model or env")

    execution_overrides = {label: path for label, path in args.execution}
    context_overrides = {label: path for label, path in args.context}
    headers = dict(args.header)
    cohort: dict[str, Any] = {
        "schema_version": (
            "sis-v3-1-adaptive-fertility-cycle-cohort-v1"
            if args.adaptive_fertility
            else "sis-v3-0-closed-generation-cycle-cohort-v1"
        ),
        "case_count": len(args.case),
        "cases": {},
        "generate_next": args.generate_next,
        "adaptive_fertility_enabled": args.adaptive_fertility,
        "active_is_not_equal_to_fertile": True,
        "feedback_loop_closed": True,
        "evidence_acquisition_is_not_blocking_inner_loop": True,
        "strict_hypothesis_card_contract_preserved": True,
        "canonical_graph_mutated": False,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }
    aggregate = {
        "population_idea_count": 0,
        "realization_attempt_count": 0,
        "materialized_hypothesis_count": 0,
        "usable_grounded_realization_count": 0,
        "epistemic_realization_count": 0,
        "active_idea_count": 0,
        "next_generation_handoff_count": 0,
        "fertile_idea_count": 0,
        "persistent_nonfertile_idea_count": 0,
        "next_generation_selected_parent_count": 0,
        "next_generation_raw_offspring_count": 0,
        "next_generation_genuine_child_count": 0,
        "carried_forward_idea_count": 0,
        "replaced_parent_idea_count": 0,
        "realization_llm_call_count": 0,
        "next_generation_llm_call_count": 0,
    }

    for label, root in args.case:
        out = root / "scientific_portfolio_shadow"
        execution_path = execution_overrides.get(
            label,
            out / "sis_v2_9.g4_epistemic_evolution_execution.json",
        ).expanduser().resolve()
        execution = _load(execution_path, EpistemicG4ExecutionReport)
        generation = execution.generation_index
        population = _typed_population(execution)
        if not population:
            raise ValueError(f"{label}: generation G{generation} has no retained population")
        first = population[0]
        context, context_matches = _find_context(
            root=root,
            context_id=first.source_context_id,
            context_sha256=first.source_context_sha256,
            override=context_overrides.get(label),
        )

        cycle_tag = "sis_v3_1" if args.adaptive_fertility else "sis_v3_0"
        work = out / f"{cycle_tag}_closed_generation_cycle" / f"g{generation}"
        legacy_execution = adapt_epistemic_execution_for_realization(execution)
        realization_model = args.realization_model or args.model
        realization_backend = InstructorOpenAICompatibleHypothesisBackend(
            model=realization_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.realization_temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=headers,
            telemetry_path=work / "realization.telemetry.jsonl",
            telemetry_context={
                "pipeline": (
                    "sis_v3_1_adaptive_fertility_cycle_realization"
                    if args.adaptive_fertility
                    else "sis_v3_0_closed_generation_cycle_realization"
                ),
                "case": label,
                "generation": generation,
            },
        )

        prospective_runner = None
        if args.run_prospective:
            prospective_model = args.prospective_model or realization_model

            def prospective_runner(*, context, candidate, source_stage, output_prefix):
                return run_prospective_identification_shadow(
                    context=context,
                    candidate=candidate,
                    source_stage=source_stage,
                    model=prospective_model,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    parse_retries=args.parse_retries,
                    output_prefix=output_prefix,
                )

        lifecycle, portfolio = run_realization_lifecycle(
            execution=legacy_execution,
            context=context,
            backend=realization_backend,
            output_dir=work / "realization",
            max_ideas=min(args.max_ideas, len(population)),
            max_realizations_per_idea=args.max_realizations_per_idea,
            max_per_parent=args.max_realizations_per_parent,
            max_repair_attempts=args.realization_repairs,
            prospective_runner=prospective_runner,
            max_prospective_audits=args.max_prospective_audits,
        )
        _write(out / f"{cycle_tag}.g{generation}_realization_lifecycle.json", lifecycle)
        _write(out / f"{cycle_tag}.g{generation}_materialized_portfolio.json", portfolio)

        decomposition, archive, parallel = build_cycle_epistemic_artifacts(
            execution=execution,
            lifecycle=lifecycle,
            portfolio=portfolio,
            acquisition_persistence_threshold=args.acquisition_persistence_threshold,
        )
        _write(out / f"{cycle_tag}.g{generation}_epistemic_decomposition.json", decomposition)
        _write(out / f"{cycle_tag}.g{generation}_multi_realization_archive.json", archive)
        _write(out / f"{cycle_tag}.g{generation}_parallel_partial_search.json", parallel)
        _write(
            out / f"{cycle_tag}.g{generation}_next_generation_handoffs.json",
            {
                "schema_version": (
                    "sis-v3-1-pre-fertility-handoff-bundle-v1"
                    if args.adaptive_fertility
                    else "sis-v3-0-next-generation-handoff-bundle-v1"
                ),
                "source_parallel_report_id": parallel.report_id,
                "current_generation_index": generation,
                "next_generation_index": generation + 1,
                "handoffs": [row.model_dump(mode="json") for row in parallel.evolution_handoffs],
                "handoff_count": parallel.evolution_handoff_count,
                "grounding_required_for_child_generation": False,
                "grounding_required_before_scientific_claim": True,
                "positive_premise_authority": False,
            },
        )

        fertility = None
        persistence = None
        generation_parallel = parallel
        if args.adaptive_fertility:
            fertility = build_adaptive_fertility_report(
                parallel_report=parallel,
                lifecycle=lifecycle,
                cycle_generation_index=generation,
            )
            _write(
                out / f"sis_v3_1.g{generation}_adaptive_fertility.json",
                fertility,
            )
            generation_parallel = build_fertility_gated_parallel_view(
                parallel_report=parallel,
                fertility_report=fertility,
            )
            _write(
                out / f"sis_v3_1.g{generation}_fertile_handoffs.json",
                {
                    "schema_version": "sis-v3-1-fertility-gated-handoff-bundle-v1",
                    "source_parallel_report_id": parallel.report_id,
                    "source_fertility_report_id": fertility.report_id,
                    "cycle_generation_index": generation,
                    "next_generation_index": generation + 1,
                    "handoffs": [
                        row.model_dump(mode="json") if hasattr(row, "model_dump") else vars(row)
                        for row in generation_parallel.evolution_handoffs
                    ],
                    "handoff_count": len(generation_parallel.evolution_handoffs),
                    "active_idea_count": fertility.active_idea_count,
                    "fertile_idea_count": fertility.fertile_idea_count,
                    "persistent_nonfertile_idea_count": fertility.persistent_nonfertile_idea_count,
                    "active_is_not_equal_to_fertile": True,
                    "positive_premise_authority": False,
                },
            )

        history = _collect_history(root, execution)
        next_plan = build_epistemic_generation_plan(
            parallel_report=generation_parallel,
            parent_by_id=history,
            generation_index=generation + 1,
            max_parents=args.max_next_parents,
            max_outputs_per_parent=args.max_next_outputs_per_parent,
        )
        _write(out / f"{cycle_tag}.g{generation + 1}_generation_plan.json", next_plan)

        next_execution = None
        raw_next_execution = None
        next_prompts: tuple[Any, ...] = ()
        if args.generate_next and next_plan.task_count:
            generation_model = args.generation_model or args.model
            generation_backend = InstructorOpenAICompatibleOffspringBackend(
                model=generation_model,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                instructor_mode=args.instructor_mode,
                temperature=args.generation_temperature,
                parse_retries=args.parse_retries,
                timeout=args.timeout,
                extra_headers=headers,
                telemetry_path=work / "next_generation.telemetry.jsonl",
                telemetry_context={
                    "pipeline": (
                        "sis_v3_1_adaptive_fertility_generation"
                        if args.adaptive_fertility
                        else "sis_v3_0_closed_generation_cycle_generation"
                    ),
                    "case": label,
                    "generation": generation + 1,
                },
            )
            raw_next_execution, next_prompts = execute_epistemic_generation(
                plan=next_plan,
                parallel_report=generation_parallel,
                parent_by_id=history,
                research_question=context.question,
                backend=generation_backend,
                semantic_retry_limit=args.next_semantic_retries,
            )

        if args.generate_next and args.adaptive_fertility:
            next_execution, persistence = compose_persistent_population_execution(
                current_execution=execution,
                fertility_report=fertility,
                next_generation_index=generation + 1,
                next_plan=next_plan,
                raw_next_execution=raw_next_execution,
                population_growth_budget=args.population_growth_budget,
            )
            if raw_next_execution is not None:
                _write(
                    out / f"sis_v3_1.g{generation + 1}_raw_generation_execution.json",
                    raw_next_execution,
                )
            _write(
                out / f"sis_v3_1.g{generation}_population_persistence.json",
                persistence,
            )
        elif raw_next_execution is not None:
            next_execution = raw_next_execution

        if next_execution is not None:
            _write(
                out / (
                    f"sis_v3_1.g{generation + 1}_adaptive_population_execution.json"
                    if args.adaptive_fertility
                    else f"sis_v3_0.g{generation + 1}_generation_execution.json"
                ),
                next_execution,
            )
            _write(
                out / (
                    f"sis_v3_1.g{generation + 1}_adaptive_population.json"
                    if args.adaptive_fertility
                    else f"sis_v3_0.g{generation + 1}_population.json"
                ),
                {
                    "schema_version": (
                        "sis-v3-1-adaptive-persistent-population-v1"
                        if args.adaptive_fertility
                        else "sis-v3-0-epistemic-generation-population-v1"
                    ),
                    "generation_index": generation + 1,
                    "source_execution_report_id": next_execution.report_id,
                    "research_ideas": [
                        row.model_dump(mode="json") if hasattr(row, "model_dump") else row
                        for row in next_execution.g4_population_nodes
                    ],
                    "research_idea_count": next_execution.g4_population_count,
                    "epistemic_status": "INSPIRATION_ONLY",
                    "grounding_required_before_scientific_claim": True,
                    "active_idea_persistence_enabled": bool(args.adaptive_fertility),
                    "production_generation_authority": False,
                },
            )
        if args.save_prompts and next_prompts:
            _write_prompts(work / "next_generation_prompts", next_prompts)

        report = build_cycle_report(
            execution=execution,
            lifecycle=lifecycle,
            decomposition=decomposition,
            archive=archive,
            parallel=parallel,
            next_plan=next_plan,
            next_execution=next_execution,
        )
        _write(out / f"{cycle_tag}.g{generation}_closed_cycle_summary.json", report)

        print()
        print("=" * 88)
        print(label, f"G{generation} -> G{generation + 1}")
        print("=" * 88)
        print("context:", context_matches[0])
        print("current generation population:", report.population_idea_count)
        print("realization targets / attempts:", report.realization_target_count, "/", report.realization_attempt_count)
        print("materialized / usable grounded:", report.materialized_hypothesis_count, "/", report.usable_grounded_realization_count)
        print("same-idea rescues:", report.same_idea_rescue_count)
        print("epistemic maturity:", report.epistemic_maturity_counts)
        print("archive retained:", report.retained_archive_entry_count)
        print("parallel active ideas / members:", report.active_idea_count, "/", report.active_parallel_member_count)
        print("epistemic debts:", report.epistemic_debt_count)
        print("next-generation handoffs:", report.next_generation_handoff_count)
        if fertility is not None:
            print("fertility dispositions:", fertility.disposition_counts)
            print(
                "active / fertile / persistent-nonfertile:",
                fertility.active_idea_count,
                "/",
                fertility.fertile_idea_count,
                "/",
                fertility.persistent_nonfertile_idea_count,
            )
            print("fertility-gated handoffs:", fertility.fertile_handoff_count)
        print("next-generation selected parents:", report.next_generation_selected_parent_count)
        if next_execution is not None:
            print("next-generation raw / population / genuine:", report.next_generation_raw_offspring_count, "/", report.next_generation_population_count, "/", report.next_generation_genuine_child_count)
            print("next-generation LLM calls:", next_execution.llm_call_count)
            if persistence is not None:
                print(
                    "carried-forward / replaced / final population:",
                    persistence.carried_forward_count,
                    "/",
                    persistence.replaced_parent_count,
                    "/",
                    persistence.final_population_count,
                )
                print(
                    "SEARCH_QUIESCENT=",
                    fertility.fertile_idea_count == 0,
                    sep="",
                )
            print("SAME_RUNNER_CAN_CONSUME_NEXT_EXECUTION=True")
        else:
            print("next generation execution: NOT_EXECUTED")
            print("SAME_RUNNER_CAN_CONSUME_NEXT_EXECUTION=False")
        print("FEEDBACK_LOOP_CLOSED=True")
        if args.adaptive_fertility:
            print("ACTIVE_EQUALS_FERTILE=False")
            print("AUTOMATIC_CHILD_FOR_EVERY_ACTIVE_IDEA=False")
        print("EVIDENCE_ACQUISITION_BLOCKING_INNER_LOOP=False")
        print("STRICT_HYPOTHESIS_CARD_CONTRACT_PRESERVED=True")
        print("CANONICAL_GRAPH_MUTATED=False")

        cohort["cases"][label] = {
            "case_root": str(root),
            "source_execution": str(execution_path),
            "context_path": context_matches[0],
            "current_generation_index": generation,
            "next_generation_index": generation + 1,
            "lifecycle": str(out / f"{cycle_tag}.g{generation}_realization_lifecycle.json"),
            "epistemic_decomposition": str(out / f"{cycle_tag}.g{generation}_epistemic_decomposition.json"),
            "parallel_partial_search": str(out / f"{cycle_tag}.g{generation}_parallel_partial_search.json"),
            "next_generation_plan": str(out / f"{cycle_tag}.g{generation + 1}_generation_plan.json"),
            "next_generation_execution": (
                str(out / (
                    f"sis_v3_1.g{generation + 1}_adaptive_population_execution.json"
                    if args.adaptive_fertility
                    else f"sis_v3_0.g{generation + 1}_generation_execution.json"
                ))
                if next_execution is not None
                else None
            ),
            "adaptive_fertility": (
                str(out / f"sis_v3_1.g{generation}_adaptive_fertility.json")
                if fertility is not None
                else None
            ),
            "population_persistence": (
                str(out / f"sis_v3_1.g{generation}_population_persistence.json")
                if persistence is not None
                else None
            ),
            "fertility_summary": (
                fertility.model_dump(mode="json") if fertility is not None else None
            ),
            "summary": report.model_dump(mode="json"),
        }

        aggregate["population_idea_count"] += report.population_idea_count
        aggregate["realization_attempt_count"] += report.realization_attempt_count
        aggregate["materialized_hypothesis_count"] += report.materialized_hypothesis_count
        aggregate["usable_grounded_realization_count"] += report.usable_grounded_realization_count
        aggregate["epistemic_realization_count"] += report.epistemic_realization_count
        aggregate["active_idea_count"] += report.active_idea_count
        aggregate["next_generation_handoff_count"] += report.next_generation_handoff_count
        if fertility is not None:
            aggregate["fertile_idea_count"] += fertility.fertile_idea_count
            aggregate["persistent_nonfertile_idea_count"] += fertility.persistent_nonfertile_idea_count
        aggregate["next_generation_selected_parent_count"] += report.next_generation_selected_parent_count
        aggregate["next_generation_raw_offspring_count"] += report.next_generation_raw_offspring_count
        aggregate["next_generation_genuine_child_count"] += report.next_generation_genuine_child_count
        if persistence is not None:
            aggregate["carried_forward_idea_count"] += persistence.carried_forward_count
            aggregate["replaced_parent_idea_count"] += persistence.replaced_parent_count
        aggregate["realization_llm_call_count"] += report.realization_llm_call_count
        if next_execution is not None:
            aggregate["next_generation_llm_call_count"] += next_execution.llm_call_count

    cohort["aggregate"] = aggregate
    _write(args.output.expanduser().resolve(), cohort)
    print()
    print(
        "SIS-v3.1 Adaptive Fertility Cycle complete"
        if args.adaptive_fertility
        else "SIS-v3.0 Closed Generational Cycle complete"
    )
    print("aggregate:", aggregate)
    print("FEEDBACK_LOOP_CLOSED=True")
    if args.adaptive_fertility:
        print("ACTIVE_EQUALS_FERTILE=False")
        print("AUTOMATIC_CHILD_FOR_EVERY_ACTIVE_IDEA=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("output:", args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
