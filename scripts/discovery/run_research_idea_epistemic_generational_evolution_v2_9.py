from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
    EpistemicG4GenerationPlan,
    build_epistemic_g4_generation_plan,
    execute_epistemic_g4_generation,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    InstructorOpenAICompatibleOffspringBackend,
    OffspringExecutionReport,
)
from pipeline_core.discovery.research_idea_parallel_partial_search import (
    ParallelPartialSearchReport,
)
from pipeline_core.discovery.research_idea_search_contracts import (
    GenerationalIdeaSearchShadowReport,
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
            "Run SIS-v2.9 Epistemic-to-Generational Evolution. Bounded deterministic "
            "selection chooses a small diverse subset of SIS-v2.8b evolution handoffs; "
            "the existing structured offspring backend then generates actual G4 "
            "ResearchIdea children. Partial/speculative realization feedback and "
            "epistemic debt are search context only, never positive premises. No "
            "evidence acquisition, hypothesis grounding, or canonical graph mutation "
            "is executed by this stage."
        )
    )
    p.add_argument("--case", action="append", type=_case, required=True)
    p.add_argument(
        "--context",
        action="append",
        type=_context_override,
        default=[],
        help="Optional LABEL=/path/to/hypothesis-context.json override.",
    )
    p.add_argument(
        "--max-parents-per-case",
        type=int,
        default=4,
        help="Maximum v2.8b handoff parents sent to actual G4 generation per case.",
    )
    p.add_argument(
        "--max-outputs-per-parent",
        type=int,
        default=1,
        choices=(1, 2),
    )
    p.add_argument("--semantic-retries", type=int, default=1)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--save-prompts", action="store_true")
    p.add_argument(
        "--model",
        default=(
            os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL")
            or os.getenv("OPENROUTER_AGENT_MODEL")
            or ""
        ),
    )
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    p.add_argument("--output", type=Path, required=True)
    return p


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "None of the expected artifact paths exist: " + ", ".join(map(str, paths))
    )


def _case_paths(root: Path) -> dict[str, Path]:
    out = root / "scientific_portfolio_shadow"
    work = out / "sis_v2_9_epistemic_generational_evolution"
    return {
        "out": out,
        "source_generational": _first_existing(
            out / "sis_v2_2.source_v2_1_1.generational_shadow.json",
            out / "sis_v2_1_1.feedback_aware.generational_shadow.json",
            out / "sis_v2_1.generational_shadow.json",
        ),
        "g2_execution": out / "sis_v2_3.offspring_execution.json",
        "g3_execution": out / "sis_v2_4.g3_offspring_execution.json",
        "parallel": out / "sis_v2_8b.parallel_partial_search.json",
        "plan": out / "sis_v2_9.g4_epistemic_evolution_plan.json",
        "execution": out / "sis_v2_9.g4_epistemic_evolution_execution.json",
        "population": out / "sis_v2_9.g4_population.json",
        "summary": out / "sis_v2_9.case_summary.json",
        "prompts": work / "prompts",
        "telemetry": work / "offspring.telemetry.jsonl",
    }


def _research_ideas(
    source: GenerationalIdeaSearchShadowReport,
    g2: OffspringExecutionReport,
    g3: OffspringExecutionReport,
) -> dict[str, Any]:
    rows = [*source.research_ideas, *g2.offspring_nodes, *g3.offspring_nodes]
    return {row.idea_id: row for row in rows}


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
            raise ValueError(
                "context override does not match ResearchIdea source context lineage"
            )
        return context, [str(path)]

    matches: list[tuple[Path, HypothesisContext]] = []
    preferred = [
        root / "hypothesis.context.json",
        root / "scientific_portfolio_shadow" / "hypothesis.context.json",
    ]
    scanned: set[Path] = set()
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
            "No exact-lineage HypothesisContext found under case root. "
            "Use --context LABEL=/path/to/context.json."
        )
    matches.sort(key=lambda item: (len(str(item[0])), str(item[0])))
    return matches[0][1], [str(path) for path, _ in matches]


def _write_prompts(directory: Path, prompts) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts, start=1):
        suffix = prompt.task_id.split(":")[-1]
        (directory / f"g4_{index:02d}_{suffix}.txt").write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )


def _summary(
    *,
    plan: EpistemicG4GenerationPlan,
    execution: EpistemicG4ExecutionReport | None,
    context_path: str,
    prepare_only: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "sis-v2-9-epistemic-generational-case-summary-v1",
        "source_parallel_report_id": plan.source_parallel_report_id,
        "available_handoff_count": plan.available_handoff_count,
        "selected_parent_count": plan.task_count,
        "selected_parent_idea_ids": plan.selected_parent_idea_ids,
        "selected_parent_source_mode_counts": plan.selected_parent_source_mode_counts,
        "task_count_by_channel": plan.task_count_by_channel,
        "planned_max_offspring": plan.planned_max_offspring,
        "context_path": context_path,
        "prepare_only": prepare_only,
        "grounding_required_for_child_generation": False,
        "grounding_required_before_scientific_claim": True,
        "epistemic_feedback_used_as_search_context_only": True,
        "evidence_acquisition_executed": False,
        "canonical_graph_mutated": False,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }
    if execution is not None:
        payload.update(
            {
                "report_id": execution.report_id,
                "raw_offspring_count": execution.raw_offspring_count,
                "g4_population_count": execution.g4_population_count,
                "genuine_child_count": execution.genuine_child_count,
                "indeterminate_probe_count": execution.indeterminate_probe_count,
                "same_idea_refinement_count": execution.same_idea_refinement_count,
                "identity_relation_counts": execution.identity_relation_counts,
                "disposition_counts": execution.disposition_counts,
                "generated_count_by_channel": execution.generated_count_by_channel,
                "generated_count_by_operator": execution.generated_count_by_operator,
                "genuine_child_from_non_grounded_feedback_count": (
                    execution.genuine_child_from_non_grounded_feedback_count
                ),
                "genuine_child_from_speculative_feedback_count": (
                    execution.genuine_child_from_speculative_feedback_count
                ),
                "genuine_child_from_evidence_seeking_feedback_count": (
                    execution.genuine_child_from_evidence_seeking_feedback_count
                ),
                "llm_call_count": execution.llm_call_count,
                "semantic_retry_count": execution.semantic_retry_count,
                "offspring_generation_executed": True,
            }
        )
    else:
        payload["offspring_generation_executed"] = False
    return payload


def main() -> int:
    args = parser().parse_args()
    if args.max_parents_per_case < 1:
        raise SystemExit("--max-parents-per-case must be >= 1")
    if args.semantic_retries < 0:
        raise SystemExit("--semantic-retries must be >= 0")
    if not args.prepare_only and not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    overrides = {label: path for label, path in args.context}
    headers = dict(args.header)
    cohort: dict[str, Any] = {
        "schema_version": "sis-v2-9-epistemic-to-generational-evolution-cohort-v1",
        "case_count": len(args.case),
        "cases": {},
        "max_parents_per_case": args.max_parents_per_case,
        "max_outputs_per_parent": args.max_outputs_per_parent,
        "semantic_retry_limit": args.semantic_retries,
        "grounding_required_for_child_generation": False,
        "grounding_required_before_scientific_claim": True,
        "epistemic_feedback_used_as_search_context_only": True,
        "evidence_acquisition_executed": False,
        "canonical_graph_mutated": False,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }
    aggregate = Counter()

    for label, root in args.case:
        paths = _case_paths(root)
        missing = [
            str(paths[name])
            for name in ("g2_execution", "g3_execution", "parallel")
            if not paths[name].is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{label} missing SIS prerequisites: {missing}")

        source = _load(paths["source_generational"], GenerationalIdeaSearchShadowReport)
        g2 = _load(paths["g2_execution"], OffspringExecutionReport)
        g3 = _load(paths["g3_execution"], OffspringExecutionReport)
        parallel = _load(paths["parallel"], ParallelPartialSearchReport)
        parent_by_id = _research_ideas(source, g2, g3)

        missing_parents = sorted(
            {
                handoff.idea_id
                for handoff in parallel.evolution_handoffs
                if handoff.idea_id not in parent_by_id
            }
        )
        if missing_parents:
            raise ValueError(
                f"{label}: v2.8b handoff parents missing from G0-G3 ResearchIdea population: "
                f"{missing_parents[:8]}"
            )

        if not parallel.evolution_handoffs:
            raise ValueError(f"{label}: v2.8b report contains no evolution handoffs")
        first_parent = parent_by_id[parallel.evolution_handoffs[0].idea_id]
        context, context_matches = _find_context(
            root=root,
            context_id=first_parent.source_context_id,
            context_sha256=first_parent.source_context_sha256,
            override=overrides.get(label),
        )

        plan = build_epistemic_g4_generation_plan(
            parallel_report=parallel,
            parent_by_id=parent_by_id,
            max_parents=args.max_parents_per_case,
            max_outputs_per_parent=args.max_outputs_per_parent,
        )
        _write(paths["plan"], plan)

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print("context:", context_matches[0])
        if len(context_matches) > 1:
            print("equivalent context copies:", len(context_matches))
        print("available v2.8b handoffs:", plan.available_handoff_count)
        print("selected G4 parents:", plan.task_count, "/", plan.candidate_parent_count)
        print("parent source modes:", plan.selected_parent_source_mode_counts)
        print("G4 task channels:", plan.task_count_by_channel)
        print("planned max G4 offspring:", plan.planned_max_offspring)

        execution: EpistemicG4ExecutionReport | None = None
        prompts = ()
        if not args.prepare_only:
            backend = InstructorOpenAICompatibleOffspringBackend(
                model=args.model,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                instructor_mode=args.instructor_mode,
                temperature=args.temperature,
                parse_retries=args.parse_retries,
                timeout=args.timeout,
                extra_headers=headers,
                telemetry_path=paths["telemetry"],
                telemetry_context={
                    "pipeline": "sis_v2_9_epistemic_to_generational_evolution",
                    "case": label,
                    "source_parallel_report_id": parallel.report_id,
                    "generation": 4,
                },
            )
            execution, prompts = execute_epistemic_g4_generation(
                plan=plan,
                parallel_report=parallel,
                parent_by_id=parent_by_id,
                research_question=context.question,
                backend=backend,
                semantic_retry_limit=args.semantic_retries,
            )
            _write(paths["execution"], execution)
            _write(
                paths["population"],
                {
                    "schema_version": "sis-v2-9-g4-research-idea-population-v1",
                    "source_execution_report_id": execution.report_id,
                    "generation_index": 4,
                    "research_ideas": [
                        row.model_dump(mode="json")
                        if hasattr(row, "model_dump")
                        else row
                        for row in execution.g4_population_nodes
                    ],
                    "research_idea_count": execution.g4_population_count,
                    "genuine_child_count": execution.genuine_child_count,
                    "indeterminate_probe_count": execution.indeterminate_probe_count,
                    "inspiration_only": True,
                    "grounding_required_before_scientific_claim": True,
                    "canonical_graph_mutated": False,
                    "production_generation_authority": False,
                    "production_selection_authority": False,
                },
            )
            if args.save_prompts:
                _write_prompts(paths["prompts"], prompts)

            print("raw G4 offspring:", execution.raw_offspring_count)
            print("identity:", execution.identity_relation_counts)
            print("semantic dispositions:", execution.disposition_counts)
            print("genuine G4 children:", execution.genuine_child_count)
            print("indeterminate G4 probes:", execution.indeterminate_probe_count)
            print("same-idea returns to realization lane:", execution.same_idea_refinement_count)
            print(
                "genuine children from non-grounded feedback:",
                execution.genuine_child_from_non_grounded_feedback_count,
            )
            print(
                "  speculative / evidence-seeking:",
                execution.genuine_child_from_speculative_feedback_count,
                "/",
                execution.genuine_child_from_evidence_seeking_feedback_count,
            )
            print("G4 OFFSPRING_LLM_CALLS:", execution.llm_call_count)
            print("semantic retries:", execution.semantic_retry_count)

        summary = _summary(
            plan=plan,
            execution=execution,
            context_path=context_matches[0],
            prepare_only=args.prepare_only,
        )
        _write(paths["summary"], summary)
        cohort["cases"][label] = {
            "case_root": str(root),
            "plan": str(paths["plan"]),
            "execution": None if execution is None else str(paths["execution"]),
            "g4_population": None if execution is None else str(paths["population"]),
            "summary": summary,
        }

        aggregate["case_count"] += 1
        aggregate["available_handoff_count"] += plan.available_handoff_count
        aggregate["selected_parent_count"] += plan.task_count
        aggregate["planned_max_offspring"] += plan.planned_max_offspring
        if execution is not None:
            aggregate["raw_offspring_count"] += execution.raw_offspring_count
            aggregate["g4_population_count"] += execution.g4_population_count
            aggregate["genuine_child_count"] += execution.genuine_child_count
            aggregate["indeterminate_probe_count"] += execution.indeterminate_probe_count
            aggregate["same_idea_refinement_count"] += execution.same_idea_refinement_count
            aggregate["genuine_child_from_non_grounded_feedback_count"] += (
                execution.genuine_child_from_non_grounded_feedback_count
            )
            aggregate["genuine_child_from_speculative_feedback_count"] += (
                execution.genuine_child_from_speculative_feedback_count
            )
            aggregate["genuine_child_from_evidence_seeking_feedback_count"] += (
                execution.genuine_child_from_evidence_seeking_feedback_count
            )
            aggregate["llm_call_count"] += execution.llm_call_count
            aggregate["semantic_retry_count"] += execution.semantic_retry_count

        print("GROUNDING_REQUIRED_FOR_CHILD_GENERATION=False")
        print("GROUNDING_REQUIRED_BEFORE_SCIENTIFIC_CLAIM=True")
        print("EPISTEMIC_FEEDBACK_POSITIVE_PREMISE_AUTHORITY=False")
        print("EVIDENCE_ACQUISITION_EXECUTED=False")
        print("CANONICAL_GRAPH_MUTATED=False")
        print("PRODUCTION_GENERATION_AUTHORITY=False")

    cohort["aggregate"] = dict(aggregate)
    cohort["offspring_generation_executed"] = not args.prepare_only
    cohort["new_llm_calls"] = not args.prepare_only
    output = args.output.expanduser().resolve()
    _write(output, cohort)
    print()
    print("SIS-v2.9 Epistemic-to-Generational Evolution complete")
    print("aggregate:", dict(aggregate))
    print("PREPARE_ONLY=", args.prepare_only)
    print("GROUNDING_REQUIRED_FOR_CHILD_GENERATION=False")
    print("EVIDENCE_ACQUISITION_EXECUTED=False")
    print("CANONICAL_GRAPH_MUTATED=False")
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
