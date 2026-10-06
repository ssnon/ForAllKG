from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.research_idea_evolutionary_policy import (
    EvolutionaryIdeaSearchShadowReport,
)
from pipeline_core.discovery.research_idea_offspring_execution import (
    InstructorOpenAICompatibleOffspringBackend,
    build_g3_feedback_seed,
    build_offspring_generation_plan,
    execute_offspring_plan,
    make_hypothesis_backend,
    realize_offspring_bounded,
)
from pipeline_core.discovery.research_idea_search_contracts import (
    GenerationalIdeaSearchShadowReport,
)


def _write(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load(path: Path, model):
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    label, raw = value.split("=", 1)
    label = label.strip()
    raw = raw.strip()
    if not label or not raw:
        raise argparse.ArgumentTypeError("--case must use LABEL=/path/to/case-root")
    return label, Path(raw).expanduser().resolve()


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
            "Execute SIS-v2.3 actual G2 ResearchIdea offspring generation, post-hoc "
            "semantic identity audit, bounded grounded hypothesis realization, and a "
            "typed G3 feedback seed across existing SIS-v2.2 cases."
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
    p.add_argument("--max-raw-offspring", type=int, default=12)
    p.add_argument("--max-realizations", type=int, default=6)
    p.add_argument("--max-realizations-per-parent", type=int, default=2)
    p.add_argument("--semantic-retries", type=int, default=1)
    p.add_argument("--realization-repairs", type=int, default=1)
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
    p.add_argument("--realization-model", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--instructor-mode", default="JSON")
    p.add_argument("--offspring-temperature", type=float, default=0.2)
    p.add_argument("--realization-temperature", type=float, default=0.0)
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--header", action="append", default=[], type=_header)
    p.add_argument("--output", required=True, type=Path)
    return p


def _case_paths(root: Path) -> dict[str, Path]:
    portfolio = root / "scientific_portfolio_shadow"
    return {
        "source_v211": portfolio / "sis_v2_2.source_v2_1_1.generational_shadow.json",
        "v22": portfolio / "sis_v2_2.evolutionary_search_shadow.json",
        "plan": portfolio / "sis_v2_3.offspring_plan.json",
        "execution": portfolio / "sis_v2_3.offspring_execution.json",
        "realization": portfolio / "sis_v2_3.realization.report.json",
        "portfolio": portfolio / "sis_v2_3.materialized.shadow.portfolio.json",
        "g3_seed": portfolio / "sis_v2_3.g3_feedback_seed.json",
        "prompts": portfolio / "sis_v2_3_prompts",
        "offspring_telemetry": portfolio / "sis_v2_3.offspring.telemetry.jsonl",
        "realization_telemetry": portfolio / "sis_v2_3.realization.telemetry.jsonl",
    }


def _find_context(
    *,
    root: Path,
    context_id: str,
    context_sha256: str,
    override: Path | None,
) -> tuple[HypothesisContext, list[str]]:
    if override is not None:
        context = _load(override.expanduser().resolve(), HypothesisContext)
        if context.context_id != context_id or context.context_sha256 != context_sha256:
            raise ValueError(
                "context override does not match ResearchIdea source context lineage"
            )
        return context, [str(override.expanduser().resolve())]

    matches: list[tuple[Path, HypothesisContext]] = []
    for path in root.rglob("*.json"):
        # Large evaluation trees can contain many JSONs. Parse only files whose
        # first chunk advertises the expected schema/context shape.
        try:
            with path.open("r", encoding="utf-8") as handle:
                prefix = handle.read(16384)
            if '"hypothesis-context-v1"' not in prefix:
                continue
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
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
        matches.append((path.resolve(), context))

    if not matches:
        raise FileNotFoundError(
            "No HypothesisContext with exact context_id/context_sha256 lineage "
            f"was found under {root}. Use --context LABEL=/path/to/context.json."
        )
    # Multiple exact-lineage copies are equivalent for this shadow run. Pick a
    # deterministic path and retain every matching path in the cohort report.
    matches.sort(key=lambda item: (len(str(item[0])), str(item[0])))
    return matches[0][1], [str(path) for path, _ in matches]


def _write_prompts(
    directory: Path,
    *,
    offspring_prompts,
    realization_prompts,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(offspring_prompts, start=1):
        (directory / f"offspring_{index:02d}_{prompt.task_id.split(':')[-1]}.txt").write_text(
            "SYSTEM\n======\n"
            + prompt.system_prompt
            + "\n\nUSER\n====\n"
            + prompt.user_prompt
            + "\n",
            encoding="utf-8",
        )
    for index, prompt in enumerate(realization_prompts, start=1):
        (directory / f"realization_{index:02d}.txt").write_text(
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
        "max_raw_offspring",
        "max_realizations",
        "max_realizations_per_parent",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.semantic_retries < 0 or args.realization_repairs < 0:
        raise ValueError("retry/repair counts must be >= 0")
    if not args.prepare_only and not args.model:
        raise SystemExit(
            "--model is required unless GRAPHAGENTS_HYPOTHESIS_MODEL or "
            "OPENROUTER_AGENT_MODEL is set"
        )

    overrides = {label: path for label, path in args.context}
    headers = dict(args.header)
    cohort: dict[str, Any] = {
        "schema_version": "sis-v2-3-generational-offspring-cohort-v1",
        "case_count": len(args.case),
        "cases": {},
        "max_raw_offspring": args.max_raw_offspring,
        "max_realizations": args.max_realizations,
        "semantic_retry_limit": args.semantic_retries,
        "realization_repair_limit": args.realization_repairs,
        "conceptual_generation_is_inspiration_only": True,
        "grounded_premise_boundary_enforced_at_realization": True,
        "semantic_identity_is_posthoc_diagnostic": True,
        "verification_is_not_fertility_authority": True,
        "production_generation_authority": False,
        "production_selection_authority": False,
        "scientific_truth_authority": False,
        "novelty_authority": False,
    }

    for label, root in args.case:
        paths = _case_paths(root)
        missing = [
            str(paths[name])
            for name in ("source_v211", "v22")
            if not paths[name].is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{label} is missing SIS-v2.2 prerequisite artifacts: {missing}"
            )

        generational = _load(paths["source_v211"], GenerationalIdeaSearchShadowReport)
        evolutionary = _load(paths["v22"], EvolutionaryIdeaSearchShadowReport)
        if not generational.research_ideas:
            raise ValueError(f"{label} generational report has no ResearchIdea population")
        first = generational.research_ideas[0]
        context, context_matches = _find_context(
            root=root,
            context_id=first.source_context_id,
            context_sha256=first.source_context_sha256,
            override=overrides.get(label),
        )

        plan = build_offspring_generation_plan(
            evolutionary_report=evolutionary,
            generational_report=generational,
            max_raw_offspring=args.max_raw_offspring,
        )
        _write(paths["plan"], plan)

        print()
        print("=" * 88)
        print(label)
        print("=" * 88)
        print("context:", context_matches[0])
        if len(context_matches) > 1:
            print("equivalent context copies:", len(context_matches))
        print("G2 parent allocations:", evolutionary.selected_parent_channel_counts)
        print("offspring tasks:", plan.task_count_by_channel)
        print("planned max offspring:", plan.planned_max_offspring)

        if args.prepare_only:
            cohort["cases"][label] = {
                "case_root": str(root),
                "context_path": context_matches[0],
                "equivalent_context_paths": context_matches,
                "offspring_plan": str(paths["plan"]),
                "task_count_by_channel": plan.task_count_by_channel,
                "planned_max_offspring": plan.planned_max_offspring,
                "prepare_only": True,
            }
            continue

        offspring_backend = InstructorOpenAICompatibleOffspringBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.offspring_temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=headers,
            telemetry_path=paths["offspring_telemetry"],
            telemetry_context={"case": label, "source_v2_2_report_id": evolutionary.report_id},
        )
        execution, offspring_prompts = execute_offspring_plan(
            plan=plan,
            evolutionary_report=evolutionary,
            generational_report=generational,
            research_question=context.question,
            backend=offspring_backend,
            semantic_retry_limit=args.semantic_retries,
        )
        _write(paths["execution"], execution)

        realization_model = args.realization_model or args.model
        hypothesis_backend = make_hypothesis_backend(
            model=realization_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode=args.instructor_mode,
            temperature=args.realization_temperature,
            parse_retries=args.parse_retries,
            timeout=args.timeout,
            extra_headers=headers,
            telemetry_path=paths["realization_telemetry"],
            telemetry_context={"case": label, "source_g2_execution_report_id": execution.report_id},
        )
        realization, portfolio, realization_prompts = realize_offspring_bounded(
            execution=execution,
            context=context,
            backend=hypothesis_backend,
            max_realizations=args.max_realizations,
            max_per_parent=args.max_realizations_per_parent,
            max_repair_attempts=args.realization_repairs,
        )
        _write(paths["realization"], realization)
        _write(paths["portfolio"], portfolio)

        g3_seed = build_g3_feedback_seed(
            execution=execution,
            realization=realization,
        )
        _write(paths["g3_seed"], g3_seed)
        if args.save_prompts:
            _write_prompts(
                paths["prompts"],
                offspring_prompts=offspring_prompts,
                realization_prompts=realization_prompts,
            )

        cohort["cases"][label] = {
            "case_root": str(root),
            "context_path": context_matches[0],
            "equivalent_context_paths": context_matches,
            "source_v2_2_report": str(paths["v22"]),
            "offspring_plan": str(paths["plan"]),
            "offspring_execution": str(paths["execution"]),
            "realization_report": str(paths["realization"]),
            "materialized_portfolio": str(paths["portfolio"]),
            "g3_feedback_seed": str(paths["g3_seed"]),
            "raw_offspring_count": execution.raw_offspring_count,
            "accepted_for_realization_count": execution.accepted_for_realization_count,
            "identity_relation_counts": execution.identity_relation_counts,
            "semantic_disposition_counts": execution.disposition_counts,
            "generated_count_by_channel": execution.generated_count_by_channel,
            "generated_count_by_operator": execution.generated_count_by_operator,
            "semantic_noop_count": execution.semantic_noop_count,
            "mutation_attempt_count": execution.mutation_attempt_count,
            "distinct_child_count": execution.distinct_child_count,
            "mutation_semantic_yield_fraction": execution.mutation_semantic_yield_fraction,
            "indeterminate_probe_count": execution.indeterminate_probe_count,
            "channel_drift_child_count": execution.channel_drift_child_count,
            "semantic_retry_count": execution.semantic_retry_count,
            "offspring_llm_call_count": execution.llm_call_count,
            "realization_selected_count": realization.selected_idea_count,
            "realization_status_counts": realization.status_counts,
            "materialized_hypothesis_count": realization.materialized_hypothesis_count,
            "materialization_success_fraction": realization.materialization_success_fraction,
            "realization_llm_call_count": realization.llm_call_count,
            "g3_observation_count": g3_seed.observation_count,
        }

        print("raw G2 offspring:", execution.raw_offspring_count)
        print("identity:", execution.identity_relation_counts)
        print("semantic dispositions:", execution.disposition_counts)
        print("semantic no-ops:", execution.semantic_noop_count)
        print(
            "mutation semantic yield:",
            f"{execution.distinct_child_count}/{execution.mutation_attempt_count}",
            f"({execution.mutation_semantic_yield_fraction:.3f})",
        )
        print("indeterminate probes retained:", execution.indeterminate_probe_count)
        print("semantic retries:", execution.semantic_retry_count)
        print("accepted for bounded realization:", execution.accepted_for_realization_count)
        print("realization selected:", realization.selected_idea_count)
        print("realization status:", realization.status_counts)
        print(
            "materialized hypotheses:",
            realization.materialized_hypothesis_count,
            f"({realization.materialization_success_fraction:.3f})",
        )
        print("G3 feedback observations:", g3_seed.observation_count)
        print("OFFSPRING_LLM_CALLS:", execution.llm_call_count)
        print("REALIZATION_LLM_CALLS:", realization.llm_call_count)
        print("PRODUCTION_GENERATION_AUTHORITY=False")
        print("PRODUCTION_SELECTION_AUTHORITY=False")

    cohort["new_llm_calls"] = not args.prepare_only
    _write(args.output.expanduser().resolve(), cohort)
    print()
    print("SIS-v2.3 generational offspring cohort complete")
    print("cases:", len(args.case))
    print("PREPARE_ONLY=", args.prepare_only)
    print("PRODUCTION_GENERATION_AUTHORITY=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
