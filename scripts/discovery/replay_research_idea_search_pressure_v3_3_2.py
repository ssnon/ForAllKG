from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import ExternalNoveltyReport
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
)
from pipeline_core.discovery.research_idea_program_family_v3_3_1 import (
    PROGRAM_FAMILY_SYSTEM_PROMPT,
    ScientificProgramPairBatch,
    build_scientific_program_family_report,
    program_family_prompt_payload,
    validate_program_pair_batch,
)
from pipeline_core.discovery.research_idea_search_assessment import (
    ResearchIdeaSearchAssessmentReport,
)
from pipeline_core.discovery.research_idea_search_pressure_v3_3_2 import (
    build_bounded_parent_schedule,
    build_novelty_depth_signals_by_idea,
    build_research_idea_search_pressure_report,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _must(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _nodes_for_generation(*, e2e: Path, sp: Path, generation: int) -> list[ResearchIdeaNode]:
    if generation == 2:
        payload = _load_json(_must(e2e / "p0.population.json", "P0 population"))
        raw = payload.get("research_ideas") or []
    else:
        execution_path = _must(
            sp / f"sis_v3_1.g{generation}_adaptive_population_execution.json",
            f"G{generation} adaptive population execution",
        )
        execution = EpistemicG4ExecutionReport.model_validate_json(
            execution_path.read_text(encoding="utf-8")
        )
        raw = execution.g4_population_nodes
    return [ResearchIdeaNode.model_validate(row) for row in raw]


def _program_pairs(
    *,
    nodes: list[ResearchIdeaNode],
    out: Path,
    model: str,
    api_key_env: str,
    base_url: str | None,
    parse_retries: int,
    force: bool,
) -> ScientificProgramPairBatch:
    artifact = out / "scientific_program.semantic_pairs.json"
    if artifact.is_file() and not force:
        batch = ScientificProgramPairBatch.model_validate_json(
            artifact.read_text(encoding="utf-8")
        )
        validate_program_pair_batch(
            idea_ids=[row.idea_id for row in nodes],
            batch=batch,
        )
        return batch

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"No API key available. Set {api_key_env}.")
    try:
        import instructor
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("semantic program-family audit requires openai and instructor") from exc

    payload = program_family_prompt_payload(nodes)
    prompt_path = out / "scientific_program.prompt.txt"
    prompt_path.write_text(
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
        telemetry_path=out / "telemetry.jsonl",
        telemetry_context={
            "pipeline": "sis_v3_3_2_search_pressure_replay",
            "stage": "semantic_scientific_program_family",
        },
        semantic_components={
            "authority": "SHADOW_PROGRAM_FAMILY_ONLY",
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
    _write(artifact, batch)
    return batch


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replay SIS-v3.3.2 scientific-program family, search-pressure, and "
            "bounded parent scheduling over frozen v3.1/v3.3 artifacts. This is "
            "shadow-only and never executes child generation."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--generations", default="2,3,4")
    p.add_argument("--max-selected-parents", type=int, default=3)
    p.add_argument("--max-selected-per-same-program", type=int, default=1)
    p.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_AGENT_MODEL") or os.getenv("GRAPHAGENTS_HYPOTHESIS_MODEL") or "",
    )
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1",
    )
    p.add_argument("--parse-retries", type=int, default=2)
    p.add_argument("--force-program-family", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if not args.model:
        raise SystemExit("--model or OPENROUTER_AGENT_MODEL is required")
    if args.max_selected_parents < 1 or args.max_selected_per_same_program < 1:
        raise ValueError("parent budgets must be >= 1")

    run = args.run_dir.expanduser().resolve()
    e2e = run / "scientific_portfolio_shadow" / "sis_v3_2_e2e"
    arm = e2e / "arms" / "v3_1"
    sp = arm / "case_root" / "scientific_portfolio_shadow"
    v33 = run / "sis_v3_3_search_assessment_replay"
    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run / "sis_v3_3_2_search_pressure_replay"
    )
    out.mkdir(parents=True, exist_ok=True)

    generations = [int(x.strip()) for x in args.generations.split(",") if x.strip()]
    if not generations:
        raise ValueError("--generations must not be empty")

    nodes_by_generation = {
        generation: _nodes_for_generation(e2e=e2e, sp=sp, generation=generation)
        for generation in generations
    }
    unique_nodes: dict[str, ResearchIdeaNode] = {}
    for nodes in nodes_by_generation.values():
        for node in nodes:
            unique_nodes[node.idea_id] = node

    v331 = run / "sis_v3_3_1_search_pressure_replay"
    reusable_program_out = (
        v331
        if (not args.force_program_family and (v331 / "scientific_program.semantic_pairs.json").is_file())
        else out
    )
    pair_batch = _program_pairs(
        nodes=sorted(unique_nodes.values(), key=lambda row: row.idea_id),
        out=reusable_program_out,
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        parse_retries=args.parse_retries,
        force=args.force_program_family,
    )

    terminal_parallel_path = _must(
        sp / "sis_v3_1.g4_parallel_partial_search.json",
        "terminal G4 parallel partial search",
    )
    terminal_parallel = _load_json(terminal_parallel_path)
    external_path = _must(
        run / "full_current_verification" / "external_novelty.report.json",
        "terminal external novelty report",
    )
    external = ExternalNoveltyReport.model_validate_json(
        external_path.read_text(encoding="utf-8")
    )
    novelty_by_idea = build_novelty_depth_signals_by_idea(
        terminal_parallel_report=terminal_parallel,
        external_novelty_report=external,
    )
    _write(
        out / "terminal_novelty_depth_by_idea.json",
        {
            "schema_version": "sis-v3-3-2-terminal-novelty-depth-by-idea-v1",
            "source_external_novelty_report_id": external.report_id,
            "source_mode": "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
            "signals": [
                novelty_by_idea[key].model_dump(mode="json")
                for key in sorted(novelty_by_idea)
            ],
            "interpretation_guard": (
                "Retrospective terminal prior-art information is injected only to "
                "calibrate frozen search decisions. It was not available to v3.1."
            ),
        },
    )

    comparison_rows: list[dict[str, Any]] = []
    cycle_summaries: list[dict[str, Any]] = []

    for generation in generations:
        nodes = nodes_by_generation[generation]
        node_ids = {row.idea_id for row in nodes}
        sliced_pairs = ScientificProgramPairBatch(
            pairs=[
                row
                for row in pair_batch.pairs
                if row.idea_id_a in node_ids and row.idea_id_b in node_ids
            ]
        )
        family = build_scientific_program_family_report(
            nodes=nodes,
            pair_batch=sliced_pairs,
        )
        _write(out / f"g{generation}.scientific_program_family.json", family)

        assessment_path = _must(
            v33 / f"g{generation}.search_assessment.json",
            f"v3.3 G{generation} search assessment",
        )
        assessment = ResearchIdeaSearchAssessmentReport.model_validate_json(
            assessment_path.read_text(encoding="utf-8")
        )
        pressure = build_research_idea_search_pressure_report(
            search_assessment=assessment,
            program_family=family,
            novelty_by_idea=novelty_by_idea,
        )
        schedule = build_bounded_parent_schedule(
            pressure_report=pressure,
            program_family=family,
            max_selected_parents=args.max_selected_parents,
            max_selected_per_same_program=args.max_selected_per_same_program,
        )
        _write(out / f"g{generation}.search_pressure.json", pressure)
        _write(out / f"g{generation}.bounded_parent_schedule.json", schedule)

        old_fertility_path = sp / f"sis_v3_1.g{generation}_adaptive_fertility.json"
        old = {}
        if old_fertility_path.is_file():
            payload = _load_json(old_fertility_path)
            old = {
                str(row.get("idea_id")): str(row.get("disposition"))
                for row in payload.get("decisions", [])
                if isinstance(row, dict)
            }
        v33_by_id = {row.idea_id: row for row in assessment.decisions}
        selected_ids = {row.idea_id for row in schedule.selected_parents}

        for row in pressure.pressures:
            comparison_rows.append(
                {
                    "generation": generation,
                    "idea_id": row.idea_id,
                    "old_v3_1_disposition": old.get(row.idea_id),
                    "v3_3_search_action": (
                        v33_by_id[row.idea_id].search_action
                        if row.idea_id in v33_by_id
                        else None
                    ),
                    "v3_3_2_preferred_action": row.preferred_action,
                    "v3_3_2_search_priority": row.search_priority,
                    "selected_for_bounded_parent_schedule": row.idea_id in selected_ids,
                    "terminal_worthy": row.terminal_worthy,
                    "search_worthy": row.search_worthy,
                    "novelty_shape": row.novelty_shape,
                    "same_program_key": row.same_program_key,
                    "same_program_size": row.same_program_size,
                    "adjacent_program_neighbor_count": row.adjacent_program_neighbor_count,
                    "prior_art_escape_pressure": row.prior_art_escape_pressure,
                    "residual_refinement_pressure": row.residual_refinement_pressure,
                    "discrimination_upside": row.discrimination_upside,
                    "evidence_pressure": row.evidence_pressure,
                    "family_diversification_pressure": row.family_diversification_pressure,
                    "reason_codes": row.reason_codes,
                }
            )

        cycle_summaries.append(
            {
                "generation": generation,
                "program_count": family.program_count,
                "program_size_histogram": family.program_size_histogram,
                "program_relation_counts": family.relation_counts,
                "program_transitivity_diagnostic_count": family.transitivity_diagnostic_count,
                "old_v3_1_disposition_counts": dict(sorted(Counter(old.values()).items())),
                "v3_3_search_action_counts": dict(
                    sorted(Counter(row.search_action for row in assessment.decisions).items())
                ),
                "v3_3_2_preferred_action_counts": pressure.preferred_action_counts,
                "v3_3_2_search_priority_counts": pressure.search_priority_counts,
                "novelty_shape_counts": pressure.novelty_shape_counts,
                "terminal_worthy_count": pressure.terminal_worthy_count,
                "search_worthy_count": pressure.search_worthy_count,
                "bounded_selected_parent_count": schedule.selected_parent_count,
                "bounded_selected_parent_ids": [row.idea_id for row in schedule.selected_parents],
                "bounded_selected_program_count": schedule.selected_program_count,
                "bounded_selected_pair_relation_counts": schedule.selected_pair_relation_counts,
                "deferred_search_worthy_idea_ids": schedule.deferred_search_worthy_idea_ids,
            }
        )

    aggregate = {
        "row_count": len(comparison_rows),
        "unique_idea_count": len({row["idea_id"] for row in comparison_rows}),
        "v3_1_disposition_counts": dict(
            sorted(Counter(row["old_v3_1_disposition"] for row in comparison_rows).items())
        ),
        "v3_3_search_action_counts": dict(
            sorted(Counter(row["v3_3_search_action"] for row in comparison_rows).items())
        ),
        "v3_3_2_preferred_action_counts": dict(
            sorted(Counter(row["v3_3_2_preferred_action"] for row in comparison_rows).items())
        ),
        "v3_3_2_selected_parent_event_count": sum(
            bool(row["selected_for_bounded_parent_schedule"])
            for row in comparison_rows
        ),
        "terminal_worthy_event_count": sum(bool(row["terminal_worthy"]) for row in comparison_rows),
        "search_worthy_event_count": sum(bool(row["search_worthy"]) for row in comparison_rows),
    }
    comparison = {
        "schema_version": "sis-v3-3-2-search-pressure-replay-comparison-v1",
        "source_run": str(run),
        "source_v3_3_replay": str(v33),
        "generations": generations,
        "max_selected_parents": args.max_selected_parents,
        "max_selected_per_same_program": args.max_selected_per_same_program,
        "cycles": cycle_summaries,
        "comparison_rows": comparison_rows,
        "aggregate": aggregate,
        "interpretation_policy": {
            "retrospective_terminal_novelty_is_hindsight_only": True,
            "semantic_program_family_is_diagnostic_only": True,
            "terminal_worthy_does_not_mean_search_complete": True,
            "search_worthy_does_not_guarantee_parent_selection": True,
            "bounded_scheduler_executes_no_child_generation": True,
            "single_scalar_fitness_used": False,
            "production_generation_authority": False,
        },
    }
    _write(out / "comparison.json", comparison)

    print("=== SIS-v3.3.2 SEARCH PRESSURE REPLAY ===")
    print("run:", run)
    for row in cycle_summaries:
        print(
            f"G{row['generation']}:",
            "programs=", row["program_count"],
            "family_sizes=", row["program_size_histogram"],
            "preferred=", row["v3_3_2_preferred_action_counts"],
            "terminal_worthy=", row["terminal_worthy_count"],
            "search_worthy=", row["search_worthy_count"],
            "selected=", row["bounded_selected_parent_ids"],
        )
    print("aggregate:", json.dumps(aggregate, ensure_ascii=False))
    print("output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
