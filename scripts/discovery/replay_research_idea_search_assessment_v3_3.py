from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import ExternalNoveltyReport
from pipeline_core.discovery.research_idea_contracts import ResearchIdeaNode
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
)
from pipeline_core.discovery.research_idea_search_assessment import (
    build_research_idea_search_assessment,
    build_terminal_novelty_signals_by_idea,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _must(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _nodes_for_generation(*, arm: Path, sp: Path, generation: int) -> list[ResearchIdeaNode]:
    if generation == 2:
        payload = _load_json(_must(arm.parent.parent / "p0.population.json", "P0 population"))
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


def _old_fertility_by_idea(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = _load_json(path)
    return {
        str(row.get("idea_id")): str(row.get("disposition"))
        for row in payload.get("decisions", [])
        if isinstance(row, dict)
    }


def _comparison_rows(report: Any, old: dict[str, str]) -> list[dict[str, Any]]:
    assessment_by_id = {row.idea_id: row for row in report.assessments}
    rows = []
    for decision in report.decisions:
        a = assessment_by_id[decision.idea_id]
        old_disposition = old.get(decision.idea_id)
        rows.append(
            {
                "idea_id": decision.idea_id,
                "idea_birth_generation_index": decision.idea_birth_generation_index,
                "cycle_generation_index": decision.cycle_generation_index,
                "old_v3_1_disposition": old_disposition,
                "new_persistence": decision.persistence,
                "new_search_action": decision.search_action,
                "search_complete_candidate": decision.search_complete_candidate,
                "usable_grounded_realization_count": a.usable_grounded_realization_count,
                "novelty_state": a.novelty.novelty_state,
                "external_status_counts": a.novelty.external_status_counts,
                "program_family_size": a.program_family_size,
                "family_pressure": a.family_pressure,
                "discrimination_sketch_complete": a.discrimination_sketch_complete,
                "epistemic_debt_kinds": a.epistemic_debt_kinds,
                "reason_codes": decision.reason_codes,
            }
        )
    return rows


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replay SIS-v3.3 ResearchIdea search assessment over an existing "
            "FULL_CURRENT_AI_SCIENTIST_V1 case without changing v3.1 fertility or "
            "executing child generation. Terminal external novelty is used only as "
            "explicit retrospective hindsight for the frozen replay."
        )
    )
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--program-similarity-floor", type=float, default=0.58)
    p.add_argument("--dense-family-size", type=int, default=3)
    p.add_argument(
        "--generations",
        default="2,3,4",
        help="Comma-separated existing cycle generations to replay (default: 2,3,4).",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    run = args.run_dir.expanduser().resolve()
    arm = run / "scientific_portfolio_shadow" / "sis_v3_2_e2e" / "arms" / "v3_1"
    sp = arm / "case_root" / "scientific_portfolio_shadow"
    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run / "sis_v3_3_search_assessment_replay"
    )
    out.mkdir(parents=True, exist_ok=True)

    external_path = _must(
        run / "full_current_verification" / "external_novelty.report.json",
        "terminal external novelty report",
    )
    external = ExternalNoveltyReport.model_validate_json(
        external_path.read_text(encoding="utf-8")
    )

    terminal_parallel_path = _must(
        sp / "sis_v3_1.g4_parallel_partial_search.json",
        "terminal G4 parallel partial-search report",
    )
    terminal_parallel = _load_json(terminal_parallel_path)
    novelty_by_idea = build_terminal_novelty_signals_by_idea(
        terminal_parallel_report=terminal_parallel,
        external_novelty_report=external,
    )
    _write(
        out / "terminal_novelty_by_idea.json",
        {
            "schema_version": "sis-v3-3-terminal-novelty-by-idea-replay-v1",
            "source_external_novelty_report": str(external_path),
            "source_terminal_parallel_report": str(terminal_parallel_path),
            "source_mode": "RETROSPECTIVE_TERMINAL_EXTERNAL_NOVELTY",
            "warning": (
                "Hindsight-only replay input. These novelty results were not available "
                "to the original v3.1 fertility policy and carry no generation authority."
            ),
            "ideas": {
                idea_id: signal.model_dump(mode="json")
                for idea_id, signal in sorted(novelty_by_idea.items())
            },
        },
    )

    generations = [
        int(value.strip())
        for value in str(args.generations).split(",")
        if value.strip()
    ]
    if not generations or any(value < 2 for value in generations):
        raise ValueError("--generations must contain integers >= 2")

    cycle_summaries = []
    all_rows: list[dict[str, Any]] = []
    for generation in generations:
        lifecycle_path = sp / f"sis_v3_1.g{generation}_realization_lifecycle.json"
        parallel_path = sp / f"sis_v3_1.g{generation}_parallel_partial_search.json"
        fertility_path = sp / f"sis_v3_1.g{generation}_adaptive_fertility.json"
        if not lifecycle_path.is_file() or not parallel_path.is_file():
            cycle_summaries.append(
                {
                    "generation": generation,
                    "status": "SKIPPED_MISSING_CYCLE_ARTIFACTS",
                    "lifecycle": str(lifecycle_path),
                    "parallel": str(parallel_path),
                }
            )
            continue

        lifecycle = _load_json(lifecycle_path)
        parallel = _load_json(parallel_path)
        nodes = _nodes_for_generation(arm=arm, sp=sp, generation=generation)
        report = build_research_idea_search_assessment(
            parallel_report=parallel,
            lifecycle=lifecycle,
            population_nodes=nodes,
            novelty_signals_by_idea=novelty_by_idea,
            program_similarity_floor=args.program_similarity_floor,
            dense_family_size=args.dense_family_size,
        )
        report_path = out / f"g{generation}.search_assessment.json"
        _write(report_path, report)

        old = _old_fertility_by_idea(fertility_path)
        rows = _comparison_rows(report, old)
        all_rows.extend(rows)
        old_counts = Counter(old.values())
        new_counts = Counter(row["new_search_action"] for row in rows)
        old_hold_ids = {
            row["idea_id"]
            for row in rows
            if row["old_v3_1_disposition"] == "HOLD_STABLE"
        }
        hold_reopened = [
            row["idea_id"]
            for row in rows
            if row["idea_id"] in old_hold_ids
            and row["new_search_action"] != "NONE"
        ]
        cycle_summaries.append(
            {
                "generation": generation,
                "status": "COMPLETE",
                "report": str(report_path),
                "idea_count": len(rows),
                "old_v3_1_disposition_counts": dict(sorted(old_counts.items())),
                "new_search_action_counts": dict(sorted(new_counts.items())),
                "old_hold_count": len(old_hold_ids),
                "old_hold_reopened_by_v3_3_count": len(hold_reopened),
                "old_hold_reopened_by_v3_3_idea_ids": hold_reopened,
            }
        )

    overall = {
        "schema_version": "sis-v3-3-search-assessment-replay-summary-v1",
        "source_run": str(run),
        "mode": "RETROSPECTIVE_HINDSIGHT_SHADOW",
        "program_similarity_floor": args.program_similarity_floor,
        "dense_family_size": args.dense_family_size,
        "cycles": cycle_summaries,
        "comparison_rows": all_rows,
        "aggregate": {
            "row_count": len(all_rows),
            "old_v3_1_disposition_counts": dict(
                sorted(Counter(row.get("old_v3_1_disposition") for row in all_rows).items(), key=lambda x: str(x[0]))
            ),
            "new_search_action_counts": dict(
                sorted(Counter(row["new_search_action"] for row in all_rows).items())
            ),
            "novelty_state_counts": dict(
                sorted(Counter(row["novelty_state"] for row in all_rows).items())
            ),
            "old_hold_reopened_count": sum(
                row.get("old_v3_1_disposition") == "HOLD_STABLE"
                and row["new_search_action"] != "NONE"
                for row in all_rows
            ),
            "search_complete_candidate_count": sum(
                bool(row["search_complete_candidate"]) for row in all_rows
            ),
        },
        "interpretation_guard": (
            "This replay does not prove that contemporaneous v3.3 search would make the "
            "same decisions. It intentionally injects terminal prior-art results as "
            "hindsight to test whether the old grounded=>HOLD behavior masked scientific "
            "search upside. No child generation, KG mutation, truth authority, or production "
            "selection authority is created."
        ),
    }
    _write(out / "comparison.json", overall)

    print("=== SIS-v3.3 SEARCH ASSESSMENT REPLAY ===")
    print("run:", run)
    for cycle in cycle_summaries:
        print(
            f"G{cycle['generation']}:",
            cycle["status"],
            "old=", cycle.get("old_v3_1_disposition_counts"),
            "new=", cycle.get("new_search_action_counts"),
            "hold_reopened=", cycle.get("old_hold_reopened_by_v3_3_count"),
        )
    print("aggregate:", json.dumps(overall["aggregate"], ensure_ascii=False))
    print("output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
