from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.scientific_verifier_materialization_planner import (
    build_scientific_verifier_materialization_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan lineage-safe materialization of frozen scientific-verifier holdout cases. "
            "This is a zero-LLM, zero-retrieval filesystem inventory only."
        )
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    plan = build_scientific_verifier_materialization_plan(
        benchmark_root=args.benchmark_root,
        case_keys=list(args.case_ids),
        exclude_case_keys=list(args.exclude_case),
    )
    output = args.output or (
        args.benchmark_root
        / "scientific_verifier_holdout_materialization_plan.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Scientific verifier holdout materialization plan complete")
    print("LLM calls: 0")
    print("Retrieval performed: false")
    print("Filesystem inventory only: true")
    print("Cohort selected before artifact outcomes: true")
    print("Unseen status is caller-declared only: true")
    print("Contamination audit performed: false")
    print("Planner is iterative; rerun after materialization to bind new lineage: true")
    print("Requested cases:", len(plan.requested_case_keys))
    print("Selected cases:", len(plan.selected_case_keys))
    print("Excluded by caller:", len(plan.excluded_case_keys))
    print("Tasks:", plan.task_count)
    print("Status counts:", plan.status_counts)

    for task in plan.tasks:
        print()
        print(f"{task.task_key}: {task.status}")
        if task.status == "EXCLUDED_BY_CALLER":
            continue
        print("  minimum planned LLM calls:", task.planned_minimum_llm_calls)
        print("  planned retrieval runs:", task.planned_retrieval_runs)
        if task.runtime_dependent_llm_stage_ids:
            print(
                "  runtime-dependent LLM stages:",
                ", ".join(task.runtime_dependent_llm_stage_ids),
            )
        for stage in task.stages:
            print(f"  {stage.stage_id}: {stage.state}")
            if stage.next_action:
                print(f"    next={stage.next_action}")
            if stage.missing_prerequisites:
                print(
                    "    missing=" + ",".join(stage.missing_prerequisites)
                )
            if stage.ignored_stale_or_mismatched_artifacts:
                print(
                    "    ignored_stale_or_mismatched="
                    + str(len(stage.ignored_stale_or_mismatched_artifacts))
                )

    print()
    print("Scientific quality judgment performed: false")
    print("Validation outcome assessed: false")
    print("Production selection changed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
