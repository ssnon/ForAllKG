from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_core.discovery.reframing.sparse_replication_materializer import (
    build_sparse_replication_plan,
    execute_sparse_replication_plan,
    execution_report_to_dict,
    plan_to_dict,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the zero-LLM downstream replication pipeline from existing sparse "
            "scientific-reframing shadow candidates. Missing optional reasoning modes are "
            "preserved as absent rather than generated merely for wiring completeness."
        )
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--max-comparisons", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    plans = []
    for case_id in args.case_ids:
        canonical = benchmark_root / case_id / "replicate_01" / "canonical"
        plans.append(
            build_sparse_replication_plan(
                canonical_dir=canonical,
                case_key=case_id,
                max_comparisons=args.max_comparisons,
            )
        )

    print("Sparse-operator replication materialization plan")
    print("LLM calls planned: 0")
    print(f"Tasks: {len(plans)}")
    for plan in plans:
        print(
            f"  {plan.case_key}: reframe={plan.reframe_shadow.name}; "
            f"proxy={'present' if plan.proxy_shadow else 'absent'}; "
            f"contradiction={'present' if plan.contradiction_shadow else 'absent'}"
        )
        for stage in plan.stages:
            existing = all(path.is_file() for path in stage.outputs)
            print(
                f"    {stage.stage_id}: "
                f"{'existing' if existing else 'pending'}"
            )

    output = args.report or (
        benchmark_root / "scientific_reasoning_sparse_replication_materialization.json"
    )
    if not args.execute:
        write_json(
            output,
            {
                "schema_version": "scientific-reasoning-sparse-replication-plan-v1",
                "executed": False,
                "llm_calls_performed": 0,
                "plans": [plan_to_dict(plan) for plan in plans],
            },
        )
        print("Execution performed: false")
        print("Run again with --execute after reviewing the plan.")
        print(f"Plan: {output}")
        return 0

    reports = []
    for plan in plans:
        print(f"\n== {plan.case_key} ==")
        reports.append(
            execute_sparse_replication_plan(
                plan=plan,
                repository_root=args.repository_root,
                force=args.force,
                python_executable=sys.executable,
            )
        )
    write_json(
        output,
        {
            "schema_version": "scientific-reasoning-sparse-replication-materialization-v1",
            "executed": True,
            "llm_calls_performed": 0,
            "scientific_quality_judgment_performed": False,
            "production_selection_changed": False,
            "reports": [execution_report_to_dict(row) for row in reports],
        },
    )
    print("\nSparse-operator replication materialization complete")
    print("LLM calls: 0")
    print("Scientific quality judgment performed: false")
    print("Production selection changed: false")
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
