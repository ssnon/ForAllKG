from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.multitask_replication_preflight import (
    build_multitask_replication_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory artifact readiness for multi-task scientific-reasoning replication."
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build_multitask_replication_preflight(
        benchmark_root=args.benchmark_root,
        case_keys=args.case_ids,
    )
    output = args.output or (
        args.benchmark_root / "scientific_reasoning_multitask_replication_preflight.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Scientific reasoning multi-task replication preflight complete")
    print("LLM calls: 0")
    print(f"Benchmark root: {report.benchmark_root}")
    print(f"Tasks: {report.task_count}")
    print(f"Status counts: {report.status_counts}")
    for task in report.tasks:
        print(f"  {task.task_key}: {task.status}")
        if task.candidate_counts:
            print(f"    candidate_counts={task.candidate_counts}")
        for action in task.next_actions:
            print(f"    next={action}")
        for note in task.diagnostic_notes:
            print(f"    note={note}")
    print("Scientific task ranking performed: false")
    print("Replication success established: false")
    print("Production selection changed: false")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
