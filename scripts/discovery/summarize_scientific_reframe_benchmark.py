from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.benchmark_synthesis import (
    build_benchmark_reframe_synthesis,
)
from pipeline_core.discovery.reframing.selective_execution import (
    ScientificReframeSelectiveExecutionReport,
    load_json_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate trigger-selected scientific reframing candidates, critic vectors, "
            "and shadow portfolio routing across a benchmark. This is deterministic and "
            "does not infer semantic equivalence or candidate ranking."
        )
    )
    parser.add_argument("--execution", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--lexical-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--allow-lineage-errors", action="store_true")
    args = parser.parse_args()

    execution = load_json_model(
        args.execution.resolve(),
        ScientificReframeSelectiveExecutionReport,
    )
    synthesis = build_benchmark_reframe_synthesis(
        execution=execution,
        lexical_overlap_threshold=args.lexical_overlap_threshold,
        fail_on_lineage_error=not args.allow_lineage_errors,
    )
    output = args.output or Path(execution.benchmark_root) / "scientific_reframing_benchmark_synthesis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(synthesis.model_dump_json(indent=2), encoding="utf-8")

    print("Scientific reframing benchmark synthesis complete")
    print("LLM calls: 0")
    print("Triggered tasks:", synthesis.triggered_task_count)
    print("Completed tasks:", synthesis.completed_task_count)
    print("Candidates:", synthesis.candidate_count)
    print("Slots:")
    for slot, count in sorted(synthesis.slot_counts.items()):
        print(f"  {slot}: {count}")
    print("Operators:")
    for operator in synthesis.operator_summaries:
        print(
            f"  {operator.operator_id}: candidates={operator.candidate_count}; "
            f"tasks={operator.task_count}; lexical_pairs={operator.lexical_overlap_pair_count}; "
            f"above_threshold={operator.lexical_overlap_threshold_exceeded_count}"
        )
        if operator.repeated_construct_tokens:
            repeated = ", ".join(
                f"{token}({count})"
                for token, count in operator.repeated_construct_tokens.items()
            )
            print("    repeated construct tokens:", repeated)
        if operator.exact_construct_signature_duplicate_groups:
            print(
                "    exact construct-signature duplicate groups:",
                len(operator.exact_construct_signature_duplicate_groups),
            )
    print("Lexical overlap is diagnostic only; semantic collapse was not evaluated.")
    print("No overall score, ranking, novelty, N10, or production selection was performed.")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
