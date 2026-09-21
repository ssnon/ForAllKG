from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.corpus.semantic_ir.task_gap import (
    detect_task_local_capability_gaps,
)
from pipeline_core.corpus.semantic_ir.task_scope import (
    resolve_graph_explorer_task_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a GraphExplorerPacket back to existing paper-local "
            "Semantic IR and inspect task-local structured-condition gaps."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--duplicate-paper-policy",
        choices=("error", "latest_attempt"),
        default="error",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolution, bundles = resolve_graph_explorer_task_scope(
        packet_path=args.packet,
        corpus_root=args.root,
        duplicate_paper_policy=args.duplicate_paper_policy,
    )
    gaps = detect_task_local_capability_gaps(
        bundles=bundles,
        task_source_chunks=resolution.source_chunks,
        capabilities=[
            "measurement_condition_coverage",
            "experiment_condition_coverage",
            "calculation_condition_coverage",
        ],
    )

    output_path = Path(args.output) if args.output else (
        Path(args.packet).resolve().parent / "semantic_task_scope.json"
    )
    payload = {
        "resolution": resolution.model_dump(mode="json"),
        "task_local_gap_report": gaps.model_dump(mode="json"),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Task:", resolution.task_id)
    print("Packet:", resolution.packet_path)
    print("Referenced corpus nodes:", resolution.referenced_corpus_node_count)
    print("Paper-local nodes:", resolution.paper_local_node_count)
    print("Referenced papers:", len(resolution.referenced_paper_ids))
    print("Available extraction papers:", len(resolution.available_paper_ids))
    print("Missing extraction papers:", len(resolution.missing_paper_ids))
    if resolution.missing_paper_ids:
        print("Missing paper IDs:", ", ".join(resolution.missing_paper_ids[:30]))
    print("Resolved paper-local nodes:", len(resolution.resolved_objects))
    print(
        "Resolved semantic mentions:",
        sum(row.selected_match_count for row in resolution.resolved_objects),
    )
    print(
        "Multi-mention paper-level nodes:",
        len(resolution.multi_mention_paper_local_node_ids),
    )
    print(
        "Provenance-narrowed multi-mention nodes:",
        len(resolution.provenance_narrowed_paper_local_node_ids),
    )
    print(
        "Collision-scoped nodes:",
        len(resolution.collision_scoped_paper_local_node_ids),
    )
    print("Unresolved paper-local nodes:", len(resolution.unresolved_paper_local_node_ids))
    print("Truly ambiguous paper-local nodes:", len(resolution.ambiguous_paper_local_node_ids))
    print("Task source chunks:", len(resolution.source_chunks))
    print("Task-local condition capabilities:")
    for name, metric in sorted(gaps.metrics.items()):
        coverage = (
            "n/a"
            if metric.coverage_fraction is None
            else f"{metric.coverage_fraction:.1%}"
        )
        print(
            f"  {name}: {metric.supported_record_count}/"
            f"{metric.applicable_record_count} supported, "
            f"coverage={coverage}, witness_chunks={metric.witness_chunk_count}"
        )
    print("Output:", output_path)


if __name__ == "__main__":
    main()
