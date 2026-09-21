from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.corpus.semantic_ir.grounded_scope import (
    resolve_grounded_semantic_task_scope,
)
from pipeline_core.corpus.semantic_ir.task_gap import (
    detect_task_local_capability_gaps,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the grounded HypothesisContext evidence surface back to "
            "paper-local source chunks across one or more extraction roots."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Extraction root. Repeat for merged/legacy corpus roots.",
    )
    parser.add_argument("--packet", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument(
        "--scope-mode",
        choices=("premise_and_gap", "premise_only"),
        default="premise_and_gap",
    )
    parser.add_argument(
        "--duplicate-paper-policy",
        choices=("error", "latest_attempt"),
        default="error",
    )
    parser.add_argument(
        "--cross-root-duplicate-policy",
        choices=("error", "prefer_last_root"),
        default="error",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result, bundles = resolve_grounded_semantic_task_scope(
        packet_path=args.packet,
        context_path=args.context,
        corpus_roots=args.root,
        scope_mode=args.scope_mode,
        duplicate_paper_policy=args.duplicate_paper_policy,
        cross_root_duplicate_policy=args.cross_root_duplicate_policy,
    )

    gaps = detect_task_local_capability_gaps(
        bundles=bundles,
        task_source_chunks=result.resolution.source_chunks,
        capabilities=[
            "calculation_condition_coverage",
            "experiment_condition_coverage",
            "measurement_condition_coverage",
        ],
    )

    output = (
        Path(args.output)
        if args.output
        else Path(args.context).parent / "semantic_grounded_scope.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    payload["task_local_condition_gaps"] = gaps.model_dump(mode="json")
    output.write_text(
        __import__("json").dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("Task:", result.selection.task_id)
    print("Scope mode:", result.selection.scope_mode)
    print("Extraction roots:", len(result.roots.roots))
    print("Selected statements:", len(result.selection.selected_statement_ids))
    print(
        "Selected positive premises:",
        len(result.selection.selected_premise_statement_ids),
    )
    print("Selected research gaps:", len(result.selection.selected_gap_statement_ids))
    print("Broad packet catalog nodes:", result.broad_packet_node_count)
    print("Grounded support nodes:", result.grounded_selected_node_count)
    print("Grounded support edges:", result.grounded_selected_edge_count)
    print("Referenced grounded papers:", len(result.roots.requested_paper_ids))
    print("Available grounded papers:", len(result.roots.available_paper_ids))
    print("Missing grounded papers:", len(result.roots.missing_paper_ids))
    if result.roots.missing_paper_ids:
        print(
            "Missing grounded paper IDs:",
            ", ".join(result.roots.missing_paper_ids[:30]),
        )
    print("Resolved grounded paper-local nodes:", len(result.resolution.resolved_objects))
    print(
        "Resolved grounded semantic mentions:",
        sum(row.selected_match_count for row in result.resolution.resolved_objects),
    )
    print(
        "Grounded multi-mention nodes:",
        len(result.resolution.multi_mention_paper_local_node_ids),
    )
    print(
        "Grounded provenance-narrowed nodes:",
        len(result.resolution.provenance_narrowed_paper_local_node_ids),
    )
    print(
        "Grounded unresolved nodes:",
        len(result.resolution.unresolved_paper_local_node_ids),
    )
    print(
        "Grounded truly ambiguous nodes:",
        len(result.resolution.ambiguous_paper_local_node_ids),
    )
    print("Grounded source chunks:", result.source_chunk_count)
    print("Task-local condition capabilities:")
    for name, metric in sorted(gaps.metrics.items()):
        coverage = (
            "unknown"
            if metric.coverage_fraction is None
            else f"{metric.coverage_fraction:.1%}"
        )
        print(
            f"  {name}: {metric.supported_record_count}/"
            f"{metric.applicable_record_count} supported, "
            f"coverage={coverage}, witness_chunks={metric.witness_chunk_count}"
        )
    print("Output:", output)


if __name__ == "__main__":
    main()
