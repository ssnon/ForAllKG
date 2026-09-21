from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.corpus.semantic_ir.grounded_scope import (
    resolve_grounded_semantic_task_scope,
)
from pipeline_core.corpus.semantic_ir.task_capability import (
    build_task_capability_snapshot,
)
from pipeline_core.corpus.semantic_ir.task_gap import (
    detect_task_local_capability_gaps,
)
from pipeline_core.discovery.reframing.operator_readiness import (
    assess_reframing_operator_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assess shadow-only scientific reframing operator readiness from "
            "grounded task evidence. No LLM or enrichment call is performed."
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

    grounded, bundles = resolve_grounded_semantic_task_scope(
        packet_path=args.packet,
        context_path=args.context,
        corpus_roots=args.root,
        scope_mode=args.scope_mode,
        duplicate_paper_policy=args.duplicate_paper_policy,
        cross_root_duplicate_policy=args.cross_root_duplicate_policy,
    )
    gaps = detect_task_local_capability_gaps(
        bundles=bundles,
        task_source_chunks=grounded.resolution.source_chunks,
        capabilities=[
            "calculation_condition_coverage",
            "experiment_condition_coverage",
            "measurement_condition_coverage",
        ],
    )
    snapshot = build_task_capability_snapshot(
        scope_id=grounded.selection.task_id,
        bundles=bundles,
        task_source_chunks=grounded.resolution.source_chunks,
        task_gap_report=gaps,
    )
    report = assess_reframing_operator_readiness(
        scope_id=grounded.selection.task_id,
        task_capabilities=snapshot,
        gap_report=gaps,
        bundles=bundles,
        task_source_chunks=grounded.resolution.source_chunks,
        unresolved_grounded_node_count=len(
            grounded.resolution.unresolved_paper_local_node_ids
        ),
        ambiguous_grounded_node_count=len(
            grounded.resolution.ambiguous_paper_local_node_ids
        ),
    )

    output = (
        Path(args.output)
        if args.output
        else Path(args.context).parent / "reframing_operator_readiness.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Task:", report.scope_id)
    print("Grounded source chunks:", snapshot.source_chunk_count)
    print("Grounded semantic records:", snapshot.semantic_record_count)
    print("Unresolved grounded nodes:", report.unresolved_grounded_node_count)
    print("Ambiguous grounded nodes:", report.ambiguous_grounded_node_count)
    print("Operator readiness:")
    for row in report.assessments:
        print(
            f"  {row.operator_id}: {row.status}; "
            f"mandatory_backfill={row.mandatory_backfill_plan.target_count}; "
            f"optional_backfill={row.optional_backfill_plan.target_count}; "
            f"measurement_chunks={row.measurement_bearing_chunk_count}"
        )
        for condition in row.condition_families:
            print(
                f"    {condition.capability}: {condition.state} "
                f"({condition.supported_count}/{condition.applicable_count}, "
                f"missing={condition.missing_count}, "
                f"witness_chunks={condition.witness_chunk_count})"
            )
        for reason in row.reasons:
            print("    reason:", reason)
    print("Output:", output)


if __name__ == "__main__":
    main()
