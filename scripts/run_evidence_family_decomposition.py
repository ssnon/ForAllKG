from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.evidence_compression import (
    EvidenceCompressionAssessor,
)
from dac_her.evidence_family_decomposition import (
    ConservativeEvidenceFamilyDecomposer,
)
from dac_her.evidence_family_diagnostics import (
    EvidenceFamilyCandidateAssessor,
)
from dac_her.explorer_contracts import (
    ExplorationReport,
    GraphExplorerPacket,
)
from dac_her.hypothesis_contracts import HypothesisContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EC2-B conservative additive evidence-family decomposition. "
            "The input context must already contain PL1-B path lineage."
        )
    )
    parser.add_argument(
        "--packet",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--explorer-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--decomposition-report",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--compression-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--family-diagnostics-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--path-propagation-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--statement-id",
        action="append",
        default=None,
        help=(
            "Optional EC2-A candidate statement ID to expand. "
            "May be repeated. Default: all candidates whose parent role is "
            "evidence_synthesis."
        ),
    )
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if args.context.resolve() == args.output_context.resolve():
        raise ValueError(
            "Refusing to overwrite the source context. "
            "Use a side-by-side output context for EC2-B validation."
        )

    packet = GraphExplorerPacket.model_validate_json(
        args.packet.read_text(encoding="utf-8")
    )
    explorer_report = ExplorationReport.model_validate_json(
        args.explorer_report.read_text(encoding="utf-8")
    )
    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )

    compression = EvidenceCompressionAssessor().assess(
        packet,
        explorer_report,
        context,
    )
    family_diagnostics = (
        EvidenceFamilyCandidateAssessor().assess(
            compression
        )
    )

    selected = (
        set(args.statement_id)
        if args.statement_id
        else None
    )

    output_context, decomposition, propagation = (
        ConservativeEvidenceFamilyDecomposer().decompose(
            packet,
            context,
            family_diagnostics,
            statement_ids=selected,
        )
    )

    _write_json(
        args.output_context,
        output_context,
    )
    _write_json(
        args.decomposition_report,
        decomposition,
    )

    if args.compression_output is not None:
        _write_json(
            args.compression_output,
            compression,
        )
    if args.family_diagnostics_output is not None:
        _write_json(
            args.family_diagnostics_output,
            family_diagnostics,
        )
    if args.path_propagation_output is not None:
        _write_json(
            args.path_propagation_output,
            propagation,
        )

    print("EC2-B conservative evidence-family decomposition")
    print(
        "Context SHA:",
        context.context_sha256,
        "->",
        output_context.context_sha256,
    )
    print(
        "Candidates:",
        decomposition.candidate_statement_count,
    )
    print(
        "Expanded parents:",
        decomposition.expanded_parent_count,
    )
    print(
        "Child statements:",
        decomposition.child_statement_count,
    )
    print(
        "Eligible statements before/after:",
        f"{decomposition.eligible_statement_count_before}/"
        f"{decomposition.eligible_statement_count_after}",
    )
    print(
        "Original statements changed/missing:",
        f"{decomposition.original_statement_changed_count}/"
        f"{decomposition.original_statement_missing_count}",
    )
    print(
        "Child path propagation:",
        f"{propagation.propagated_statement_count} propagated",
    )
    print("Saved context:", args.output_context)
    print(
        "Saved decomposition report:",
        args.decomposition_report,
    )

    for row in decomposition.child_lineages:
        print()
        print(
            row.parent_statement_id,
            "->",
            row.child_statement_id,
        )
        print("  papers:", row.paper_ids)
        print("  kind:", row.claim_kind)
        print(
            "  nodes:",
            row.scientific_support_node_ids,
        )
        print(
            "  edges:",
            row.scientific_support_edge_ids,
        )
        print(
            "  paths:",
            row.support_path_ids,
        )
        print(
            "  PL1-B cover size:",
            row.pl1b_minimum_cover_size,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
