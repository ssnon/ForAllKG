from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.evidence_compression import EvidenceCompressionAssessor
from dac_her.evidence_constituent_resolution import (
    ExistingFirstConstituentResolver,
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
            "EC2-D existing-first constituent resolution with conditional "
            "family-child materialization."
        )
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--explorer-report", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output-context", required=True, type=Path)
    parser.add_argument("--resolution-report", required=True, type=Path)
    parser.add_argument("--compression-output", type=Path, default=None)
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
            "Optional EC2-A candidate parent statement ID. May be repeated. "
            "Default: all eligible EC2-A decomposition candidates."
        ),
    )
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if args.context.resolve() == args.output_context.resolve():
        raise ValueError(
            "Refusing to overwrite the source context. "
            "Use a side-by-side EC2-D output context."
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
    family_diagnostics = EvidenceFamilyCandidateAssessor().assess(
        compression
    )

    selected = set(args.statement_id) if args.statement_id else None

    output_context, report, propagation = (
        ExistingFirstConstituentResolver().resolve(
            packet,
            context,
            family_diagnostics,
            statement_ids=selected,
        )
    )

    _write_json(args.output_context, output_context)
    _write_json(args.resolution_report, report)

    if args.compression_output is not None:
        _write_json(args.compression_output, compression)
    if args.family_diagnostics_output is not None:
        _write_json(
            args.family_diagnostics_output,
            family_diagnostics,
        )
    if args.path_propagation_output is not None:
        if propagation is not None:
            _write_json(args.path_propagation_output, propagation)
        else:
            _write_json(
                args.path_propagation_output,
                {
                    "schema_version": "ec2d-path-propagation-status-v1",
                    "status": "not_required",
                    "reason": "no_materialized_family_children",
                    "source_context_id": context.context_id,
                    "source_context_sha256": context.context_sha256,
                },
            )

    print("EC2-D existing-first constituent resolution")
    print(
        "Context SHA:",
        context.context_sha256,
        "->",
        output_context.context_sha256,
    )
    print("Candidate parents:", report.candidate_parent_count)
    print("Candidate families:", report.candidate_family_count)
    print(
        "Resolved to existing:",
        report.resolved_existing_family_count,
    )
    print(
        "Materialized children:",
        report.materialized_family_count,
    )
    print(
        "Eligible statements before/after:",
        f"{report.eligible_statement_count_before}/"
        f"{report.eligible_statement_count_after}",
    )
    print(
        "Original statements changed/missing:",
        f"{report.original_statement_changed_count}/"
        f"{report.original_statement_missing_count}",
    )
    print("Context SHA unchanged:", report.context_sha_unchanged)
    print("Saved context:", args.output_context)
    print("Saved report:", args.resolution_report)

    for row in report.family_resolutions:
        print()
        print(
            row.parent_statement_id,
            "/",
            row.family_id,
        )
        print("  family kind:", row.family_claim_kind)
        print("  family papers:", row.family_paper_ids)
        print("  status:", row.resolution_status)
        print("  basis:", row.resolution_basis)
        print(
            "  constituent:",
            row.resulting_constituent_statement_id,
        )
        if row.candidates:
            print("  candidates:")
            for candidate in row.candidates:
                print(
                    "   -",
                    candidate.statement_id,
                    "exact=",
                    candidate.exact_support_match,
                    "extra_edges=",
                    candidate.extra_edge_count,
                    "extra_nodes=",
                    candidate.extra_node_count,
                    "extra_papers=",
                    candidate.extra_paper_count,
                )
        print("  paths:", row.support_path_ids_after)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
