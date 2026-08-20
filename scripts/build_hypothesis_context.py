from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.evidence_compression import EvidenceCompressionAssessor
from pipeline_core.discovery.evidence_family_diagnostics import (
    EvidenceFamilyCandidateAssessor,
)
from pipeline_core.discovery.path_lineage_diagnostics import (
    StatementPathLineageAssessor,
)
from pipeline_core.discovery.path_lineage_propagation import (
    MinimalPathLineagePropagator,
)
from pipeline_core.discovery.explorer_contracts import ExplorationReport, GraphExplorerPacket
from dac_her.hypothesis_context import HypothesisContextBuilder


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic HypothesisContext v2.6.0")
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--compression-output",
        type=Path,
        default=None,
        help=(
            "Optional EC1 diagnostic-only Explorer-to-context evidence "
            "compression report."
        ),
    )
    parser.add_argument(
        "--family-diagnostics-output",
        type=Path,
        default=None,
        help=(
            "Optional EC2-A evidence-family decomposition-candidate "
            "diagnostic report. No statement is modified."
        ),
    )
    parser.add_argument(
        "--path-lineage-output",
        type=Path,
        default=None,
        help=(
            "Optional PL1-A post-propagation statement-to-selected-path "
            "diagnostic report."
        ),
    )
    parser.add_argument(
        "--path-lineage-propagation-output",
        type=Path,
        default=None,
        help="Optional PL1-B minimal deterministic propagation report.",
    )
    parser.add_argument(
        "--disable-path-lineage-propagation",
        action="store_true",
        help="Preserve legacy empty eligible support_path_ids.",
    )
    parser.add_argument(
        "--allow-invalid-source-report",
        action="store_true",
        help="Development-only escape hatch; production should require a validated ExplorationReport.",
    )
    args = parser.parse_args()

    packet = GraphExplorerPacket.model_validate(_load(args.packet))
    report = ExplorationReport.model_validate(_load(args.report))
    context = HypothesisContextBuilder().build(
        packet,
        report,
        require_valid_report=not args.allow_invalid_source_report,
    )

    propagation = None
    if not args.disable_path_lineage_propagation:
        pre_lineage = StatementPathLineageAssessor().assess(
            packet,
            context,
        )
        context, propagation = MinimalPathLineagePropagator().propagate(
            packet,
            context,
            diagnostic=pre_lineage,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        context.model_dump_json(indent=2),
        encoding="utf-8",
    )

    if args.path_lineage_propagation_output is not None:
        if propagation is None:
            raise ValueError(
                "--path-lineage-propagation-output cannot be used with "
                "--disable-path-lineage-propagation"
            )
        args.path_lineage_propagation_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.path_lineage_propagation_output.write_text(
            propagation.model_dump_json(indent=2),
            encoding="utf-8",
        )

    path_lineage = None
    if args.path_lineage_output is not None:
        path_lineage = StatementPathLineageAssessor().assess(
            packet,
            context,
        )
        args.path_lineage_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.path_lineage_output.write_text(
            path_lineage.model_dump_json(indent=2),
            encoding="utf-8",
        )

    compression = None
    family_diagnostics = None
    if (
        args.compression_output is not None
        or args.family_diagnostics_output is not None
    ):
        compression = EvidenceCompressionAssessor().assess(
            packet,
            report,
            context,
        )

    if args.compression_output is not None:
        args.compression_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.compression_output.write_text(
            compression.model_dump_json(indent=2),
            encoding="utf-8",
        )

    if args.family_diagnostics_output is not None:
        family_diagnostics = (
            EvidenceFamilyCandidateAssessor().assess(
                compression
            )
        )
        args.family_diagnostics_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.family_diagnostics_output.write_text(
            family_diagnostics.model_dump_json(indent=2),
            encoding="utf-8",
        )

    eligible = sum(x.eligible_as_premise for x in context.evidence_statements)
    gaps = sum(x.eligible_as_gap for x in context.evidence_statements)
    candidate = sum(x.requires_verification for x in context.evidence_statements)
    alignment_blocked = sum(bool(x.alignment_path_ids) for x in context.evidence_statements)
    print("HypothesisContext built")
    print("Context ID:", context.context_id)
    print("Context SHA256:", context.context_sha256)
    print("Source report:", context.source_report_id)
    print("Domain profile:", context.domain_profile_id)
    print("Evidence statements:", len(context.evidence_statements))
    print("Eligible positive premises:", eligible)
    print("Eligible research gaps:", gaps)
    print("Candidate-dependent statements:", candidate)
    print("Alignment-blocked statements:", alignment_blocked)
    print("Partial absence blocked papers:", len(context.partial_absence_blocked_paper_ids))
    if compression is not None:
        print(
            "EC1 eligible paper/statement counts:",
            compression.eligible_premise_paper_count,
            "/",
            compression.eligible_statement_count,
        )
        print(
            "EC1 papers only in multi-paper statements:",
            compression.eligible_papers_only_in_multi_paper_statements_count,
        )
        if args.compression_output is not None:
            print(
                "Saved evidence compression:",
                args.compression_output,
            )
    if family_diagnostics is not None:
        print(
            "EC2-A decomposition candidates:",
            family_diagnostics.decomposition_candidate_count,
            family_diagnostics.decomposition_candidate_statement_ids,
        )
        print(
            "EC2-A eligible statements without explicit path lineage:",
            family_diagnostics.eligible_statements_without_explicit_path_lineage_count,
        )
        print(
            "Saved evidence-family diagnostics:",
            args.family_diagnostics_output,
        )
    if propagation is not None:
        print(
            "PL1-B propagated statements:",
            propagation.propagated_statement_count,
            "/",
            propagation.eligible_statement_count,
        )
        print(
            "PL1-B explicit lineage before/after:",
            propagation.pre_explicit_path_lineage_statement_count,
            "/",
            propagation.post_explicit_path_lineage_statement_count,
        )
        print(
            "PL1-B propagated path IDs:",
            propagation.total_propagated_path_id_count,
        )
        print(
            "PL1-B support/eligibility changes:",
            propagation.scientific_support_changed_statement_count,
            "/",
            propagation.premise_eligibility_changed_statement_count,
        )
        if args.path_lineage_propagation_output is not None:
            print(
                "Saved path-lineage propagation:",
                args.path_lineage_propagation_output,
            )
    if path_lineage is not None:
        print(
            "PL1-A deterministic attribution:",
            path_lineage.eligible_with_deterministic_attribution_count,
            "/",
            path_lineage.eligible_statement_count,
        )
        print(
            "PL1-A mechanistic attribution:",
            path_lineage.eligible_with_deterministic_mechanistic_attribution_count,
            "/",
            path_lineage.eligible_statement_count,
        )
        print(
            "Saved statement-path lineage diagnostics:",
            args.path_lineage_output,
        )
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
