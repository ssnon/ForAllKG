from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.explorer_contracts import GraphExplorerPacket
from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.path_lineage_diagnostics import StatementPathLineageAssessor
from dac_her.path_lineage_propagation import MinimalPathLineagePropagator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply PL1-B minimal deterministic path-lineage propagation."
    )
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output-context", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    packet = GraphExplorerPacket.model_validate_json(
        args.packet.read_text(encoding="utf-8")
    )
    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )
    diagnostic = StatementPathLineageAssessor().assess(packet, context)
    updated, report = MinimalPathLineagePropagator().propagate(
        packet, context, diagnostic=diagnostic
    )

    args.output_context.parent.mkdir(parents=True, exist_ok=True)
    args.output_context.write_text(
        updated.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("PL1-B minimal path-lineage propagation")
    print("Context SHA:", context.context_sha256, "->", updated.context_sha256)
    print("Eligible statements:", report.eligible_statement_count)
    print("Propagated:", report.propagated_statement_count)
    print(
        "Explicit lineage before/after:",
        f"{report.pre_explicit_path_lineage_statement_count}/"
        f"{report.post_explicit_path_lineage_statement_count}",
    )
    print("Total propagated path IDs:", report.total_propagated_path_id_count)
    print("Scientific support changes:", report.scientific_support_changed_statement_count)
    print("Premise eligibility changes:", report.premise_eligibility_changed_statement_count)
    print("Saved context:", args.output_context)
    print("Saved report:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
