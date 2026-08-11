from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.explorer_contracts import GraphExplorerPacket
from dac_her.hypothesis_contracts import HypothesisContext
from dac_her.path_lineage_diagnostics import (
    StatementPathLineageAssessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PL1-A diagnostic-only attribution between "
            "HypothesisContext statements and selected GraphExplorer paths."
        )
    )
    parser.add_argument(
        "--packet",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    packet = GraphExplorerPacket.model_validate_json(
        args.packet.read_text(
            encoding="utf-8"
        )
    )
    context = HypothesisContext.model_validate_json(
        args.context.read_text(
            encoding="utf-8"
        )
    )

    report = StatementPathLineageAssessor().assess(
        packet,
        context,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Statement-path lineage diagnostic")
    print("Report:", report.report_id)
    print(
        "Selected paths / mechanistic:",
        f"{report.selected_path_count}/"
        f"{report.selected_mechanistic_path_count}",
    )
    print(
        "Eligible statements:",
        report.eligible_statement_count,
    )
    print(
        "Explicit path lineage:",
        f"{report.eligible_with_explicit_path_lineage_count}/"
        f"{report.eligible_statement_count}",
    )
    print(
        "Deterministically attributable:",
        f"{report.eligible_with_deterministic_attribution_count}/"
        f"{report.eligible_statement_count}",
        f"({report.eligible_statement_deterministic_attribution_fraction:.3f})",
    )
    print(
        "Mechanistically attributable:",
        f"{report.eligible_with_deterministic_mechanistic_attribution_count}/"
        f"{report.eligible_statement_count}",
        f"({report.eligible_statement_mechanistic_attribution_fraction:.3f})",
    )
    print(
        "Recoverable missing explicit lineage:",
        report.recoverable_missing_explicit_path_lineage_count,
    )
    print(
        "Unrecoverable missing explicit lineage:",
        report.unrecoverable_missing_explicit_path_lineage_count,
    )
    print(
        "Candidate-union full edge/node coverage:",
        f"{report.eligible_candidate_union_full_edge_coverage_count}/"
        f"{report.eligible_candidate_union_full_node_coverage_count}",
    )
    print(
        "Selected paths attributable to eligible statements:",
        f"{report.selected_paths_attributable_to_eligible_statement_count}/"
        f"{report.selected_path_count}",
    )
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
