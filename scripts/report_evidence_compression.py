from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.evidence_compression import (
    EvidenceCompressionAssessor,
)
from dac_her.explorer_contracts import (
    ExplorationReport,
    GraphExplorerPacket,
)
from dac_her.hypothesis_contracts import HypothesisContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute EC1 diagnostic-only Explorer-to-HypothesisContext "
            "evidence-compression metrics for an existing run."
        )
    )
    parser.add_argument(
        "--packet",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--report",
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
        args.packet.read_text(encoding="utf-8")
    )
    report = ExplorationReport.model_validate_json(
        args.report.read_text(encoding="utf-8")
    )
    context = HypothesisContext.model_validate_json(
        args.context.read_text(encoding="utf-8")
    )

    compression = EvidenceCompressionAssessor().assess(
        packet,
        report,
        context,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            compression.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Evidence compression diagnostic")
    print("Report:", compression.report_id)
    print(
        "Paper funnel:",
        f"{compression.selected_path_paper_count} selected-path ->",
        f"{compression.explorer_statement_paper_count} Explorer ->",
        f"{compression.eligible_premise_paper_count} eligible-premise",
    )
    print(
        "Eligible statements:",
        compression.eligible_statement_count,
    )
    print(
        "Single/multi-paper eligible statements:",
        f"{compression.eligible_single_paper_statement_count}/"
        f"{compression.eligible_multi_paper_statement_count}",
    )
    print(
        "Mean papers per eligible statement:",
        f"{compression.mean_papers_per_eligible_statement:.3f}",
    )
    print(
        "Eligible papers only represented in multi-paper statements:",
        f"{compression.eligible_papers_only_in_multi_paper_statements_count}/"
        f"{compression.eligible_premise_paper_count}",
        f"({compression.eligible_papers_only_in_multi_paper_statements_fraction:.3f})",
    )
    print(
        "Multi-paper statements with heterogeneous structural support profiles:",
        compression.eligible_multi_paper_heterogeneous_profile_count,
    )
    print(
        "Declared/direct-support mismatch statements:",
        compression.statements_with_declared_support_mismatch_count,
    )
    print(
        "Repeated paper-incidence groups:",
        len(compression.repeated_paper_incidence_groups),
    )
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
