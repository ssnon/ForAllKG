from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.evidence_compression import EvidenceCompressionReport
from dac_her.evidence_family_diagnostics import (
    EvidenceFamilyCandidateAssessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute EC2-A evidence-family decomposition-candidate diagnostics "
            "from an EC1 evidence-compression report."
        )
    )
    parser.add_argument(
        "--compression",
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

    compression = EvidenceCompressionReport.model_validate_json(
        args.compression.read_text(encoding="utf-8")
    )
    report = EvidenceFamilyCandidateAssessor().assess(
        compression
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

    print("Evidence-family candidate diagnostic")
    print("Report:", report.report_id)
    print(
        "Eligible statements:",
        report.eligible_statement_count,
    )
    print(
        "Eligible multi-paper statements:",
        report.eligible_multi_paper_statement_count,
    )
    print(
        "Homogeneous/heterogeneous multi-paper:",
        f"{report.eligible_homogeneous_multi_paper_statement_count}/"
        f"{report.eligible_heterogeneous_multi_paper_statement_count}",
    )
    print(
        "Decomposition candidates:",
        report.decomposition_candidate_count,
        report.decomposition_candidate_statement_ids,
    )
    print(
        "Candidate family count:",
        report.candidate_family_count,
    )
    print(
        "Candidate papers:",
        report.candidate_paper_count,
        report.candidate_paper_ids,
    )
    print(
        "Eligible statements without explicit path lineage:",
        f"{report.eligible_statements_without_explicit_path_lineage_count}/"
        f"{report.eligible_statement_count}",
        f"({report.eligible_statements_without_explicit_path_lineage_fraction:.3f})",
    )
    print("Saved:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
