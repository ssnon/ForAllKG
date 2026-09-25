from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.pre_n10_initial_semantic_gate_v1 import (
    execute_pre_n10_initial_semantic_gate_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile initial semantic authority from existing semantic-critic "
            "artifacts before any pre-N10 contract work. No semantic LLM call "
            "is performed by this stage."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--semantic-run", required=True, type=Path)
    parser.add_argument("--semantic-review", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    report, disposition = execute_pre_n10_initial_semantic_gate_v1(
        portfolio_path=args.portfolio,
        semantic_run_path=args.semantic_run,
        semantic_review_path=args.semantic_review,
        output_root=args.output_dir,
    )

    print("Initial pre-N10 semantic authority gate complete")
    print("Report:", report.report_id)
    print("Status:", report.status)
    print("Review valid:", report.semantic_review_valid)
    print("Pre-N10 entry authorized:", report.pre_n10_entry_authorized)
    if disposition is not None:
        print("Disposition:", disposition.disposition)
        print("Failed dimensions:", disposition.failed_dimensions)
        print("Warning dimensions:", disposition.warning_dimensions)
    print("Semantic runtime reexecuted: false")
    print("External novelty/N9/N10 performed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
