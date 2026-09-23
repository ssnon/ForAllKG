from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.scientific_verifier_unseen_validation import (
    build_unseen_validation_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory lineage-safe holdout inputs for the scientific verifier "
            "shadow chain. This performs no retrieval, LLM call, certification, "
            "or scientific-quality judgment."
        )
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument(
        "--exclude-case",
        action="append",
        default=[],
        help=(
            "Caller-declared calibration/development case to exclude. "
            "Repeat as needed."
        ),
    )
    parser.add_argument(
        "--include-case",
        action="append",
        default=[],
        help="Optional case allow-list. Repeat as needed.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = build_unseen_validation_preflight(
        benchmark_root=args.benchmark_root,
        excluded_case_keys=set(args.exclude_case),
        included_case_keys=(set(args.include_case) if args.include_case else None),
    )
    output = args.output or (
        args.benchmark_root.resolve()
        / "scientific_verifier_unseen_validation_preflight.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Scientific verifier unseen-validation preflight complete")
    print("LLM calls: 0")
    print("Retrieval performed: false")
    print("Benchmark root:", report.benchmark_root)
    print("Tasks:", report.task_count)
    print("Ready:", report.ready_task_count)
    print("Excluded by caller:", report.excluded_task_count)
    print("Blocked:", report.blocked_task_count)
    print("Status counts:", report.status_counts)
    print("Unseen status is caller-declared only: true")
    print("Contamination audit performed: false")

    for row in report.tasks:
        print()
        print("Task:", row.task_key)
        print("Status:", row.status)
        print("Ready:", row.ready_for_full_shadow_validation)
        if row.caller_declared_excluded:
            print("Excluded:", row.exclusion_reason)
            continue
        print("Domain profile:", row.domain_profile_id)
        print("Atomic hypotheses:", row.atomic_hypothesis_count)
        print("Source-candidate lineage:", row.source_candidate_lineage_match)
        print("Atomic-hypothesis lineage:", row.atomic_hypothesis_lineage_match)
        print("Grounded-annotation lineage:", row.grounded_annotation_lineage_match)
        print("Matching external reports:", row.matching_external_novelty_report_count)
        if row.selected_external_novelty_report_path:
            print("Selected external report:", row.selected_external_novelty_report_path)
        if row.missing_required_files:
            print("Missing:", row.missing_required_files)
        if row.lineage_problem_codes:
            print("Lineage problems:", row.lineage_problem_codes)
        for note in row.diagnostic_notes:
            print("Note:", note)

    print()
    print("Validation outcome assessed: false")
    print("Scientific quality judgment performed: false")
    print("Production selection changed: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
