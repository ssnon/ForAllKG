from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.benchmark_coverage import (
    inspect_benchmark_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect benchmark filesystem/materialization coverage without treating "
            "numbers embedded in the benchmark name as expected task counts."
        )
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=None,
        help=(
            "Optional explicit expectation. Omit unless the benchmark definition "
            "actually states a task count; name tokens such as 'sers18' are not used."
        ),
    )
    parser.add_argument(
        "--prior-audit",
        type=Path,
        default=None,
        help=(
            "Optional 0014 audit JSON. Defaults to "
            "<benchmark-root>/scientific_reframing_trigger_audit.json when present."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = inspect_benchmark_coverage(
        benchmark_root=args.benchmark_root,
        expected_task_count=args.expected_task_count,
        prior_audit_path=args.prior_audit,
    )
    output = args.output or (
        args.benchmark_root.resolve() / "scientific_reframing_benchmark_coverage.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Scientific reframing benchmark coverage inspection complete")
    print("LLM calls: 0")
    print("Benchmark:", report.benchmark_name)
    print(
        "Benchmark-name numeric tokens (diagnostic only, never expected counts):",
        report.benchmark_name_numeric_tokens,
    )
    print("Top-level directories:", len(report.top_level_directory_names))
    print("Case-like top-level directories:", len(report.top_level_case_like_directory_names))
    if report.top_level_case_like_directory_names:
        print("  " + ", ".join(report.top_level_case_like_directory_names))
    print("replicate_* directories:", report.replicate_directory_count)
    print("Canonical directories:", report.canonical_directory_count)
    print("Task materialization candidates:", report.materialization_candidate_count)
    print("0014-discoverable tasks (explorer.packet present):", report.audit_discoverable_task_count)
    print("Fully auditable tasks (packet + report + context):", report.fully_auditable_task_count)
    print("Incomplete task materializations:", report.incomplete_materialization_count)
    print("  existing canonical but incomplete:", report.partially_materialized_canonical_count)
    print("  replicate_* with missing canonical:", report.missing_canonical_count)
    print("Unique materialized cases:", report.unique_materialized_case_count)

    partial = [row for row in report.task_materializations if not row.fully_auditable]
    if partial:
        print("Incomplete task materializations:")
        for row in partial:
            print(
                f"  {row.task_key}: canonical_exists={str(row.canonical_dir_exists).lower()}; "
                f"missing={','.join(row.missing_required_artifacts) or '-'}"
            )

    if report.prior_audit_present:
        print("Prior 0014 audit tasks:", report.prior_audited_task_count)
        print(
            "Filesystem discoverable but absent from prior audit:",
            len(report.filesystem_discoverable_missing_from_prior_audit),
        )
        for task_key in report.filesystem_discoverable_missing_from_prior_audit:
            print("  +", task_key)
        print(
            "Prior audit task absent from current filesystem:",
            len(report.prior_audit_tasks_missing_from_filesystem),
        )
        for task_key in report.prior_audit_tasks_missing_from_filesystem:
            print("  -", task_key)
    else:
        print("Prior 0014 audit: not found")

    if report.expected_task_count is None:
        print("Expected task count: unspecified")
        print(
            "Coverage conclusion: no explicit expected count; filesystem/audit consistency "
            "can be checked, but benchmark completeness is not asserted from its name."
        )
    else:
        print(
            "Expected task count (explicit CLI assertion):",
            report.expected_task_count,
        )
        print("Task-count shortfall:", report.task_count_shortfall)
        print("Coverage status:", report.coverage_status)

    if report.metadata_json_candidates:
        print("Top-level metadata JSON candidates:")
        for name in report.metadata_json_candidates:
            print("  ", name)
    else:
        print("Top-level metadata JSON candidates: none")

    print("No benchmark-name token was interpreted as task-count authority.")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
