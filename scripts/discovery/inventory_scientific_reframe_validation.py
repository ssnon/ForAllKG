from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.reframing.generalization_probe import (
    build_generalization_preflight,
)


DEFAULT_MARKERS = ("sers", "raman", "lspr", "plasmon", "hotspot", "electromagnetic")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory untouched scientific-reframing validation materializations and "
            "audit source-domain lexical dependencies without modifying frozen semantics."
        )
    )
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--marker",
        action="append",
        default=None,
        help="Source-domain marker to scan in frozen semantics. Repeat as needed.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    markers = args.marker if args.marker else list(DEFAULT_MARKERS)
    report = build_generalization_preflight(
        freeze_path=args.freeze,
        search_root=args.search_root,
        repository_root=args.repository_root,
        source_domain_markers=markers,
    )
    output = args.output or args.search_root.resolve() / "scientific_reframing_generalization_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("Scientific reframing generalization preflight complete")
    print("LLM calls: 0")
    print("Frozen semantics unchanged:", str(report.semantics_unchanged).lower())
    print("Task materializations:", report.discovered_task_materialization_count)
    print("Complete packet/report/context triplets:", report.complete_triplet_count)
    print("Incomplete materializations:", report.incomplete_materialization_count)
    print("Calibration overlap detected:", str(report.calibration_overlap_detected).lower())
    print("Frozen source-domain marker hits:", report.source_domain_marker_hit_count)
    if report.marker_hit_counts:
        print("Marker counts:")
        for marker, count in report.marker_hit_counts.items():
            print(f"  {marker}: {count}")
    print("Benchmark groups:")
    for row in report.benchmark_inventories:
        raw_domains = ",".join(row.inferred_domain_labels) if row.inferred_domain_labels else "unknown"
        canonical_domains = ",".join(row.canonical_domain_labels) if row.canonical_domain_labels else "unknown"
        print(
            f"  {row.benchmark_root}: class={row.domain_classification}; "
            f"complete={row.complete_task_count}; incomplete={row.incomplete_task_count}; "
            f"cases={row.case_count}; raw_domains={raw_domains}; "
            f"canonical_domains={canonical_domains}; overlap={str(row.frozen_overlap_detected).lower()}"
        )
    print("Cross-domain candidate roots:", len(report.cross_domain_candidate_roots))
    for root in report.cross_domain_candidate_roots:
        print("  ", root)
    print("Cross-domain candidate cohorts:", len(report.cross_domain_candidate_cohorts))
    for cohort in report.cross_domain_candidate_cohorts:
        roots = ",".join(cohort.inferred_benchmark_roots)
        raw = ",".join(cohort.raw_domain_labels)
        print(
            f"  domain={cohort.canonical_domain_label}; tasks={cohort.task_count}; "
            f"cases={cohort.case_count}; roots={roots}; raw_domains={raw}"
        )
    print("Unclassified complete roots:", len(report.unclassified_candidate_roots))
    for root in report.unclassified_candidate_roots:
        print("  ", root)
    print("Static marker hits are diagnostic only; they do not prove runtime dependency.")
    print("No trigger, prompt, critic, or portfolio semantics were modified.")
    print("Output:", output)
    return 0 if report.semantics_unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
