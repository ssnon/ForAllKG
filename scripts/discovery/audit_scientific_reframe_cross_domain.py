from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    summarize_benchmark_task_audits,
)
from pipeline_core.discovery.reframing.cross_domain_validation import (
    build_cross_domain_validation_report,
    select_cross_domain_tasks,
)
from pipeline_core.discovery.reframing.generalization_probe import (
    ScientificReframeGeneralizationPreflight,
    build_generalization_preflight,
)
from scripts.discovery.audit_scientific_reframe_benchmark import _complete_task_audit


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _error_row(task, exc: Exception) -> BenchmarkTaskReframeAudit:
    return BenchmarkTaskReframeAudit(
        task_key=task.task_key,
        case_key=task.case_key,
        replicate_key=task.replicate_key,
        canonical_dir=task.canonical_dir,
        status="error",
        task_id=task.task_id,
        error_type=type(exc).__name__,
        error_message=str(exc),
        trigger_pattern="unavailable",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a zero-LLM frozen-semantics trigger audit on one preflight-selected "
            "cross-domain validation cohort. This is descriptive validation only."
        )
    )
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Extraction/data root. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--scope-mode",
        choices=("premise_and_gap", "premise_only"),
        default="premise_and_gap",
    )
    parser.add_argument(
        "--duplicate-paper-policy",
        choices=("error", "latest_attempt"),
        default="error",
    )
    parser.add_argument(
        "--cross-root-duplicate-policy",
        choices=("error", "prefer_last_root"),
        default="error",
    )
    parser.add_argument("--max-condition-examples", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-task-artifacts", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    supplied = ScientificReframeGeneralizationPreflight.model_validate(_load(args.preflight))
    fresh = build_generalization_preflight(
        freeze_path=args.freeze,
        search_root=supplied.search_root,
        repository_root=args.repository_root,
        source_domain_markers=supplied.source_domain_markers,
    )
    if fresh.preflight_id != supplied.preflight_id:
        raise SystemExit(
            "preflight is stale relative to the filesystem or frozen semantics; "
            "rerun inventory_scientific_reframe_validation first"
        )
    if not fresh.semantics_unchanged:
        raise SystemExit("frozen semantics changed; cross-domain validation aborted")

    selected = select_cross_domain_tasks(
        fresh,
        validation_domain_label=args.domain,
    )

    task_audits: list[BenchmarkTaskReframeAudit] = []
    for task in selected:
        try:
            row = _complete_task_audit(
                benchmark_root=Path(task.inferred_benchmark_root),
                canonical_dir=Path(task.canonical_dir),
                corpus_roots=args.root,
                scope_mode=args.scope_mode,
                duplicate_paper_policy=args.duplicate_paper_policy,
                cross_root_duplicate_policy=args.cross_root_duplicate_policy,
                max_condition_examples=args.max_condition_examples,
                write_task_artifacts=args.write_task_artifacts,
            )
            # Use preflight identity as the validation key even when the underlying
            # benchmark layout is flat or nested differently from the SERS calibration.
            row = row.model_copy(
                update={
                    "task_key": task.task_key,
                    "case_key": task.case_key,
                    "replicate_key": task.replicate_key,
                }
            )
        except Exception as exc:  # batch isolation is intentional
            row = _error_row(task, exc)
        task_audits.append(row)
        if row.status == "complete":
            print(
                f"{row.task_key}: trigger={row.trigger_pattern}; "
                f"tensions={row.tension_witness_count}; chunks={row.grounded_source_chunk_count}; "
                f"direct={sum(row.direct_trigger_signal_kind_counts.values())}"
            )
        else:
            print(f"{row.task_key}: ERROR {row.error_type}: {row.error_message}")

    audit = summarize_benchmark_task_audits(
        benchmark_root=f"cross-domain:{args.domain}",
        extraction_roots=[str(Path(root).resolve()) for root in args.root],
        task_audits=task_audits,
    )
    report = build_cross_domain_validation_report(
        preflight=fresh,
        validation_domain_label=args.domain.strip().casefold().replace("-", "_"),
        extraction_roots=[str(Path(root).resolve()) for root in args.root],
        selected_tasks=selected,
        audit=audit,
    )

    output = args.output
    if output is None:
        output = Path(fresh.search_root) / f"scientific_reframing_cross_domain_{report.validation_domain_label}_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print("\nScientific reframing cross-domain trigger audit complete")
    print("LLM calls: 0")
    print("Calibration domain:", report.calibration_domain_label)
    print("Validation domain:", report.validation_domain_label)
    print("Validation scale:", report.validation_scale)
    print("Selected tasks:", report.selected_task_count)
    print("Complete tasks:", report.complete_task_count)
    print("Error tasks:", report.error_task_count)
    print("Triggered tasks:", report.triggered_task_count)
    print("Trigger patterns:")
    if report.trigger_pattern_counts:
        for pattern, count in report.trigger_pattern_counts.items():
            print(f"  {pattern}: {count}")
    else:
        print("  none")
    print("Tension taxonomy by task:")
    if report.tension_type_task_counts:
        for tension_type, count in report.tension_type_task_counts.items():
            print(f"  {tension_type}: {count}")
    else:
        print("  none")
    print("Direct trigger signals by task:")
    if report.direct_trigger_signal_task_counts:
        for kind, count in report.direct_trigger_signal_task_counts.items():
            print(f"  {kind}: {count}")
    else:
        print("  none")
    print(
        "Frozen source-domain marker hits carried as static diagnostics:",
        report.source_domain_marker_hit_count,
    )
    print(
        "This run does not evaluate cross-domain accuracy, marker causality, or "
        "establish generalization."
    )
    print("Output:", output.resolve())

    if args.fail_on_error and report.error_task_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
