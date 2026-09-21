from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.corpus.semantic_ir.grounded_scope import (
    resolve_grounded_semantic_task_scope,
)
from pipeline_core.corpus.semantic_ir.task_capability import (
    build_task_capability_snapshot,
)
from pipeline_core.corpus.semantic_ir.task_gap import (
    detect_task_local_capability_gaps,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    readiness_trigger_alignment,
    summarize_benchmark_task_audits,
    trigger_pattern_from_signals,
)
from pipeline_core.discovery.reframing.evidence_tension import (
    extract_evidence_level_tensions,
)
from pipeline_core.discovery.reframing.operator_readiness import (
    assess_reframing_operator_readiness,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    build_scientific_reframe_evidence_packet,
)
from pipeline_core.discovery.reframing.trigger_detection import (
    detect_scientific_reframe_triggers,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _task_keys(benchmark_root: Path, canonical_dir: Path) -> tuple[str, str, str]:
    relative = canonical_dir.relative_to(benchmark_root)

    # Flat materialization: packet/report/context live directly in a task
    # directory such as ``evaluation/s28_dac_her106_canonical_parity_v1``.
    # The historical benchmark layout instead ends in ``.../canonical``.
    if canonical_dir.name != "canonical":
        parts = relative.parts
        if not parts:
            raise ValueError(
                f"flat task directory must be nested under benchmark root: {canonical_dir}"
            )
        task_key = relative.as_posix()
        case_key = parts[0]
        replicate_key = parts[-1] if len(parts) > 1 else "default"
        return task_key, case_key, replicate_key

    task_relative = relative.parent
    parts = task_relative.parts
    if not parts:
        raise ValueError(f"canonical directory is not nested under benchmark root: {canonical_dir}")
    task_key = task_relative.as_posix()
    case_key = parts[0]
    replicate_key = parts[-1] if len(parts) > 1 else "default"
    return task_key, case_key, replicate_key


def _backfill_count(plan: Any) -> int:
    value = getattr(plan, "target_count", None)
    if value is not None:
        return int(value)
    targets = getattr(plan, "targets", [])
    return len(targets)


def _complete_task_audit(
    *,
    benchmark_root: Path,
    canonical_dir: Path,
    corpus_roots: list[str],
    scope_mode: str,
    duplicate_paper_policy: str,
    cross_root_duplicate_policy: str,
    max_condition_examples: int,
    write_task_artifacts: bool,
) -> BenchmarkTaskReframeAudit:
    task_key, case_key, replicate_key = _task_keys(benchmark_root, canonical_dir)
    packet_path = canonical_dir / "explorer.packet.json"
    report_path = canonical_dir / "explorer.report.json"
    context_path = canonical_dir / "hypothesis.context.json"
    missing = [
        str(path.name)
        for path in (packet_path, report_path, context_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing canonical benchmark artifact(s): " + ", ".join(missing)
        )

    grounded, bundles = resolve_grounded_semantic_task_scope(
        packet_path=packet_path,
        context_path=context_path,
        corpus_roots=corpus_roots,
        scope_mode=scope_mode,
        duplicate_paper_policy=duplicate_paper_policy,
        cross_root_duplicate_policy=cross_root_duplicate_policy,
    )
    gaps = detect_task_local_capability_gaps(
        bundles=bundles,
        task_source_chunks=grounded.resolution.source_chunks,
        capabilities=[
            "calculation_condition_coverage",
            "experiment_condition_coverage",
            "measurement_condition_coverage",
        ],
    )
    snapshot = build_task_capability_snapshot(
        scope_id=grounded.selection.task_id,
        bundles=bundles,
        task_source_chunks=grounded.resolution.source_chunks,
        task_gap_report=gaps,
    )
    readiness = assess_reframing_operator_readiness(
        scope_id=grounded.selection.task_id,
        task_capabilities=snapshot,
        gap_report=gaps,
        bundles=bundles,
        task_source_chunks=grounded.resolution.source_chunks,
        unresolved_grounded_node_count=len(
            grounded.resolution.unresolved_paper_local_node_ids
        ),
        ambiguous_grounded_node_count=len(
            grounded.resolution.ambiguous_paper_local_node_ids
        ),
    )

    context = HypothesisContext.model_validate(_load(context_path))
    evidence = build_scientific_reframe_evidence_packet(
        context=context,
        grounded=grounded,
        bundles=bundles,
        max_condition_examples=max_condition_examples,
    )
    explorer_report = _load(report_path)
    explorer_packet = _load(packet_path)
    tensions = extract_evidence_level_tensions(
        explorer_report=explorer_report,
        evidence=evidence,
        explorer_packet=explorer_packet,
    )
    triggers = detect_scientific_reframe_triggers(
        explorer_report=explorer_report,
        evidence=evidence,
        explorer_packet=explorer_packet,
        evidence_tension_report=tensions,
    )

    if write_task_artifacts:
        (canonical_dir / "scientific_reframing_evidence.json").write_text(
            evidence.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (canonical_dir / "scientific_reframing_evidence_tensions.json").write_text(
            tensions.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (canonical_dir / "scientific_reframing_triggers.json").write_text(
            triggers.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (canonical_dir / "reframing_operator_readiness.json").write_text(
            readiness.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    readiness_by_operator = {row.operator_id: row for row in readiness.assessments}
    trigger_by_operator = {row.operator_id: row for row in triggers.assessments}

    readiness_status = {
        operator_id: row.status
        for operator_id, row in readiness_by_operator.items()
    }
    mandatory_backfill = {
        operator_id: _backfill_count(row.mandatory_backfill_plan)
        for operator_id, row in readiness_by_operator.items()
    }
    optional_backfill = {
        operator_id: _backfill_count(row.optional_backfill_plan)
        for operator_id, row in readiness_by_operator.items()
    }
    trigger_decision = {
        operator_id: row.decision
        for operator_id, row in trigger_by_operator.items()
    }
    trigger_signal = {
        operator_id: row.scientific_trigger_signal
        for operator_id, row in trigger_by_operator.items()
    }

    latent_signal = trigger_signal.get("LATENT_VARIABLE")
    regime_signal = trigger_signal.get("REGIME_BOUNDARY")
    alignment = {
        "LATENT_VARIABLE": readiness_trigger_alignment(
            readiness_status=readiness_status.get("LATENT_VARIABLE"),
            trigger_signal=latent_signal,
        ),
        "REGIME_BOUNDARY": readiness_trigger_alignment(
            readiness_status=readiness_status.get("REGIME_BOUNDARY"),
            trigger_signal=regime_signal,
        ),
        "PROXY_CHALLENGE": readiness_trigger_alignment(
            readiness_status=readiness_status.get("PROXY_CHALLENGE"),
            trigger_signal=None,
            has_trigger_contract=False,
        ),
    }

    measurement_tension = tensions.tension_type_counts.get("MEASUREMENT", 0) > 0
    proxy_readiness = readiness_status.get("PROXY_CHALLENGE")
    proxy_candidate = bool(
        measurement_tension
        and proxy_readiness in {
            "targeted_enrichment_required",
            "ready_now",
            "ready_with_optional_enrichment",
        }
    )
    proxy_reason = None
    if proxy_candidate:
        proxy_reason = (
            "Evidence-level MEASUREMENT tension is present and the grounded task "
            "scope localizes a usable or targetable PROXY_CHALLENGE substrate. "
            "This is an enrichment-candidate diagnostic, not proxy-semantic authority."
        )

    return BenchmarkTaskReframeAudit(
        task_key=task_key,
        case_key=case_key,
        replicate_key=replicate_key,
        canonical_dir=str(canonical_dir.resolve()),
        status="complete",
        task_id=evidence.task_id,
        selected_premise_count=len(evidence.premise_statements),
        selected_gap_count=len(evidence.gap_statements),
        grounded_source_chunk_count=grounded.source_chunk_count,
        grounded_semantic_record_count=snapshot.semantic_record_count,
        unresolved_grounded_node_count=len(
            grounded.resolution.unresolved_paper_local_node_ids
        ),
        ambiguous_grounded_node_count=len(
            grounded.resolution.ambiguous_paper_local_node_ids
        ),
        tension_witness_count=len(tensions.witnesses),
        tension_type_counts=dict(tensions.tension_type_counts),
        direct_trigger_signal_kind_counts={
            kind: sum(signal.kind == kind for signal in triggers.direct_trigger_signals)
            for kind in sorted({signal.kind for signal in triggers.direct_trigger_signals})
        },
        condition_example_count=triggers.condition_diversity.example_count,
        condition_signature_count=(
            triggers.condition_diversity.distinct_condition_signature_count
        ),
        condition_paper_count=triggers.condition_diversity.paper_count,
        readiness_status_by_operator=readiness_status,
        mandatory_backfill_count_by_operator=mandatory_backfill,
        optional_backfill_count_by_operator=optional_backfill,
        trigger_decision_by_operator=trigger_decision,
        trigger_signal_by_operator=trigger_signal,
        readiness_trigger_alignment_by_operator=alignment,
        trigger_pattern=trigger_pattern_from_signals(
            latent=latent_signal,
            regime=regime_signal,
        ),
        measurement_tension_present=measurement_tension,
        proxy_enrichment_candidate=proxy_candidate,
        proxy_candidate_reason=proxy_reason,
    )


def _error_task_audit(
    *,
    benchmark_root: Path,
    canonical_dir: Path,
    exc: Exception,
) -> BenchmarkTaskReframeAudit:
    task_key, case_key, replicate_key = _task_keys(benchmark_root, canonical_dir)
    return BenchmarkTaskReframeAudit(
        task_key=task_key,
        case_key=case_key,
        replicate_key=replicate_key,
        canonical_dir=str(canonical_dir.resolve()),
        status="error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        trigger_pattern="unavailable",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a zero-LLM benchmark-wide audit of grounded reframing readiness, "
            "evidence-level tensions, and scientific trigger gating."
        )
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Extraction root. Repeat for merged/legacy corpus roots.",
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
    parser.add_argument(
        "--write-task-artifacts",
        action="store_true",
        help=(
            "Also write readiness/evidence/tension/trigger JSON beside each task. "
            "Default audit is read-only except for the benchmark summary."
        ),
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any discovered benchmark task cannot be audited.",
    )
    args = parser.parse_args()

    benchmark_root = args.benchmark_root.resolve()
    canonical_dirs = sorted(
        {
            path.parent
            for path in benchmark_root.rglob("canonical/explorer.packet.json")
        },
        key=lambda path: path.as_posix(),
    )
    if not canonical_dirs:
        raise SystemExit(
            f"No canonical/explorer.packet.json found under {benchmark_root}"
        )

    rows: list[BenchmarkTaskReframeAudit] = []
    for canonical_dir in canonical_dirs:
        try:
            row = _complete_task_audit(
                benchmark_root=benchmark_root,
                canonical_dir=canonical_dir,
                corpus_roots=list(args.root),
                scope_mode=args.scope_mode,
                duplicate_paper_policy=args.duplicate_paper_policy,
                cross_root_duplicate_policy=args.cross_root_duplicate_policy,
                max_condition_examples=args.max_condition_examples,
                write_task_artifacts=args.write_task_artifacts,
            )
        except Exception as exc:  # batch audit deliberately isolates task failures
            row = _error_task_audit(
                benchmark_root=benchmark_root,
                canonical_dir=canonical_dir,
                exc=exc,
            )
        rows.append(row)
        if row.status == "complete":
            print(
                f"{row.task_key}: trigger={row.trigger_pattern}; "
                f"tensions={row.tension_witness_count}; "
                f"chunks={row.grounded_source_chunk_count}; "
                f"measurement={str(row.measurement_tension_present).lower()}"
            )
        else:
            print(
                f"{row.task_key}: ERROR {row.error_type}: {row.error_message}"
            )

    report = summarize_benchmark_task_audits(
        benchmark_root=str(benchmark_root),
        extraction_roots=[str(Path(root).resolve()) for root in args.root],
        task_audits=rows,
    )
    output = args.output or (
        benchmark_root / "scientific_reframing_trigger_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    print()
    print("Scientific reframing benchmark audit complete")
    print("LLM calls: 0")
    print("Discovered tasks:", report.discovered_task_count)
    print("Complete tasks:", report.complete_task_count)
    print("Error tasks:", report.error_task_count)
    print("Unique cases:", report.unique_case_count)
    print("Trigger patterns:")
    for pattern, count in sorted(report.trigger_pattern_counts.items()):
        print(f"  {pattern}: {count}")
    print("Tension taxonomy by task:")
    if report.tension_type_task_counts:
        for tension_type, count in sorted(report.tension_type_task_counts.items()):
            witnesses = report.tension_type_witness_counts.get(tension_type, 0)
            print(f"  {tension_type}: tasks={count}; witnesses={witnesses}")
    else:
        print("  none")
    print("Direct trigger signals by task:")
    if report.direct_trigger_signal_task_counts:
        for kind, task_count in report.direct_trigger_signal_task_counts.items():
            signal_count = report.direct_trigger_signal_counts.get(kind, 0)
            print(f"  {kind}: tasks={task_count}; signals={signal_count}")
    else:
        print("  none")
    print("Readiness / trigger alignment:")
    for operator_id, counts in sorted(
        report.readiness_trigger_alignment_counts.items()
    ):
        rendered = ", ".join(
            f"{key}={value}" for key, value in sorted(counts.items())
        )
        print(f"  {operator_id}: {rendered}")
    print("Measurement-tension tasks:", report.measurement_tension_task_count)
    print(
        "Future PROXY enrichment candidates:",
        report.proxy_enrichment_candidate_task_count,
    )
    print(
        "No hypothesis generation, scientific conflict authority, novelty, "
        "ranking, or production selection was performed."
    )
    print("Output:", output)

    if args.fail_on_error and report.error_task_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
