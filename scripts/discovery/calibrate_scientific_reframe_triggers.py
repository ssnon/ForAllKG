from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.reframing.benchmark_audit import (
    ScientificReframeBenchmarkAuditReport,
)
from pipeline_core.discovery.reframing.benchmark_coverage import (
    ScientificReframeBenchmarkCoverageReport,
)
from pipeline_core.discovery.reframing.trigger_calibration import (
    TriggerCalibrationLabels,
    build_trigger_calibration_report,
    build_trigger_review_template,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _question_by_task(
    audit: ScientificReframeBenchmarkAuditReport,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in audit.task_audits:
        if row.status != "complete":
            continue
        packet_path = Path(row.canonical_dir) / "explorer.packet.json"
        if not packet_path.is_file():
            continue
        try:
            payload = _load(packet_path)
        except Exception:
            continue
        task = payload.get("task", {})
        if not isinstance(task, dict):
            continue
        question = str(task.get("question", "")).strip()
        if question:
            result[row.task_key] = question
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit scientific-reframing trigger prevalence and prepare an optional "
            "expert calibration review set. No threshold is changed automatically."
        )
    )
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--coverage", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--review-template-output", type=Path, default=None)
    args = parser.parse_args()

    audit = ScientificReframeBenchmarkAuditReport.model_validate(_load(args.audit))
    coverage = (
        ScientificReframeBenchmarkCoverageReport.model_validate(_load(args.coverage))
        if args.coverage is not None
        else None
    )
    labels = (
        TriggerCalibrationLabels.model_validate(_load(args.labels))
        if args.labels is not None
        else None
    )

    report = build_trigger_calibration_report(
        audit=audit,
        coverage=coverage,
        labels=labels,
    )
    template = build_trigger_review_template(
        audit=audit,
        question_by_task=_question_by_task(audit),
    )

    output = args.output or args.audit.with_name(
        "scientific_reframing_trigger_calibration.json"
    )
    template_output = args.review_template_output or args.audit.with_name(
        "scientific_reframing_trigger_review_template.json"
    )
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    template_output.write_text(
        template.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    print("Scientific reframing trigger calibration audit complete")
    print("LLM calls: 0")
    print("Complete tasks:", report.complete_task_count)
    print("Coverage crosscheck:", report.coverage_crosscheck_status)
    print("Calibration status:", report.calibration_status)
    for summary in report.operator_summaries:
        rate = (
            "n/a"
            if summary.trigger_rate is None
            else f"{summary.trigger_rate:.1%}"
        )
        ready_rate = (
            "n/a"
            if summary.trigger_rate_among_ready is None
            else f"{summary.trigger_rate_among_ready:.1%}"
        )
        print(
            f"  {summary.operator_id}: triggered={summary.triggered_count}/"
            f"{summary.evaluated_task_count} ({rate}); "
            f"among_ready={summary.ready_and_triggered_count}/"
            f"{summary.ready_without_mandatory_count} ({ready_rate}); "
            f"ready_not_triggered={summary.ready_not_triggered_count}"
        )
        if summary.labeled_task_count:
            print(
                "    labels:",
                f"n={summary.labeled_task_count};",
                f"TP={summary.true_positive_count};",
                f"FP={summary.false_positive_count};",
                f"TN={summary.true_negative_count};",
                f"FN={summary.false_negative_count};",
                "precision=",
                "n/a" if summary.precision is None else f"{summary.precision:.3f}",
                "recall=",
                "n/a" if summary.recall is None else f"{summary.recall:.3f}",
            )

    print("Tension associations (diagnostic, not causal):")
    for row in report.tension_associations:
        latent_rate = (
            "n/a"
            if row.latent_trigger_rate_with_type is None
            else f"{row.latent_trigger_rate_with_type:.1%}"
        )
        regime_rate = (
            "n/a"
            if row.regime_trigger_rate_with_type is None
            else f"{row.regime_trigger_rate_with_type:.1%}"
        )
        print(
            f"  {row.tension_type}: tasks={row.task_count}; "
            f"latent={row.latent_triggered_count} ({latent_rate}); "
            f"regime={row.regime_triggered_count} ({regime_rate})"
        )

    print("Ground-truth calibration labels:", report.ground_truth_label_count)
    print("Automatic threshold changes: 0")
    print("Review rows:", len(template.rows))
    print("Review template:", template_output)
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
