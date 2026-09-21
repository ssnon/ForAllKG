from __future__ import annotations

import pytest

from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    readiness_trigger_alignment,
    summarize_benchmark_task_audits,
)
from pipeline_core.discovery.reframing.trigger_calibration import (
    TriggerCalibrationLabel,
    TriggerCalibrationLabels,
    build_trigger_calibration_report,
    build_trigger_review_template,
)


def _row(
    task_key: str,
    *,
    latent: bool,
    regime: bool,
    tensions: dict[str, int] | None = None,
) -> BenchmarkTaskReframeAudit:
    pattern = (
        "latent_and_regime"
        if latent and regime
        else "latent_only"
        if latent
        else "regime_only"
        if regime
        else "none"
    )
    return BenchmarkTaskReframeAudit(
        task_key=task_key,
        case_key=task_key.split("/")[0],
        replicate_key="replicate_01",
        canonical_dir=f"/bench/{task_key}/canonical",
        status="complete",
        task_id=f"task:{task_key}",
        selected_premise_count=3,
        selected_gap_count=1,
        grounded_source_chunk_count=10,
        grounded_semantic_record_count=40,
        tension_witness_count=sum((tensions or {}).values()),
        tension_type_counts=tensions or {},
        condition_example_count=5,
        condition_signature_count=4,
        condition_paper_count=2,
        readiness_status_by_operator={
            "LATENT_VARIABLE": "ready_now",
            "REGIME_BOUNDARY": "ready_with_optional_enrichment",
            "PROXY_CHALLENGE": "targeted_enrichment_required",
        },
        mandatory_backfill_count_by_operator={
            "LATENT_VARIABLE": 0,
            "REGIME_BOUNDARY": 0,
            "PROXY_CHALLENGE": 2,
        },
        optional_backfill_count_by_operator={
            "LATENT_VARIABLE": 0,
            "REGIME_BOUNDARY": 2,
            "PROXY_CHALLENGE": 0,
        },
        trigger_decision_by_operator={
            "LATENT_VARIABLE": "triggered" if latent else "not_triggered",
            "REGIME_BOUNDARY": "triggered" if regime else "not_triggered",
        },
        trigger_signal_by_operator={
            "LATENT_VARIABLE": latent,
            "REGIME_BOUNDARY": regime,
        },
        readiness_trigger_alignment_by_operator={
            "LATENT_VARIABLE": readiness_trigger_alignment(
                readiness_status="ready_now", trigger_signal=latent
            ),
            "REGIME_BOUNDARY": readiness_trigger_alignment(
                readiness_status="ready_with_optional_enrichment",
                trigger_signal=regime,
            ),
            "PROXY_CHALLENGE": "no_trigger_contract",
        },
        trigger_pattern=pattern,
    )


def _audit():
    return summarize_benchmark_task_audits(
        benchmark_root="/bench",
        extraction_roots=["/corpus"],
        task_audits=[
            _row(
                "K01/replicate_01",
                latent=True,
                regime=True,
                tensions={"BOUNDARY": 1, "CONTEXT": 1},
            ),
            _row(
                "K02/replicate_01",
                latent=True,
                regime=False,
                tensions={"CONTEXT": 1, "MECHANISTIC": 1},
            ),
            _row("K03/replicate_01", latent=False, regime=False),
        ],
    )


def test_unlabeled_report_is_prevalence_only_and_never_tunes_thresholds() -> None:
    report = build_trigger_calibration_report(audit=_audit())
    assert report.calibration_status == "unlabeled_prevalence_audit"
    assert report.ground_truth_label_count == 0
    assert report.prevalence_is_not_accuracy is True
    assert report.automatic_threshold_change_performed is False
    assert report.threshold_selection_performed is False


def test_operator_prevalence_is_conditioned_on_readiness_without_collapsing_axes() -> None:
    report = build_trigger_calibration_report(audit=_audit())
    by_id = {row.operator_id: row for row in report.operator_summaries}
    latent = by_id["LATENT_VARIABLE"]
    regime = by_id["REGIME_BOUNDARY"]
    assert latent.triggered_count == 2
    assert latent.ready_not_triggered_count == 1
    assert latent.trigger_rate == pytest.approx(2 / 3)
    assert regime.triggered_count == 1
    assert regime.ready_not_triggered_count == 2
    assert regime.trigger_rate_among_ready == pytest.approx(1 / 3)


def test_tension_association_is_descriptive_not_causal() -> None:
    report = build_trigger_calibration_report(audit=_audit())
    by_type = {row.tension_type: row for row in report.tension_associations}
    context = by_type["CONTEXT"]
    assert context.task_count == 2
    assert context.latent_triggered_count == 2
    assert context.regime_triggered_count == 1
    assert context.diagnostic_association_only is True
    assert context.causal_attribution_performed is False


def test_signature_cohorts_preserve_zero_tension_tasks() -> None:
    report = build_trigger_calibration_report(audit=_audit())
    by_signature = {
        tuple(row.tension_signature): row for row in report.tension_signature_cohorts
    }
    assert () in by_signature
    assert by_signature[()].task_keys == ["K03/replicate_01"]
    assert by_signature[()].trigger_pattern_counts == {"none": 1}


def test_review_template_has_two_operator_rows_per_complete_task_and_no_auto_labels() -> None:
    audit = _audit()
    template = build_trigger_review_template(
        audit=audit,
        question_by_task={"K01/replicate_01": "What changes across a boundary?"},
    )
    assert len(template.rows) == 6
    assert all(row.expert_label is None for row in template.rows)
    row = next(
        row
        for row in template.rows
        if row.task_key == "K01/replicate_01"
        and row.operator_id == "REGIME_BOUNDARY"
    )
    assert row.question == "What changes across a boundary?"
    assert row.tension_types == ["BOUNDARY", "CONTEXT"]


def test_labels_enable_confusion_metrics_without_changing_thresholds() -> None:
    audit = _audit()
    labels = TriggerCalibrationLabels(
        source_audit_id=audit.audit_id,
        labels=[
            TriggerCalibrationLabel(
                task_key="K01/replicate_01",
                operator_id="LATENT_VARIABLE",
                expert_label="should_attempt",
            ),
            TriggerCalibrationLabel(
                task_key="K02/replicate_01",
                operator_id="LATENT_VARIABLE",
                expert_label="should_not_attempt",
            ),
            TriggerCalibrationLabel(
                task_key="K03/replicate_01",
                operator_id="LATENT_VARIABLE",
                expert_label="should_not_attempt",
            ),
        ],
    )
    report = build_trigger_calibration_report(audit=audit, labels=labels)
    latent = next(
        row for row in report.operator_summaries if row.operator_id == "LATENT_VARIABLE"
    )
    assert report.calibration_status == "partially_labeled_performance_audit"
    assert latent.true_positive_count == 1
    assert latent.false_positive_count == 1
    assert latent.true_negative_count == 1
    assert latent.false_negative_count == 0
    assert latent.precision == pytest.approx(0.5)
    assert latent.recall == pytest.approx(1.0)
    assert report.automatic_threshold_change_performed is False


def test_uncertain_labels_are_excluded_from_precision_recall() -> None:
    audit = _audit()
    labels = TriggerCalibrationLabels(
        source_audit_id=audit.audit_id,
        labels=[
            TriggerCalibrationLabel(
                task_key="K01/replicate_01",
                operator_id="REGIME_BOUNDARY",
                expert_label="uncertain",
            )
        ],
    )
    report = build_trigger_calibration_report(audit=audit, labels=labels)
    regime = next(
        row for row in report.operator_summaries if row.operator_id == "REGIME_BOUNDARY"
    )
    assert regime.labeled_task_count == 0
    assert regime.uncertain_label_count == 1
    assert regime.precision is None
    assert regime.recall is None


def test_labels_fail_closed_on_wrong_audit_or_unknown_task() -> None:
    audit = _audit()
    wrong = TriggerCalibrationLabels(
        source_audit_id="other-audit",
        labels=[],
    )
    with pytest.raises(ValueError):
        build_trigger_calibration_report(audit=audit, labels=wrong)

    unknown = TriggerCalibrationLabels(
        source_audit_id=audit.audit_id,
        labels=[
            TriggerCalibrationLabel(
                task_key="K99/replicate_01",
                operator_id="LATENT_VARIABLE",
                expert_label="should_attempt",
            )
        ],
    )
    with pytest.raises(ValueError):
        build_trigger_calibration_report(audit=audit, labels=unknown)
