from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    ScientificReframeBenchmarkAuditReport,
    operator_ready_without_mandatory_enrichment,
)
from pipeline_core.discovery.reframing.benchmark_coverage import (
    ScientificReframeBenchmarkCoverageReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


OperatorId = Literal["LATENT_VARIABLE", "REGIME_BOUNDARY"]
ExpertTriggerLabel = Literal["should_attempt", "should_not_attempt", "uncertain"]
CoverageCrosscheckStatus = Literal[
    "not_supplied",
    "filesystem_consistent",
    "filesystem_mismatch",
]
CalibrationStatus = Literal[
    "unlabeled_prevalence_audit",
    "partially_labeled_performance_audit",
    "fully_labeled_performance_audit",
]

_OPERATORS: tuple[OperatorId, ...] = (
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
)


class TriggerReviewItem(StrictModel):
    task_key: str = Field(min_length=1)
    case_key: str = Field(min_length=1)
    operator_id: OperatorId
    question: str | None = None
    readiness_status: str | None = None
    trigger_decision: str | None = None
    trigger_signal: bool | None = None
    readiness_trigger_alignment: str | None = None
    tension_types: list[str] = Field(default_factory=list)
    tension_witness_count: int = Field(ge=0, default=0)
    condition_signature_count: int = Field(ge=0, default=0)
    grounded_source_chunk_count: int = Field(ge=0, default=0)
    expert_label: ExpertTriggerLabel | None = None
    expert_rationale: str | None = None

    @model_validator(mode="after")
    def validate_label_rationale(self) -> "TriggerReviewItem":
        if self.expert_label is None and self.expert_rationale is not None:
            raise ValueError("expert_rationale requires expert_label")
        return self


class TriggerReviewTemplate(StrictModel):
    schema_version: Literal[
        "scientific-reframe-trigger-review-template-v1"
    ] = "scientific-reframe-trigger-review-template-v1"
    source_audit_id: str = Field(min_length=1)
    rows: list[TriggerReviewItem]
    operator_ids: list[OperatorId] = Field(default_factory=lambda: list(_OPERATORS))
    label_semantics: dict[str, str] = Field(
        default_factory=lambda: {
            "should_attempt": (
                "Given only the grounded task evidence, this scientific reasoning "
                "operator is worth one shadow generation attempt."
            ),
            "should_not_attempt": (
                "Given only the grounded task evidence, this operator should not "
                "consume a generation call for this task."
            ),
            "uncertain": (
                "The evidence is insufficient for a reliable calibration label; "
                "exclude this row from precision/recall metrics."
            ),
        }
    )
    labels_are_scientific_authority: Literal[False] = False
    threshold_selection_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_unique_rows(self) -> "TriggerReviewTemplate":
        keys = [(row.task_key, row.operator_id) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("review template rows must be unique by task/operator")
        return self


class TriggerCalibrationLabel(StrictModel):
    task_key: str = Field(min_length=1)
    operator_id: OperatorId
    expert_label: ExpertTriggerLabel
    rationale: str | None = None


class TriggerCalibrationLabels(StrictModel):
    schema_version: Literal[
        "scientific-reframe-trigger-calibration-labels-v1"
    ] = "scientific-reframe-trigger-calibration-labels-v1"
    source_audit_id: str = Field(min_length=1)
    labels: list[TriggerCalibrationLabel]
    label_source: str = Field(default="expert_review", min_length=1)
    labels_are_scientific_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_unique_labels(self) -> "TriggerCalibrationLabels":
        keys = [(row.task_key, row.operator_id) for row in self.labels]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate task/operator calibration label")
        return self


class OperatorCalibrationSummary(StrictModel):
    operator_id: OperatorId
    evaluated_task_count: int = Field(ge=0)
    ready_without_mandatory_count: int = Field(ge=0)
    triggered_count: int = Field(ge=0)
    not_triggered_count: int = Field(ge=0)
    ready_and_triggered_count: int = Field(ge=0)
    ready_not_triggered_count: int = Field(ge=0)
    triggered_not_ready_count: int = Field(ge=0)
    trigger_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    trigger_rate_among_ready: float | None = Field(default=None, ge=0.0, le=1.0)

    labeled_task_count: int = Field(ge=0, default=0)
    uncertain_label_count: int = Field(ge=0, default=0)
    true_positive_count: int = Field(ge=0, default=0)
    false_positive_count: int = Field(ge=0, default=0)
    true_negative_count: int = Field(ge=0, default=0)
    false_negative_count: int = Field(ge=0, default=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    specificity: float | None = Field(default=None, ge=0.0, le=1.0)


class TensionAssociationSummary(StrictModel):
    tension_type: str = Field(min_length=1)
    task_count: int = Field(ge=0)
    latent_triggered_count: int = Field(ge=0)
    regime_triggered_count: int = Field(ge=0)
    latent_trigger_rate_with_type: float | None = Field(default=None, ge=0.0, le=1.0)
    regime_trigger_rate_with_type: float | None = Field(default=None, ge=0.0, le=1.0)
    diagnostic_association_only: Literal[True] = True
    causal_attribution_performed: Literal[False] = False


class TensionSignatureCohort(StrictModel):
    tension_signature: list[str]
    task_keys: list[str]
    task_count: int = Field(ge=0)
    trigger_pattern_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_count(self) -> "TensionSignatureCohort":
        if self.task_count != len(self.task_keys):
            raise ValueError("task_count must equal task_keys length")
        return self


class ScientificReframeTriggerCalibrationReport(StrictModel):
    schema_version: Literal[
        "scientific-reframe-trigger-calibration-v1"
    ] = "scientific-reframe-trigger-calibration-v1"
    calibration_id: str = Field(min_length=1)
    source_audit_id: str = Field(min_length=1)
    source_coverage_id: str | None = None
    coverage_crosscheck_status: CoverageCrosscheckStatus
    calibration_status: CalibrationStatus

    complete_task_count: int = Field(ge=0)
    operator_summaries: list[OperatorCalibrationSummary]
    tension_associations: list[TensionAssociationSummary]
    tension_signature_cohorts: list[TensionSignatureCohort]
    ground_truth_label_count: int = Field(ge=0, default=0)
    uncertain_label_count: int = Field(ge=0, default=0)

    interpretation_limits: list[str] = Field(default_factory=list)
    llm_calls_performed: Literal[0] = 0
    deterministic_calibration_audit: Literal[True] = True
    prevalence_is_not_accuracy: Literal[True] = True
    automatic_threshold_change_performed: Literal[False] = False
    threshold_selection_performed: Literal[False] = False
    candidate_ranking_performed: Literal[False] = False
    scientific_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_operator_rows(self) -> "ScientificReframeTriggerCalibrationReport":
        ids = [row.operator_id for row in self.operator_summaries]
        if sorted(ids) != sorted(_OPERATORS):
            raise ValueError("calibration report must contain exactly LATENT and REGIME summaries")
        return self


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _complete_rows(
    audit: ScientificReframeBenchmarkAuditReport,
) -> list[BenchmarkTaskReframeAudit]:
    return [row for row in audit.task_audits if row.status == "complete"]


def _coverage_crosscheck(
    audit: ScientificReframeBenchmarkAuditReport,
    coverage: ScientificReframeBenchmarkCoverageReport | None,
) -> CoverageCrosscheckStatus:
    if coverage is None:
        return "not_supplied"
    audit_keys = {
        row.task_key
        for row in audit.task_audits
        if row.status == "complete"
    }
    filesystem_keys = {
        row.task_key
        for row in coverage.task_materializations
        if row.fully_auditable
    }
    consistent = (
        audit_keys == filesystem_keys
        and not coverage.filesystem_discoverable_missing_from_prior_audit
        and not coverage.prior_audit_tasks_missing_from_filesystem
    )
    return "filesystem_consistent" if consistent else "filesystem_mismatch"


def build_trigger_review_template(
    *,
    audit: ScientificReframeBenchmarkAuditReport,
    question_by_task: Mapping[str, str] | None = None,
) -> TriggerReviewTemplate:
    questions = dict(question_by_task or {})
    rows: list[TriggerReviewItem] = []
    for task in sorted(_complete_rows(audit), key=lambda row: row.task_key):
        tension_types = sorted(
            tension_type
            for tension_type, count in task.tension_type_counts.items()
            if int(count) > 0
        )
        for operator_id in _OPERATORS:
            rows.append(
                TriggerReviewItem(
                    task_key=task.task_key,
                    case_key=task.case_key,
                    operator_id=operator_id,
                    question=(questions.get(task.task_key) or None),
                    readiness_status=task.readiness_status_by_operator.get(operator_id),
                    trigger_decision=task.trigger_decision_by_operator.get(operator_id),
                    trigger_signal=task.trigger_signal_by_operator.get(operator_id),
                    readiness_trigger_alignment=(
                        task.readiness_trigger_alignment_by_operator.get(operator_id)
                    ),
                    tension_types=tension_types,
                    tension_witness_count=task.tension_witness_count,
                    condition_signature_count=task.condition_signature_count,
                    grounded_source_chunk_count=task.grounded_source_chunk_count,
                )
            )
    return TriggerReviewTemplate(source_audit_id=audit.audit_id, rows=rows)


def _label_index(
    *,
    audit: ScientificReframeBenchmarkAuditReport,
    labels: TriggerCalibrationLabels | None,
) -> dict[tuple[str, str], TriggerCalibrationLabel]:
    if labels is None:
        return {}
    if labels.source_audit_id != audit.audit_id:
        raise ValueError("calibration labels source_audit_id does not match audit")
    valid_keys = {
        (row.task_key, operator_id)
        for row in _complete_rows(audit)
        for operator_id in _OPERATORS
    }
    result: dict[tuple[str, str], TriggerCalibrationLabel] = {}
    for label in labels.labels:
        key = (label.task_key, label.operator_id)
        if key not in valid_keys:
            raise ValueError(f"calibration label references unknown task/operator: {key}")
        result[key] = label
    return result


def _operator_summary(
    *,
    operator_id: OperatorId,
    rows: list[BenchmarkTaskReframeAudit],
    labels: Mapping[tuple[str, str], TriggerCalibrationLabel],
) -> OperatorCalibrationSummary:
    evaluated = [
        row
        for row in rows
        if operator_id in row.trigger_signal_by_operator
    ]
    triggered = [
        row for row in evaluated if row.trigger_signal_by_operator.get(operator_id) is True
    ]
    not_triggered = [
        row for row in evaluated if row.trigger_signal_by_operator.get(operator_id) is False
    ]
    ready = [
        row
        for row in evaluated
        if operator_ready_without_mandatory_enrichment(
            row.readiness_status_by_operator.get(operator_id)
        )
    ]
    ready_triggered = [
        row for row in ready if row.trigger_signal_by_operator.get(operator_id) is True
    ]
    ready_not_triggered = [
        row for row in ready if row.trigger_signal_by_operator.get(operator_id) is False
    ]
    triggered_not_ready = [
        row
        for row in triggered
        if not operator_ready_without_mandatory_enrichment(
            row.readiness_status_by_operator.get(operator_id)
        )
    ]

    tp = fp = tn = fn = uncertain = labeled = 0
    for row in evaluated:
        label = labels.get((row.task_key, operator_id))
        if label is None:
            continue
        if label.expert_label == "uncertain":
            uncertain += 1
            continue
        labeled += 1
        predicted = bool(row.trigger_signal_by_operator.get(operator_id))
        expected = label.expert_label == "should_attempt"
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and not expected:
            tn += 1
        else:
            fn += 1

    return OperatorCalibrationSummary(
        operator_id=operator_id,
        evaluated_task_count=len(evaluated),
        ready_without_mandatory_count=len(ready),
        triggered_count=len(triggered),
        not_triggered_count=len(not_triggered),
        ready_and_triggered_count=len(ready_triggered),
        ready_not_triggered_count=len(ready_not_triggered),
        triggered_not_ready_count=len(triggered_not_ready),
        trigger_rate=_ratio(len(triggered), len(evaluated)),
        trigger_rate_among_ready=_ratio(len(ready_triggered), len(ready)),
        labeled_task_count=labeled,
        uncertain_label_count=uncertain,
        true_positive_count=tp,
        false_positive_count=fp,
        true_negative_count=tn,
        false_negative_count=fn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp),
    )


def _tension_associations(
    rows: list[BenchmarkTaskReframeAudit],
) -> list[TensionAssociationSummary]:
    types = sorted(
        {
            tension_type
            for row in rows
            for tension_type, count in row.tension_type_counts.items()
            if int(count) > 0
        }
    )
    result: list[TensionAssociationSummary] = []
    for tension_type in types:
        with_type = [
            row
            for row in rows
            if int(row.tension_type_counts.get(tension_type, 0)) > 0
        ]
        latent = sum(
            row.trigger_signal_by_operator.get("LATENT_VARIABLE") is True
            for row in with_type
        )
        regime = sum(
            row.trigger_signal_by_operator.get("REGIME_BOUNDARY") is True
            for row in with_type
        )
        result.append(
            TensionAssociationSummary(
                tension_type=tension_type,
                task_count=len(with_type),
                latent_triggered_count=latent,
                regime_triggered_count=regime,
                latent_trigger_rate_with_type=_ratio(latent, len(with_type)),
                regime_trigger_rate_with_type=_ratio(regime, len(with_type)),
            )
        )
    return result


def _signature_cohorts(
    rows: list[BenchmarkTaskReframeAudit],
) -> list[TensionSignatureCohort]:
    grouped: dict[tuple[str, ...], list[BenchmarkTaskReframeAudit]] = defaultdict(list)
    for row in rows:
        signature = tuple(
            sorted(
                tension_type
                for tension_type, count in row.tension_type_counts.items()
                if int(count) > 0
            )
        )
        grouped[signature].append(row)

    result: list[TensionSignatureCohort] = []
    for signature, members in sorted(grouped.items(), key=lambda item: item[0]):
        patterns = Counter(row.trigger_pattern for row in members)
        result.append(
            TensionSignatureCohort(
                tension_signature=list(signature),
                task_keys=sorted(row.task_key for row in members),
                task_count=len(members),
                trigger_pattern_counts=dict(sorted(patterns.items())),
            )
        )
    return result


def _calibration_id(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"scientific_reframe_trigger_calibration:{digest}"


def build_trigger_calibration_report(
    *,
    audit: ScientificReframeBenchmarkAuditReport,
    coverage: ScientificReframeBenchmarkCoverageReport | None = None,
    labels: TriggerCalibrationLabels | None = None,
) -> ScientificReframeTriggerCalibrationReport:
    rows = sorted(_complete_rows(audit), key=lambda row: row.task_key)
    label_index = _label_index(audit=audit, labels=labels)
    ground_truth_count = sum(
        label.expert_label != "uncertain" for label in label_index.values()
    )
    uncertain_count = sum(
        label.expert_label == "uncertain" for label in label_index.values()
    )
    possible = len(rows) * len(_OPERATORS)
    if ground_truth_count == 0:
        status: CalibrationStatus = "unlabeled_prevalence_audit"
    elif ground_truth_count < possible:
        status = "partially_labeled_performance_audit"
    else:
        status = "fully_labeled_performance_audit"

    operator_summaries = [
        _operator_summary(
            operator_id=operator_id,
            rows=rows,
            labels=label_index,
        )
        for operator_id in _OPERATORS
    ]
    tension_associations = _tension_associations(rows)
    cohorts = _signature_cohorts(rows)
    crosscheck = _coverage_crosscheck(audit, coverage)

    limits = [
        "Trigger prevalence alone cannot establish precision, recall, or an optimal threshold.",
        "Tension-type associations are descriptive co-occurrence statistics, not causal attribution to a trigger decision.",
        "Expert should_attempt/should_not_attempt labels calibrate call allocation only; they do not create scientific truth, conflict, novelty, or hypothesis authority.",
    ]
    if coverage is None:
        limits.append(
            "No filesystem coverage report was supplied; this report does not independently verify benchmark materialization coverage."
        )
    elif crosscheck == "filesystem_mismatch":
        limits.append(
            "The supplied filesystem coverage report does not match the complete task set in the trigger audit."
        )
    if ground_truth_count == 0:
        limits.append(
            "No non-uncertain expert calibration labels were supplied; automatic threshold changes are therefore unsupported."
        )

    body = {
        "source_audit_id": audit.audit_id,
        "source_coverage_id": coverage.coverage_id if coverage is not None else None,
        "coverage_crosscheck_status": crosscheck,
        "calibration_status": status,
        "complete_task_count": len(rows),
        "operator_summaries": [row.model_dump(mode="json") for row in operator_summaries],
        "tension_associations": [row.model_dump(mode="json") for row in tension_associations],
        "tension_signature_cohorts": [row.model_dump(mode="json") for row in cohorts],
        "ground_truth_label_count": ground_truth_count,
        "uncertain_label_count": uncertain_count,
        "interpretation_limits": limits,
    }
    return ScientificReframeTriggerCalibrationReport(
        calibration_id=_calibration_id(body),
        **body,
    )
