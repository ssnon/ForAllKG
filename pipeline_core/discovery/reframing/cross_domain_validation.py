from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.benchmark_audit import (
    ScientificReframeBenchmarkAuditReport,
)
from pipeline_core.discovery.reframing.generalization_probe import (
    ScientificReframeGeneralizationPreflight,
    ValidationTaskInventoryRow,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ValidationScale = Literal[
    "single_task_smoke_test",
    "multi_task_descriptive_validation",
]


class CrossDomainTaskSelection(StrictModel):
    task_key: str = Field(min_length=1)
    case_key: str = Field(min_length=1)
    replicate_key: str = Field(min_length=1)
    canonical_dir: str = Field(min_length=1)
    inferred_benchmark_root: str = Field(min_length=1)
    task_id: str | None = None
    canonical_domain_label: str = Field(min_length=1)


class ScientificReframeCrossDomainValidationReport(StrictModel):
    schema_version: Literal[
        "scientific-reframe-cross-domain-validation-v1"
    ] = "scientific-reframe-cross-domain-validation-v1"

    validation_id: str = Field(min_length=1)
    source_freeze_id: str = Field(min_length=1)
    source_preflight_id: str = Field(min_length=1)
    source_audit_id: str = Field(min_length=1)

    validation_role: Literal["validation_cross_domain"] = "validation_cross_domain"
    calibration_domain_label: str = Field(min_length=1)
    validation_domain_label: str = Field(min_length=1)
    search_root: str = Field(min_length=1)
    extraction_roots: list[str] = Field(min_length=1)

    frozen_semantics_fingerprint: str = Field(min_length=64, max_length=64)
    current_semantics_fingerprint: str = Field(min_length=64, max_length=64)
    semantics_unchanged: Literal[True] = True

    selected_tasks: list[CrossDomainTaskSelection] = Field(min_length=1)
    selected_task_count: int = Field(ge=1)
    complete_task_count: int = Field(ge=0)
    error_task_count: int = Field(ge=0)
    validation_scale: ValidationScale

    trigger_pattern_counts: dict[str, int] = Field(default_factory=dict)
    triggered_task_count: int = Field(ge=0)
    tension_type_task_counts: dict[str, int] = Field(default_factory=dict)
    direct_trigger_signal_task_counts: dict[str, int] = Field(default_factory=dict)
    readiness_trigger_alignment_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )

    source_domain_marker_hit_count: int = Field(ge=0)
    source_domain_marker_hit_counts: dict[str, int] = Field(default_factory=dict)

    audit: ScientificReframeBenchmarkAuditReport

    llm_calls_performed: Literal[0] = 0
    generation_performed: Literal[False] = False
    critic_performed: Literal[False] = False
    portfolio_performed: Literal[False] = False
    threshold_tuning_performed: Literal[False] = False
    prompt_tuning_performed: Literal[False] = False
    scientific_authority: Literal[False] = False
    cross_domain_accuracy_evaluated: Literal[False] = False
    trigger_quality_evaluated: Literal[False] = False
    source_marker_causality_evaluated: Literal[False] = False
    generalization_claim_established: Literal[False] = False
    calibration_overlap_detected: Literal[False] = False
    validation_is_descriptive_only: Literal[True] = True
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeCrossDomainValidationReport":
        if self.selected_task_count != len(self.selected_tasks):
            raise ValueError("selected_task_count must equal selected task rows")
        if self.complete_task_count + self.error_task_count != self.selected_task_count:
            raise ValueError("complete/error counts must equal selected task count")
        if self.audit.discovered_task_count != self.selected_task_count:
            raise ValueError("embedded audit task count must match selected task count")
        if self.audit.complete_task_count != self.complete_task_count:
            raise ValueError("embedded audit complete count mismatch")
        if self.audit.error_task_count != self.error_task_count:
            raise ValueError("embedded audit error count mismatch")
        expected_scale = (
            "single_task_smoke_test"
            if self.selected_task_count == 1
            else "multi_task_descriptive_validation"
        )
        if self.validation_scale != expected_scale:
            raise ValueError("validation_scale does not match selected task count")
        if self.calibration_domain_label == self.validation_domain_label:
            raise ValueError("cross-domain validation must use a different domain")
        return self


def _stable_id(prefix: str, payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def select_cross_domain_tasks(
    preflight: ScientificReframeGeneralizationPreflight,
    *,
    validation_domain_label: str,
) -> list[CrossDomainTaskSelection]:
    domain = validation_domain_label.strip().casefold().replace("-", "_")
    if not domain:
        raise ValueError("validation_domain_label is required")
    calibration = preflight.calibration_domain_label.strip().casefold().replace("-", "_")
    if domain == calibration:
        raise ValueError("validation domain must differ from calibration domain")
    if not preflight.semantics_unchanged:
        raise ValueError("frozen semantics changed; cross-domain comparison is not valid")

    cohort = next(
        (
            row
            for row in preflight.cross_domain_candidate_cohorts
            if row.canonical_domain_label == domain
        ),
        None,
    )
    if cohort is None:
        raise ValueError(f"no cross-domain candidate cohort for domain: {domain}")

    by_canonical = {row.canonical_dir: row for row in preflight.task_rows}
    selected: list[CrossDomainTaskSelection] = []
    for canonical_dir in cohort.canonical_dirs:
        row = by_canonical.get(canonical_dir)
        if row is None:
            raise ValueError(f"cohort canonical directory is absent from task inventory: {canonical_dir}")
        if not row.complete_triplet:
            raise ValueError(f"cross-domain cohort contains incomplete task: {row.task_key}")
        if row.any_frozen_overlap:
            raise ValueError(f"cross-domain cohort overlaps calibration: {row.task_key}")
        if row.canonical_domain_labels != [domain]:
            raise ValueError(
                f"cross-domain task does not have exactly one expected canonical domain: {row.task_key}"
            )
        selected.append(
            CrossDomainTaskSelection(
                task_key=row.task_key,
                case_key=row.case_key,
                replicate_key=row.replicate_key,
                canonical_dir=row.canonical_dir,
                inferred_benchmark_root=row.inferred_benchmark_root,
                task_id=row.task_id,
                canonical_domain_label=domain,
            )
        )

    if len(selected) != cohort.task_count:
        raise ValueError("selected cross-domain task count does not match cohort")
    if not selected:
        raise ValueError("cross-domain cohort is empty")
    return sorted(selected, key=lambda row: row.task_key)


def build_cross_domain_validation_report(
    *,
    preflight: ScientificReframeGeneralizationPreflight,
    validation_domain_label: str,
    extraction_roots: list[str],
    selected_tasks: list[CrossDomainTaskSelection],
    audit: ScientificReframeBenchmarkAuditReport,
) -> ScientificReframeCrossDomainValidationReport:
    if not preflight.semantics_unchanged:
        raise ValueError("frozen semantics changed; validation report cannot be built")
    if audit.llm_calls_performed != 0:
        raise ValueError("cross-domain trigger validation must remain zero-LLM")

    selected_keys = {row.task_key for row in selected_tasks}
    audit_keys = {row.task_key for row in audit.task_audits}
    if selected_keys != audit_keys:
        raise ValueError("embedded audit task identities do not match selected cohort")

    trigger_counts = Counter(
        row.trigger_pattern
        for row in audit.task_audits
        if row.status == "complete"
    )
    triggered = sum(
        row.status == "complete" and row.trigger_pattern not in {"none", "unavailable"}
        for row in audit.task_audits
    )
    payload = {
        "freeze": preflight.source_freeze_id,
        "preflight": preflight.preflight_id,
        "audit": audit.audit_id,
        "domain": validation_domain_label,
        "tasks": [row.model_dump(mode="json") for row in selected_tasks],
    }

    return ScientificReframeCrossDomainValidationReport(
        validation_id=_stable_id("scientific_reframe_cross_domain_validation", payload),
        source_freeze_id=preflight.source_freeze_id,
        source_preflight_id=preflight.preflight_id,
        source_audit_id=audit.audit_id,
        calibration_domain_label=preflight.calibration_domain_label,
        validation_domain_label=validation_domain_label,
        search_root=preflight.search_root,
        extraction_roots=list(extraction_roots),
        frozen_semantics_fingerprint=preflight.frozen_semantics_fingerprint,
        current_semantics_fingerprint=preflight.current_semantics_fingerprint,
        selected_tasks=selected_tasks,
        selected_task_count=len(selected_tasks),
        complete_task_count=audit.complete_task_count,
        error_task_count=audit.error_task_count,
        validation_scale=(
            "single_task_smoke_test"
            if len(selected_tasks) == 1
            else "multi_task_descriptive_validation"
        ),
        trigger_pattern_counts=dict(sorted(trigger_counts.items())),
        triggered_task_count=triggered,
        tension_type_task_counts=dict(audit.tension_type_task_counts),
        direct_trigger_signal_task_counts=dict(audit.direct_trigger_signal_task_counts),
        readiness_trigger_alignment_counts=dict(audit.readiness_trigger_alignment_counts),
        source_domain_marker_hit_count=preflight.source_domain_marker_hit_count,
        source_domain_marker_hit_counts=dict(preflight.marker_hit_counts),
        audit=audit,
    )
