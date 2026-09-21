from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.benchmark_audit import (
    ScientificReframeBenchmarkAuditReport,
)
from pipeline_core.discovery.reframing.benchmark_synthesis import (
    ScientificReframeBenchmarkSynthesis,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ValidationCohortRole = Literal["holdout_same_domain", "validation_cross_domain"]
ValidationComparabilityStatus = Literal[
    "comparable_clean",
    "contaminated_overlap",
    "semantic_drift",
    "role_domain_mismatch",
    "multiple_issues",
]


SEMANTICS_FINGERPRINT_PATHS: tuple[str, ...] = (
    "pipeline_core/discovery/reframing/operator_contracts.py",
    "pipeline_core/discovery/reframing/evidence_tension.py",
    "pipeline_core/discovery/reframing/trigger_detection.py",
    "pipeline_core/discovery/reframing/trigger_contracts.py",
    "pipeline_core/discovery/reframing/reframe_prompt.py",
    "pipeline_core/discovery/reframing/reframe_runtime.py",
    "pipeline_core/discovery/reframing/critic_prompt.py",
    "pipeline_core/discovery/reframing/critic_runtime.py",
    "pipeline_core/discovery/reframing/portfolio.py",
)


class ScientificReframeCalibrationFreeze(StrictModel):
    schema_version: Literal[
        "scientific-reframe-calibration-freeze-v1"
    ] = "scientific-reframe-calibration-freeze-v1"

    freeze_id: str = Field(min_length=1)
    calibration_domain_label: str = Field(min_length=1)
    calibration_benchmark_root: str = Field(min_length=1)
    source_audit_id: str = Field(min_length=1)
    source_synthesis_id: str = Field(min_length=1)

    calibration_task_count: int = Field(ge=1)
    task_keys: list[str] = Field(min_length=1)
    case_keys: list[str] = Field(min_length=1)
    task_ids: list[str] = Field(default_factory=list)

    semantics_fingerprint: str = Field(min_length=64, max_length=64)
    semantics_file_sha256: dict[str, str] = Field(min_length=1)

    calibration_set_frozen: Literal[True] = True
    calibration_set_may_inform_tuning: Literal[True] = True
    validation_sets_must_not_inform_tuning_before_evaluation: Literal[True] = True
    holdout_overlap_prohibited: Literal[True] = True
    semantic_drift_invalidates_direct_comparison: Literal[True] = True
    model_configuration_fingerprinted: Literal[False] = False
    direct_generation_quality_comparison_requires_same_model_configuration: Literal[True] = True

    llm_calls_performed: Literal[0] = 0
    scientific_authority: Literal[False] = False
    generalization_claim_established: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze(self) -> "ScientificReframeCalibrationFreeze":
        if self.calibration_task_count != len(self.task_keys):
            raise ValueError("calibration_task_count must equal task_keys length")
        if len(self.task_keys) != len(set(self.task_keys)):
            raise ValueError("calibration task keys must be unique")
        if len(self.case_keys) != len(set(self.case_keys)):
            raise ValueError("calibration case keys must be unique")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("calibration task IDs must be unique")
        if len(self.case_keys) > self.calibration_task_count:
            raise ValueError("case key count cannot exceed task count")
        if any(len(value) != 64 for value in self.semantics_file_sha256.values()):
            raise ValueError("semantic file hashes must be SHA-256 hex strings")
        return self


class ScientificReframeValidationCohortAudit(StrictModel):
    schema_version: Literal[
        "scientific-reframe-validation-cohort-audit-v1"
    ] = "scientific-reframe-validation-cohort-audit-v1"

    cohort_audit_id: str = Field(min_length=1)
    source_freeze_id: str = Field(min_length=1)
    source_validation_audit_id: str = Field(min_length=1)
    cohort_role: ValidationCohortRole
    calibration_domain_label: str = Field(min_length=1)
    validation_domain_label: str = Field(min_length=1)
    validation_benchmark_root: str = Field(min_length=1)

    validation_task_count: int = Field(ge=0)
    task_keys: list[str] = Field(default_factory=list)
    case_keys: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)

    overlapping_task_keys: list[str] = Field(default_factory=list)
    overlapping_case_keys: list[str] = Field(default_factory=list)
    overlapping_task_ids: list[str] = Field(default_factory=list)
    overlap_detected: bool

    frozen_semantics_fingerprint: str = Field(min_length=64, max_length=64)
    current_semantics_fingerprint: str = Field(min_length=64, max_length=64)
    semantics_unchanged: bool
    role_domain_consistent: bool
    comparability_status: ValidationComparabilityStatus
    validation_evaluable_without_contamination: bool
    model_configuration_equivalence_verified: Literal[False] = False
    full_generation_quality_comparison_ready: Literal[False] = False

    calibration_data_reused_for_validation: bool
    threshold_tuning_performed: Literal[False] = False
    prompt_tuning_performed: Literal[False] = False
    automatic_rule_changes_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    scientific_authority: Literal[False] = False
    generalization_claim_established: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeValidationCohortAudit":
        if self.validation_task_count != len(self.task_keys):
            raise ValueError("validation_task_count must equal task_keys length")
        if len(self.task_keys) != len(set(self.task_keys)):
            raise ValueError("validation task keys must be unique")
        if len(self.case_keys) != len(set(self.case_keys)):
            raise ValueError("validation case keys must be unique")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("validation task IDs must be unique")
        return self


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def semantics_fingerprint(repository_root: str | Path) -> tuple[str, dict[str, str]]:
    root = Path(repository_root).resolve()
    file_hashes: dict[str, str] = {}
    for relative in SEMANTICS_FINGERPRINT_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required reframing semantics file is missing: {path}")
        file_hashes[relative] = _sha256_bytes(path.read_bytes())
    payload = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8")), file_hashes


def _stable_id(prefix: str, payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def build_calibration_freeze(
    *,
    audit: ScientificReframeBenchmarkAuditReport,
    synthesis: ScientificReframeBenchmarkSynthesis,
    calibration_domain_label: str,
    repository_root: str | Path,
) -> ScientificReframeCalibrationFreeze:
    domain = calibration_domain_label.strip()
    if not domain:
        raise ValueError("calibration_domain_label must be nonblank")
    if synthesis.source_benchmark_audit_id != audit.audit_id:
        raise ValueError("synthesis must descend from the supplied calibration audit")
    if Path(synthesis.benchmark_root).resolve() != Path(audit.benchmark_root).resolve():
        raise ValueError("audit and synthesis benchmark roots do not match")

    complete = sorted(
        (row for row in audit.task_audits if row.status == "complete"),
        key=lambda row: row.task_key,
    )
    if not complete:
        raise ValueError("calibration freeze requires at least one complete task")

    fingerprint, file_hashes = semantics_fingerprint(repository_root)
    task_keys = [row.task_key for row in complete]
    case_keys = sorted({row.case_key for row in complete})
    task_ids = sorted({row.task_id for row in complete if row.task_id})
    payload = {
        "domain": domain,
        "audit_id": audit.audit_id,
        "synthesis_id": synthesis.synthesis_id,
        "task_keys": task_keys,
        "case_keys": case_keys,
        "task_ids": task_ids,
        "semantics_fingerprint": fingerprint,
    }
    return ScientificReframeCalibrationFreeze(
        freeze_id=_stable_id("scientific_reframe_calibration_freeze", payload),
        calibration_domain_label=domain,
        calibration_benchmark_root=str(Path(audit.benchmark_root).resolve()),
        source_audit_id=audit.audit_id,
        source_synthesis_id=synthesis.synthesis_id,
        calibration_task_count=len(task_keys),
        task_keys=task_keys,
        case_keys=case_keys,
        task_ids=task_ids,
        semantics_fingerprint=fingerprint,
        semantics_file_sha256=file_hashes,
    )


def audit_validation_cohort(
    *,
    freeze: ScientificReframeCalibrationFreeze,
    validation_audit: ScientificReframeBenchmarkAuditReport,
    cohort_role: ValidationCohortRole,
    validation_domain_label: str,
    repository_root: str | Path,
) -> ScientificReframeValidationCohortAudit:
    domain = validation_domain_label.strip()
    if not domain:
        raise ValueError("validation_domain_label must be nonblank")

    complete = sorted(
        (row for row in validation_audit.task_audits if row.status == "complete"),
        key=lambda row: row.task_key,
    )
    task_keys = [row.task_key for row in complete]
    case_keys = sorted({row.case_key for row in complete})
    task_ids = sorted({row.task_id for row in complete if row.task_id})

    overlap_task_keys = sorted(set(task_keys) & set(freeze.task_keys))
    overlap_case_keys = sorted(set(case_keys) & set(freeze.case_keys))
    overlap_task_ids = sorted(set(task_ids) & set(freeze.task_ids))
    overlap = bool(overlap_task_keys or overlap_case_keys or overlap_task_ids)

    current_fingerprint, _ = semantics_fingerprint(repository_root)
    semantics_unchanged = current_fingerprint == freeze.semantics_fingerprint
    if cohort_role == "holdout_same_domain":
        role_domain_consistent = domain == freeze.calibration_domain_label
    else:
        role_domain_consistent = domain != freeze.calibration_domain_label

    issues = sum((overlap, not semantics_unchanged, not role_domain_consistent))
    if issues > 1:
        comparability: ValidationComparabilityStatus = "multiple_issues"
    elif overlap:
        comparability = "contaminated_overlap"
    elif not semantics_unchanged:
        comparability = "semantic_drift"
    elif not role_domain_consistent:
        comparability = "role_domain_mismatch"
    else:
        comparability = "comparable_clean"

    evaluable = comparability == "comparable_clean"
    payload = {
        "freeze_id": freeze.freeze_id,
        "validation_audit_id": validation_audit.audit_id,
        "cohort_role": cohort_role,
        "validation_domain": domain,
        "task_keys": task_keys,
        "current_fingerprint": current_fingerprint,
        "comparability": comparability,
    }
    return ScientificReframeValidationCohortAudit(
        cohort_audit_id=_stable_id("scientific_reframe_validation_cohort", payload),
        source_freeze_id=freeze.freeze_id,
        source_validation_audit_id=validation_audit.audit_id,
        cohort_role=cohort_role,
        calibration_domain_label=freeze.calibration_domain_label,
        validation_domain_label=domain,
        validation_benchmark_root=str(Path(validation_audit.benchmark_root).resolve()),
        validation_task_count=len(task_keys),
        task_keys=task_keys,
        case_keys=case_keys,
        task_ids=task_ids,
        overlapping_task_keys=overlap_task_keys,
        overlapping_case_keys=overlap_case_keys,
        overlapping_task_ids=overlap_task_ids,
        overlap_detected=overlap,
        frozen_semantics_fingerprint=freeze.semantics_fingerprint,
        current_semantics_fingerprint=current_fingerprint,
        semantics_unchanged=semantics_unchanged,
        role_domain_consistent=role_domain_consistent,
        comparability_status=comparability,
        validation_evaluable_without_contamination=evaluable,
        calibration_data_reused_for_validation=overlap,
    )
