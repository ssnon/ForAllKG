from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.response_semantics_probe import (
    ScientificResponseSemanticsProbeReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


CANDIDATE_RESPONSE_SEMANTICS_PATHS: tuple[str, ...] = (
    "pipeline_core/discovery/reframing/response_semantics_probe.py",
)


class ScientificResponseSemanticsCandidateFreeze(StrictModel):
    schema_version: Literal[
        "scientific-response-semantics-candidate-freeze-v1"
    ] = "scientific-response-semantics-candidate-freeze-v1"
    freeze_id: str
    source_probe_id: str
    source_probe_sha256: str
    source_diagnostics_pack_id: str
    adaptation_task_ids: list[str] = Field(default_factory=list)
    adaptation_domain_labels: list[str] = Field(default_factory=list)
    semantics_profile_ids: list[str] = Field(default_factory=list)
    candidate_semantics_fingerprint: str
    candidate_semantics_file_sha256: dict[str, str] = Field(default_factory=dict)

    candidate_profile_frozen: Literal[True] = True
    adaptation_data_excluded_from_untouched_validation: Literal[True] = True
    frozen_v1_trigger_semantics_modified: Literal[False] = False
    production_trigger_semantics_modified: Literal[False] = False
    actual_trigger_authority: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    scientific_authority: Literal[False] = False
    generalization_claim_established: Literal[False] = False

    @model_validator(mode="after")
    def validate_unique_sets(self) -> "ScientificResponseSemanticsCandidateFreeze":
        if len(self.adaptation_task_ids) != len(set(self.adaptation_task_ids)):
            raise ValueError("adaptation_task_ids must be unique")
        if len(self.adaptation_domain_labels) != len(set(self.adaptation_domain_labels)):
            raise ValueError("adaptation_domain_labels must be unique")
        if len(self.semantics_profile_ids) != len(set(self.semantics_profile_ids)):
            raise ValueError("semantics_profile_ids must be unique")
        if not self.adaptation_task_ids:
            raise ValueError("candidate freeze requires at least one adaptation task")
        return self


ValidationEligibilityStatus = Literal[
    "eligible_untouched",
    "adaptation_overlap",
    "candidate_semantic_drift",
    "multiple_issues",
]


class ScientificResponseSemanticsValidationEligibility(StrictModel):
    schema_version: Literal[
        "scientific-response-semantics-validation-eligibility-v1"
    ] = "scientific-response-semantics-validation-eligibility-v1"
    eligibility_id: str
    source_candidate_freeze_id: str
    validation_task_id: str
    validation_domain_label: str
    adaptation_overlap: bool
    candidate_semantics_unchanged: bool
    frozen_candidate_semantics_fingerprint: str
    current_candidate_semantics_fingerprint: str
    status: ValidationEligibilityStatus
    eligible_as_untouched_validation: bool

    threshold_tuning_performed: Literal[False] = False
    prompt_tuning_performed: Literal[False] = False
    llm_calls_performed: Literal[0] = 0
    scientific_authority: Literal[False] = False
    generalization_claim_established: Literal[False] = False


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_id(prefix: str, payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{prefix}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


def candidate_semantics_fingerprint(
    repository_root: str | Path,
) -> tuple[str, dict[str, str]]:
    root = Path(repository_root).resolve()
    file_hashes: dict[str, str] = {}
    for relative in CANDIDATE_RESPONSE_SEMANTICS_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"required response-semantics candidate file is missing: {path}"
            )
        file_hashes[relative] = _sha256_bytes(path.read_bytes())
    payload = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8")), file_hashes


def build_response_semantics_candidate_freeze(
    *,
    probe: ScientificResponseSemanticsProbeReport,
    source_probe_path: str | Path,
    repository_root: str | Path,
) -> ScientificResponseSemanticsCandidateFreeze:
    if not probe.adaptation_probe:
        raise ValueError("candidate freeze requires an adaptation probe")
    if probe.eligible_as_untouched_validation_after_design_use:
        raise ValueError("adaptation probe must not be eligible as untouched validation")
    if probe.actual_trigger_decision_changed:
        raise ValueError("candidate freeze requires a diagnostic-only probe")

    source_path = Path(source_probe_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    fingerprint, file_hashes = candidate_semantics_fingerprint(repository_root)
    source_sha256 = _sha256_bytes(source_path.read_bytes())
    adaptation_task_ids = [probe.source_task_id]
    domains = [probe.validation_domain_label]
    profile_ids = [probe.semantics_profile_id]
    payload = {
        "source_probe_id": probe.probe_id,
        "source_probe_sha256": source_sha256,
        "adaptation_task_ids": adaptation_task_ids,
        "domains": domains,
        "profiles": profile_ids,
        "candidate_semantics_fingerprint": fingerprint,
    }
    return ScientificResponseSemanticsCandidateFreeze(
        freeze_id=_stable_id("scientific_response_semantics_candidate_freeze", payload),
        source_probe_id=probe.probe_id,
        source_probe_sha256=source_sha256,
        source_diagnostics_pack_id=probe.source_diagnostics_pack_id,
        adaptation_task_ids=adaptation_task_ids,
        adaptation_domain_labels=domains,
        semantics_profile_ids=profile_ids,
        candidate_semantics_fingerprint=fingerprint,
        candidate_semantics_file_sha256=file_hashes,
    )


def inspect_response_semantics_validation_eligibility(
    *,
    freeze: ScientificResponseSemanticsCandidateFreeze,
    validation_task_id: str,
    validation_domain_label: str,
    repository_root: str | Path,
) -> ScientificResponseSemanticsValidationEligibility:
    task_id = validation_task_id.strip()
    domain = validation_domain_label.strip()
    if not task_id:
        raise ValueError("validation_task_id must be nonblank")
    if not domain:
        raise ValueError("validation_domain_label must be nonblank")

    overlap = task_id in set(freeze.adaptation_task_ids)
    current, _ = candidate_semantics_fingerprint(repository_root)
    unchanged = current == freeze.candidate_semantics_fingerprint
    issues = sum((overlap, not unchanged))
    if issues > 1:
        status: ValidationEligibilityStatus = "multiple_issues"
    elif overlap:
        status = "adaptation_overlap"
    elif not unchanged:
        status = "candidate_semantic_drift"
    else:
        status = "eligible_untouched"
    eligible = status == "eligible_untouched"
    payload = {
        "freeze_id": freeze.freeze_id,
        "task_id": task_id,
        "domain": domain,
        "current": current,
        "status": status,
    }
    return ScientificResponseSemanticsValidationEligibility(
        eligibility_id=_stable_id("scientific_response_semantics_validation_eligibility", payload),
        source_candidate_freeze_id=freeze.freeze_id,
        validation_task_id=task_id,
        validation_domain_label=domain,
        adaptation_overlap=overlap,
        candidate_semantics_unchanged=unchanged,
        frozen_candidate_semantics_fingerprint=freeze.candidate_semantics_fingerprint,
        current_candidate_semantics_fingerprint=current,
        status=status,
        eligible_as_untouched_validation=eligible,
    )
