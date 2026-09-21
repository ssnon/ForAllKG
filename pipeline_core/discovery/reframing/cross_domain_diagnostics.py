from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.evidence_tension import (
    ScientificEvidenceTensionReport,
    _BOUNDARY_PATTERNS,
    _DOWN_PATTERNS,
    _MAGNITUDE_PATTERNS,
    _NULL_PATTERNS,
    _RESPONSE_FAMILY_PATTERNS,
    _UP_PATTERNS,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ScientificReframeTriggerReport,
)
from pipeline_core.discovery.reframing.trigger_detection import (
    _CONTRAST_CUE,
    _DOWN_CUE,
    _FINITE_OPTIMUM_CUE,
    _MECHANISM_SWITCH_CUE,
    _PARAMETER_CUE,
    _RESPONSE_CUE,
    _SATURATION_CUE,
    _STRONG_MEDIATION_GAP_PATTERNS,
    _TENSION_MAP,
    _THRESHOLD_CUE,
    _UP_CUE,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


StatementRole = Literal["premise", "gap"]


class RegexCueTrace(StrictModel):
    matched: bool
    matches: list[str] = Field(default_factory=list)


class StatementTriggerCueTrace(StrictModel):
    statement_id: str
    role: StatementRole
    text: str
    epistemic_role: str
    claim_kind: str
    paper_ids: list[str] = Field(default_factory=list)

    frozen_trigger_response_cue: RegexCueTrace
    evidence_response_families: list[str] = Field(default_factory=list)
    evidence_up_cue: RegexCueTrace
    evidence_down_cue: RegexCueTrace
    evidence_null_cue: RegexCueTrace
    evidence_boundary_cue: RegexCueTrace
    evidence_magnitude_cue: RegexCueTrace

    direct_up_cue: RegexCueTrace
    direct_down_cue: RegexCueTrace
    direct_contrast_cue: RegexCueTrace
    direct_finite_optimum_cue: RegexCueTrace
    direct_parameter_cue: RegexCueTrace
    direct_saturation_cue: RegexCueTrace
    direct_threshold_cue: RegexCueTrace
    direct_mechanism_switch_cue: RegexCueTrace
    strong_mediation_gap_cue: RegexCueTrace

    eligible_latent_mediation_support_premise: bool = False
    eligible_regime_direct_scan: bool = False
    potential_direct_signal_kinds: list[str] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    scientific_response_authority: Literal[False] = False


class ExplorerTensionTrace(StrictModel):
    source_tension_id: str
    source_tension_type: str
    referenced_statement_ids: list[str] = Field(default_factory=list)
    grounded_statement_ids: list[str] = Field(default_factory=list)
    grounded_reference_present: bool = False
    trigger_source_type_mapped: bool = False
    normalized_trigger_type: str | None = None
    evidence_witness_ids: list[str] = Field(default_factory=list)
    trigger_witness_ids: list[str] = Field(default_factory=list)

    diagnostic_only: Literal[True] = True
    scientific_conflict_authority: Literal[False] = False


class CrossDomainTriggerDiagnosticSummary(StrictModel):
    premise_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    premise_response_cue_match_count: int = Field(ge=0)
    premise_response_family_match_count: int = Field(ge=0)
    gap_strong_mediation_cue_count: int = Field(ge=0)
    potential_regime_direct_signal_statement_count: int = Field(ge=0)

    explorer_raw_tension_count: int = Field(ge=0)
    grounded_evidence_tension_witness_count: int = Field(ge=0)
    trigger_tension_witness_count: int = Field(ge=0)
    actual_direct_trigger_signal_count: int = Field(ge=0)

    diagnostic_flags: list[str] = Field(default_factory=list)

    trigger_failure_cause_determined: Literal[False] = False
    source_marker_causality_evaluated: Literal[False] = False
    generalization_quality_evaluated: Literal[False] = False


class CrossDomainTaskTriggerDiagnostic(StrictModel):
    task_key: str
    case_key: str
    canonical_dir: str
    task_id: str
    question: str
    trigger_pattern: str
    statement_traces: list[StatementTriggerCueTrace]
    explorer_tension_traces: list[ExplorerTensionTrace] = Field(default_factory=list)
    actual_trigger_assessments: list[dict[str, Any]] = Field(default_factory=list)
    actual_direct_trigger_signals: list[dict[str, Any]] = Field(default_factory=list)
    summary: CrossDomainTriggerDiagnosticSummary


class ScientificReframeCrossDomainTriggerDiagnosticPack(StrictModel):
    schema_version: Literal[
        "scientific-reframe-cross-domain-trigger-diagnostics-v1"
    ] = "scientific-reframe-cross-domain-trigger-diagnostics-v1"
    pack_id: str
    source_validation_id: str
    validation_domain_label: str
    frozen_semantics_fingerprint: str = Field(min_length=64, max_length=64)
    current_semantics_fingerprint: str = Field(min_length=64, max_length=64)
    semantics_unchanged: Literal[True] = True
    tasks: list[CrossDomainTaskTriggerDiagnostic] = Field(min_length=1)
    task_count: int = Field(ge=1)

    llm_calls_performed: Literal[0] = 0
    deterministic_diagnostics: Literal[True] = True
    trigger_semantics_modified: Literal[False] = False
    prompt_semantics_modified: Literal[False] = False
    threshold_tuning_performed: Literal[False] = False
    source_marker_causality_evaluated: Literal[False] = False
    trigger_quality_evaluated: Literal[False] = False
    generalization_claim_established: Literal[False] = False
    scientific_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeCrossDomainTriggerDiagnosticPack":
        if self.task_count != len(self.tasks):
            raise ValueError("task_count must equal task diagnostics length")
        keys = [row.task_key for row in self.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("task diagnostic keys must be unique")
        return self


def _regex_matches(text: str, patterns: tuple[re.Pattern[str], ...] | re.Pattern[str]) -> RegexCueTrace:
    rows = patterns if isinstance(patterns, tuple) else (patterns,)
    matches: list[str] = []
    seen: set[str] = set()
    for pattern in rows:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                matches.append(value)
    return RegexCueTrace(matched=bool(matches), matches=matches)


def _response_families(text: str) -> list[str]:
    return sorted(
        family
        for family, patterns in _RESPONSE_FAMILY_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    )


def _trace_statement(row: ReframeEvidenceStatement, *, role: StatementRole) -> StatementTriggerCueTrace:
    text = row.text
    response = _regex_matches(text, _RESPONSE_CUE)
    direct_up = _regex_matches(text, _UP_CUE)
    direct_down = _regex_matches(text, _DOWN_CUE)
    direct_contrast = _regex_matches(text, _CONTRAST_CUE)
    optimum = _regex_matches(text, _FINITE_OPTIMUM_CUE)
    parameter = _regex_matches(text, _PARAMETER_CUE)
    saturation = _regex_matches(text, _SATURATION_CUE)
    threshold = _regex_matches(text, _THRESHOLD_CUE)
    mechanism_switch = _regex_matches(text, _MECHANISM_SWITCH_CUE)
    mediation = _regex_matches(text, _STRONG_MEDIATION_GAP_PATTERNS)

    potential: list[str] = []
    if role == "premise" and response.matched:
        if direct_up.matched and direct_down.matched and direct_contrast.matched:
            potential.append("NON_MONOTONIC_RESPONSE")
        if saturation.matched:
            potential.append("SATURATION_OR_PLATEAU")
        if threshold.matched:
            potential.append("EXPLICIT_THRESHOLD")
        if mechanism_switch.matched:
            potential.append("MECHANISM_SWITCH")
        if optimum.matched and parameter.matched:
            potential.append("FINITE_OPTIMUM")

    return StatementTriggerCueTrace(
        statement_id=row.statement_id,
        role=role,
        text=text,
        epistemic_role=row.epistemic_role,
        claim_kind=row.claim_kind,
        paper_ids=sorted(set(row.paper_ids)),
        frozen_trigger_response_cue=response,
        evidence_response_families=_response_families(text),
        evidence_up_cue=_regex_matches(text, _UP_PATTERNS),
        evidence_down_cue=_regex_matches(text, _DOWN_PATTERNS),
        evidence_null_cue=_regex_matches(text, _NULL_PATTERNS),
        evidence_boundary_cue=_regex_matches(text, _BOUNDARY_PATTERNS),
        evidence_magnitude_cue=_regex_matches(text, _MAGNITUDE_PATTERNS),
        direct_up_cue=direct_up,
        direct_down_cue=direct_down,
        direct_contrast_cue=direct_contrast,
        direct_finite_optimum_cue=optimum,
        direct_parameter_cue=parameter,
        direct_saturation_cue=saturation,
        direct_threshold_cue=threshold,
        direct_mechanism_switch_cue=mechanism_switch,
        strong_mediation_gap_cue=mediation,
        eligible_latent_mediation_support_premise=(
            role == "premise"
            and row.claim_kind in {"mechanism", "association", "observation", "comparison"}
            and response.matched
        ),
        eligible_regime_direct_scan=(role == "premise" and response.matched),
        potential_direct_signal_kinds=potential,
    )


def _explorer_tension_traces(
    *,
    explorer_report: Mapping[str, Any],
    evidence: ScientificReframeEvidencePacket,
    tensions: ScientificEvidenceTensionReport,
    trigger: ScientificReframeTriggerReport,
) -> list[ExplorerTensionTrace]:
    grounded_ids = {
        row.statement_id
        for row in [*evidence.premise_statements, *evidence.gap_statements]
    }
    evidence_by_source: dict[str, list[str]] = {}
    for witness in tensions.witnesses:
        evidence_by_source.setdefault(witness.source_tension_id, []).append(witness.witness_id)
    trigger_by_source: dict[str, list[str]] = {}
    evidence_witness_by_id = {row.witness_id: row for row in tensions.witnesses}
    for witness in trigger.tension_witnesses:
        source = evidence_witness_by_id.get(witness.witness_id)
        if source is not None:
            trigger_by_source.setdefault(source.source_tension_id, []).append(witness.witness_id)

    raw_rows = explorer_report.get("evidence_tensions", [])
    if not isinstance(raw_rows, list):
        raw_rows = []
    rows: list[ExplorerTensionTrace] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            continue
        source_id = str(raw.get("tension_id", "")).strip() or f"tension:{index}"
        source_type = str(raw.get("tension_type", "")).strip()
        referenced: set[str] = set()
        for key in ("side_a_statement_ids", "side_b_statement_ids"):
            values = raw.get(key, [])
            if isinstance(values, list):
                referenced.update(str(value).strip() for value in values if str(value).strip())
        focal = str(raw.get("statement_id", "")).strip()
        if focal:
            referenced.add(focal)
        grounded = sorted(referenced & grounded_ids)
        normalized = _TENSION_MAP.get(source_type)
        rows.append(
            ExplorerTensionTrace(
                source_tension_id=source_id,
                source_tension_type=source_type,
                referenced_statement_ids=sorted(referenced),
                grounded_statement_ids=grounded,
                grounded_reference_present=bool(grounded),
                trigger_source_type_mapped=normalized is not None,
                normalized_trigger_type=normalized,
                evidence_witness_ids=sorted(evidence_by_source.get(source_id, [])),
                trigger_witness_ids=sorted(trigger_by_source.get(source_id, [])),
            )
        )
    return rows


def build_cross_domain_task_trigger_diagnostic(
    *,
    task_key: str,
    case_key: str,
    canonical_dir: str,
    trigger_pattern: str,
    evidence: ScientificReframeEvidencePacket,
    tensions: ScientificEvidenceTensionReport,
    trigger: ScientificReframeTriggerReport,
    explorer_report: Mapping[str, Any],
) -> CrossDomainTaskTriggerDiagnostic:
    if tensions.source_task_id != evidence.task_id:
        raise ValueError("tension report task does not match evidence packet")
    if trigger.source_task_id != evidence.task_id:
        raise ValueError("trigger report task does not match evidence packet")
    if trigger.source_evidence_tension_report_id != tensions.report_id:
        raise ValueError("trigger report does not reference supplied tension report")

    statement_traces = [
        *(_trace_statement(row, role="premise") for row in evidence.premise_statements),
        *(_trace_statement(row, role="gap") for row in evidence.gap_statements),
    ]
    explorer_traces = _explorer_tension_traces(
        explorer_report=explorer_report,
        evidence=evidence,
        tensions=tensions,
        trigger=trigger,
    )

    premise_traces = [row for row in statement_traces if row.role == "premise"]
    gap_traces = [row for row in statement_traces if row.role == "gap"]
    raw_tensions = explorer_report.get("evidence_tensions", [])
    raw_tension_count = len(raw_tensions) if isinstance(raw_tensions, list) else 0

    flags: list[str] = []
    if raw_tension_count == 0:
        flags.append("NO_UPSTREAM_EXPLORER_TENSIONS")
    if raw_tension_count > 0 and not tensions.witnesses:
        flags.append("EXPLORER_TENSIONS_DROPPED_BEFORE_GROUNDED_WITNESS")
    if tensions.witnesses and not trigger.tension_witnesses:
        flags.append("GROUNDED_TENSIONS_NOT_MAPPED_INTO_TRIGGER_WITNESSES")
    if not any(row.frozen_trigger_response_cue.matched for row in premise_traces):
        flags.append("NO_PREMISE_MATCHES_FROZEN_TRIGGER_RESPONSE_CUE")
    elif any(not row.frozen_trigger_response_cue.matched for row in premise_traces):
        flags.append("PARTIAL_PREMISE_RESPONSE_CUE_COVERAGE")
    if any(row.strong_mediation_gap_cue.matched for row in gap_traces) and not trigger.direct_trigger_signals:
        flags.append("MEDIATION_GAP_CUE_PRESENT_WITHOUT_ACTUAL_DIRECT_SIGNAL")
    if any(row.potential_direct_signal_kinds for row in premise_traces) and not trigger.direct_trigger_signals:
        flags.append("STATEMENT_LEVEL_REGIME_CUE_PRESENT_WITHOUT_ACTUAL_DIRECT_SIGNAL")
    if not trigger.direct_trigger_signals:
        flags.append("NO_ACTUAL_DIRECT_TRIGGER_SIGNALS")

    summary = CrossDomainTriggerDiagnosticSummary(
        premise_count=len(evidence.premise_statements),
        gap_count=len(evidence.gap_statements),
        premise_response_cue_match_count=sum(
            row.frozen_trigger_response_cue.matched for row in premise_traces
        ),
        premise_response_family_match_count=sum(
            bool(row.evidence_response_families) for row in premise_traces
        ),
        gap_strong_mediation_cue_count=sum(
            row.strong_mediation_gap_cue.matched for row in gap_traces
        ),
        potential_regime_direct_signal_statement_count=sum(
            bool(row.potential_direct_signal_kinds) for row in premise_traces
        ),
        explorer_raw_tension_count=raw_tension_count,
        grounded_evidence_tension_witness_count=len(tensions.witnesses),
        trigger_tension_witness_count=len(trigger.tension_witnesses),
        actual_direct_trigger_signal_count=len(trigger.direct_trigger_signals),
        diagnostic_flags=flags,
    )
    return CrossDomainTaskTriggerDiagnostic(
        task_key=task_key,
        case_key=case_key,
        canonical_dir=canonical_dir,
        task_id=evidence.task_id,
        question=evidence.question,
        trigger_pattern=trigger_pattern,
        statement_traces=statement_traces,
        explorer_tension_traces=explorer_traces,
        actual_trigger_assessments=[row.model_dump(mode="json") for row in trigger.assessments],
        actual_direct_trigger_signals=[
            row.model_dump(mode="json") for row in trigger.direct_trigger_signals
        ],
        summary=summary,
    )


def build_cross_domain_trigger_diagnostic_pack(
    *,
    source_validation_id: str,
    validation_domain_label: str,
    frozen_semantics_fingerprint: str,
    current_semantics_fingerprint: str,
    tasks: list[CrossDomainTaskTriggerDiagnostic],
) -> ScientificReframeCrossDomainTriggerDiagnosticPack:
    if frozen_semantics_fingerprint != current_semantics_fingerprint:
        raise ValueError("frozen semantics changed; diagnostic comparison is not valid")
    payload = {
        "source_validation_id": source_validation_id,
        "validation_domain_label": validation_domain_label,
        "semantics_fingerprint": current_semantics_fingerprint,
        "tasks": [row.model_dump(mode="json") for row in sorted(tasks, key=lambda item: item.task_key)],
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    sorted_tasks = sorted(tasks, key=lambda item: item.task_key)
    return ScientificReframeCrossDomainTriggerDiagnosticPack(
        pack_id=f"scientific_reframe_cross_domain_trigger_diagnostics:{digest}",
        source_validation_id=source_validation_id,
        validation_domain_label=validation_domain_label,
        frozen_semantics_fingerprint=frozen_semantics_fingerprint,
        current_semantics_fingerprint=current_semantics_fingerprint,
        tasks=sorted_tasks,
        task_count=len(sorted_tasks),
    )
