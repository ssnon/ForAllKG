from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


TriggerPattern = Literal[
    "none",
    "latent_only",
    "regime_only",
    "latent_and_regime",
    "unavailable",
]

ReadinessTriggerAlignment = Literal[
    "ready_and_triggered",
    "ready_not_triggered",
    "triggered_not_ready",
    "not_ready_not_triggered",
    "no_trigger_contract",
    "unavailable",
]

TaskAuditStatus = Literal["complete", "error"]

_READY_WITHOUT_MANDATORY = {
    "ready_now",
    "ready_with_optional_enrichment",
}


class BenchmarkTaskReframeAudit(StrictModel):
    schema_version: Literal[
        "scientific-reframe-benchmark-task-audit-v1"
    ] = "scientific-reframe-benchmark-task-audit-v1"

    task_key: str = Field(min_length=1)
    case_key: str = Field(min_length=1)
    replicate_key: str = Field(min_length=1)
    canonical_dir: str = Field(min_length=1)
    status: TaskAuditStatus
    task_id: str | None = None

    error_type: str | None = None
    error_message: str | None = None

    selected_premise_count: int = Field(ge=0, default=0)
    selected_gap_count: int = Field(ge=0, default=0)
    grounded_source_chunk_count: int = Field(ge=0, default=0)
    grounded_semantic_record_count: int = Field(ge=0, default=0)
    unresolved_grounded_node_count: int = Field(ge=0, default=0)
    ambiguous_grounded_node_count: int = Field(ge=0, default=0)

    tension_witness_count: int = Field(ge=0, default=0)
    tension_type_counts: dict[str, int] = Field(default_factory=dict)
    direct_trigger_signal_kind_counts: dict[str, int] = Field(default_factory=dict)
    condition_example_count: int = Field(ge=0, default=0)
    condition_signature_count: int = Field(ge=0, default=0)
    condition_paper_count: int = Field(ge=0, default=0)

    readiness_status_by_operator: dict[str, str] = Field(default_factory=dict)
    mandatory_backfill_count_by_operator: dict[str, int] = Field(default_factory=dict)
    optional_backfill_count_by_operator: dict[str, int] = Field(default_factory=dict)
    trigger_decision_by_operator: dict[str, str] = Field(default_factory=dict)
    trigger_signal_by_operator: dict[str, bool] = Field(default_factory=dict)
    readiness_trigger_alignment_by_operator: dict[
        str, ReadinessTriggerAlignment
    ] = Field(default_factory=dict)

    trigger_pattern: TriggerPattern = "unavailable"
    measurement_tension_present: bool = False
    proxy_enrichment_candidate: bool = False
    proxy_candidate_reason: str | None = None

    llm_calls_performed: Literal[0] = 0
    shadow_only: Literal[True] = True
    scientific_conflict_authority: Literal[False] = False
    regime_boundary_authority: Literal[False] = False
    proxy_semantics_authority: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    candidate_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False

    @model_validator(mode="after")
    def validate_error_state(self) -> "BenchmarkTaskReframeAudit":
        if self.status == "error":
            if not (self.error_type or "").strip():
                raise ValueError("error task audits require error_type")
            if self.trigger_pattern != "unavailable":
                raise ValueError("error task audits must use unavailable trigger_pattern")
        else:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("complete task audits cannot contain error fields")
            if self.trigger_pattern == "unavailable":
                raise ValueError("complete task audits require a resolved trigger_pattern")
        return self


class ScientificReframeBenchmarkAuditReport(StrictModel):
    schema_version: Literal[
        "scientific-reframe-benchmark-audit-v1"
    ] = "scientific-reframe-benchmark-audit-v1"

    audit_id: str = Field(min_length=1)
    benchmark_root: str = Field(min_length=1)
    extraction_roots: list[str] = Field(min_length=1)
    task_audits: list[BenchmarkTaskReframeAudit]

    discovered_task_count: int = Field(ge=0)
    complete_task_count: int = Field(ge=0)
    error_task_count: int = Field(ge=0)
    unique_case_count: int = Field(ge=0)

    trigger_pattern_counts: dict[str, int] = Field(default_factory=dict)
    tension_type_task_counts: dict[str, int] = Field(default_factory=dict)
    tension_type_witness_counts: dict[str, int] = Field(default_factory=dict)
    direct_trigger_signal_task_counts: dict[str, int] = Field(default_factory=dict)
    direct_trigger_signal_counts: dict[str, int] = Field(default_factory=dict)
    readiness_status_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    readiness_trigger_alignment_counts: dict[
        str, dict[str, int]
    ] = Field(default_factory=dict)

    measurement_tension_task_count: int = Field(ge=0, default=0)
    proxy_enrichment_candidate_task_count: int = Field(ge=0, default=0)
    total_grounded_source_chunk_count: int = Field(ge=0, default=0)
    total_unresolved_grounded_node_count: int = Field(ge=0, default=0)
    total_ambiguous_grounded_node_count: int = Field(ge=0, default=0)

    llm_calls_performed: Literal[0] = 0
    deterministic_batch_audit: Literal[True] = True
    shadow_only: Literal[True] = True
    scientific_conflict_authority: Literal[False] = False
    regime_boundary_authority: Literal[False] = False
    proxy_semantics_authority: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    candidate_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeBenchmarkAuditReport":
        if self.discovered_task_count != len(self.task_audits):
            raise ValueError("discovered_task_count must equal task audit count")
        complete = sum(row.status == "complete" for row in self.task_audits)
        errors = sum(row.status == "error" for row in self.task_audits)
        if self.complete_task_count != complete or self.error_task_count != errors:
            raise ValueError("complete/error counts do not match task audits")
        keys = [row.task_key for row in self.task_audits]
        if len(keys) != len(set(keys)):
            raise ValueError("benchmark task keys must be unique")
        return self


def operator_ready_without_mandatory_enrichment(status: str | None) -> bool:
    return str(status or "") in _READY_WITHOUT_MANDATORY


def trigger_pattern_from_signals(
    *,
    latent: bool | None,
    regime: bool | None,
) -> TriggerPattern:
    if latent is None or regime is None:
        return "unavailable"
    if latent and regime:
        return "latent_and_regime"
    if latent:
        return "latent_only"
    if regime:
        return "regime_only"
    return "none"


def readiness_trigger_alignment(
    *,
    readiness_status: str | None,
    trigger_signal: bool | None,
    has_trigger_contract: bool = True,
) -> ReadinessTriggerAlignment:
    if not has_trigger_contract:
        return "no_trigger_contract"
    if readiness_status is None or trigger_signal is None:
        return "unavailable"
    ready = operator_ready_without_mandatory_enrichment(readiness_status)
    if ready and trigger_signal:
        return "ready_and_triggered"
    if ready and not trigger_signal:
        return "ready_not_triggered"
    if not ready and trigger_signal:
        return "triggered_not_ready"
    return "not_ready_not_triggered"


def _stable_audit_id(
    *,
    benchmark_root: str,
    extraction_roots: list[str],
    rows: list[BenchmarkTaskReframeAudit],
) -> str:
    payload = {
        "benchmark_root": benchmark_root,
        "extraction_roots": list(extraction_roots),
        "tasks": [row.model_dump(mode="json") for row in rows],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"scientific_reframe_benchmark_audit:{digest}"


def summarize_benchmark_task_audits(
    *,
    benchmark_root: str,
    extraction_roots: list[str],
    task_audits: list[BenchmarkTaskReframeAudit],
) -> ScientificReframeBenchmarkAuditReport:
    rows = sorted(task_audits, key=lambda row: row.task_key)
    complete = [row for row in rows if row.status == "complete"]

    pattern_counts = Counter(row.trigger_pattern for row in complete)
    tension_task_counts: Counter[str] = Counter()
    tension_witness_counts: Counter[str] = Counter()
    direct_signal_task_counts: Counter[str] = Counter()
    direct_signal_counts: Counter[str] = Counter()
    readiness_counts: dict[str, Counter[str]] = defaultdict(Counter)
    alignment_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in complete:
        for tension_type, count in row.tension_type_counts.items():
            if count > 0:
                tension_task_counts[tension_type] += 1
                tension_witness_counts[tension_type] += int(count)
        for signal_kind, count in row.direct_trigger_signal_kind_counts.items():
            if count > 0:
                direct_signal_task_counts[signal_kind] += 1
                direct_signal_counts[signal_kind] += int(count)
        for operator_id, status in row.readiness_status_by_operator.items():
            readiness_counts[operator_id][status] += 1
        for operator_id, alignment in (
            row.readiness_trigger_alignment_by_operator.items()
        ):
            alignment_counts[operator_id][alignment] += 1

    return ScientificReframeBenchmarkAuditReport(
        audit_id=_stable_audit_id(
            benchmark_root=benchmark_root,
            extraction_roots=extraction_roots,
            rows=rows,
        ),
        benchmark_root=benchmark_root,
        extraction_roots=list(extraction_roots),
        task_audits=rows,
        discovered_task_count=len(rows),
        complete_task_count=len(complete),
        error_task_count=len(rows) - len(complete),
        unique_case_count=len({row.case_key for row in rows}),
        trigger_pattern_counts=dict(sorted(pattern_counts.items())),
        tension_type_task_counts=dict(sorted(tension_task_counts.items())),
        tension_type_witness_counts=dict(sorted(tension_witness_counts.items())),
        direct_trigger_signal_task_counts=dict(sorted(direct_signal_task_counts.items())),
        direct_trigger_signal_counts=dict(sorted(direct_signal_counts.items())),
        readiness_status_counts={
            operator_id: dict(sorted(counts.items()))
            for operator_id, counts in sorted(readiness_counts.items())
        },
        readiness_trigger_alignment_counts={
            operator_id: dict(sorted(counts.items()))
            for operator_id, counts in sorted(alignment_counts.items())
        },
        measurement_tension_task_count=sum(
            row.measurement_tension_present for row in complete
        ),
        proxy_enrichment_candidate_task_count=sum(
            row.proxy_enrichment_candidate for row in complete
        ),
        total_grounded_source_chunk_count=sum(
            row.grounded_source_chunk_count for row in complete
        ),
        total_unresolved_grounded_node_count=sum(
            row.unresolved_grounded_node_count for row in complete
        ),
        total_ambiguous_grounded_node_count=sum(
            row.ambiguous_grounded_node_count for row in complete
        ),
    )
