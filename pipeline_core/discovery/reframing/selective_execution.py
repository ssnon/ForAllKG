from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.benchmark_audit import (
    BenchmarkTaskReframeAudit,
    ScientificReframeBenchmarkAuditReport,
)
from pipeline_core.discovery.reframing.critic_contracts import (
    ScientificReframeCriticReport,
)
from pipeline_core.discovery.reframing.portfolio_contracts import (
    ScientificReframeShadowPortfolio,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ImplementedReframeOperatorId,
    ScientificReframingShadowReport,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ScientificReframeTriggerReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


StageStatus = Literal[
    "skipped",
    "executed",
    "resumed",
    "executed_with_errors",
    "failed",
    "blocked",
]

TaskExecutionStatus = Literal[
    "skipped_not_triggered",
    "completed",
    "completed_with_errors",
    "error",
]


class SelectiveExecutionStage(StrictModel):
    stage: Literal["generation", "critic", "portfolio"]
    status: StageStatus
    artifact_path: str | None = None
    llm_calls_attempted: int = Field(default=0, ge=0)
    llm_calls_succeeded: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None
    resume_reason: str | None = None

    @model_validator(mode="after")
    def validate_stage_state(self) -> "SelectiveExecutionStage":
        if self.llm_calls_succeeded > self.llm_calls_attempted:
            raise ValueError("successful LLM calls cannot exceed attempted calls")
        if self.status == "failed" and not (self.error_type or "").strip():
            raise ValueError("failed stages require error_type")
        if self.status != "failed" and (
            self.error_type is not None or self.error_message is not None
        ):
            raise ValueError("only failed stages may carry error fields")
        if self.status == "resumed" and not (self.resume_reason or "").strip():
            raise ValueError("resumed stages require resume_reason")
        return self


class BenchmarkTaskSelectiveExecution(StrictModel):
    schema_version: Literal[
        "scientific-reframe-selective-task-execution-v1"
    ] = "scientific-reframe-selective-task-execution-v1"

    task_key: str = Field(min_length=1)
    case_key: str = Field(min_length=1)
    replicate_key: str = Field(min_length=1)
    canonical_dir: str = Field(min_length=1)
    task_id: str | None = None
    trigger_pattern: str
    triggered_operator_ids: list[ImplementedReframeOperatorId]
    status: TaskExecutionStatus

    generation: SelectiveExecutionStage
    critic: SelectiveExecutionStage
    portfolio: SelectiveExecutionStage

    candidate_count: int = Field(default=0, ge=0)
    critic_review_count: int = Field(default=0, ge=0)
    portfolio_assessment_count: int = Field(default=0, ge=0)
    resume_used: bool = False

    llm_calls_attempted: int = Field(default=0, ge=0)
    llm_calls_succeeded: int = Field(default=0, ge=0)

    shadow_only: Literal[True] = True
    candidate_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False
    winner_selected: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    legacy_portfolio_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_task_state(self) -> "BenchmarkTaskSelectiveExecution":
        attempted = sum(
            row.llm_calls_attempted
            for row in (self.generation, self.critic, self.portfolio)
        )
        succeeded = sum(
            row.llm_calls_succeeded
            for row in (self.generation, self.critic, self.portfolio)
        )
        if self.llm_calls_attempted != attempted:
            raise ValueError("task llm_calls_attempted must equal stage sum")
        if self.llm_calls_succeeded != succeeded:
            raise ValueError("task llm_calls_succeeded must equal stage sum")
        if self.status == "skipped_not_triggered":
            if self.triggered_operator_ids:
                raise ValueError("skipped_not_triggered tasks cannot have triggered operators")
            if any(
                row.status != "skipped"
                for row in (self.generation, self.critic, self.portfolio)
            ):
                raise ValueError("skipped tasks require all stages skipped")
        return self


class ScientificReframeSelectiveExecutionReport(StrictModel):
    schema_version: Literal[
        "scientific-reframe-selective-execution-report-v1"
    ] = "scientific-reframe-selective-execution-report-v1"

    execution_id: str = Field(min_length=1)
    source_benchmark_audit_id: str = Field(min_length=1)
    benchmark_root: str = Field(min_length=1)
    extraction_roots: list[str] = Field(min_length=1)
    resume_enabled: bool
    task_executions: list[BenchmarkTaskSelectiveExecution]

    task_count: int = Field(ge=0)
    triggered_task_count: int = Field(ge=0)
    skipped_task_count: int = Field(ge=0)
    completed_task_count: int = Field(ge=0)
    completed_with_errors_task_count: int = Field(ge=0)
    error_task_count: int = Field(ge=0)
    resumed_task_count: int = Field(ge=0)

    status_counts: dict[str, int] = Field(default_factory=dict)
    llm_calls_attempted: int = Field(ge=0)
    llm_calls_succeeded: int = Field(ge=0)
    generation_calls_attempted: int = Field(ge=0)
    critic_calls_attempted: int = Field(ge=0)

    shadow_only: Literal[True] = True
    selective_execution_performed: Literal[True] = True
    resume_lineage_validation_performed: Literal[True] = True
    candidate_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False
    winner_selected: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    legacy_portfolio_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_report_counts(self) -> "ScientificReframeSelectiveExecutionReport":
        if self.task_count != len(self.task_executions):
            raise ValueError("task_count must equal task execution count")
        keys = [row.task_key for row in self.task_executions]
        if len(keys) != len(set(keys)):
            raise ValueError("task execution keys must be unique")
        status_counts = Counter(row.status for row in self.task_executions)
        expected = {
            "skipped_not_triggered": self.skipped_task_count,
            "completed": self.completed_task_count,
            "completed_with_errors": self.completed_with_errors_task_count,
            "error": self.error_task_count,
        }
        if any(status_counts.get(key, 0) != value for key, value in expected.items()):
            raise ValueError("task status counts do not match task executions")
        if self.triggered_task_count != self.task_count - self.skipped_task_count:
            raise ValueError("triggered_task_count must exclude skipped tasks")
        if self.resumed_task_count != sum(row.resume_used for row in self.task_executions):
            raise ValueError("resumed_task_count does not match task executions")
        if self.llm_calls_attempted != sum(
            row.llm_calls_attempted for row in self.task_executions
        ):
            raise ValueError("report llm_calls_attempted must equal task sum")
        if self.llm_calls_succeeded != sum(
            row.llm_calls_succeeded for row in self.task_executions
        ):
            raise ValueError("report llm_calls_succeeded must equal task sum")
        return self


def triggered_operator_ids(
    row: BenchmarkTaskReframeAudit,
) -> list[ImplementedReframeOperatorId]:
    ordered: tuple[ImplementedReframeOperatorId, ...] = (
        "LATENT_VARIABLE",
        "REGIME_BOUNDARY",
    )
    return [
        operator_id
        for operator_id in ordered
        if row.trigger_signal_by_operator.get(operator_id) is True
    ]


def load_json_model(path: Path, model_type):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return model_type.model_validate(payload)


def resumable_shadow_report(
    *,
    path: Path,
    trigger: ScientificReframeTriggerReport,
    expected_task_id: str | None,
) -> tuple[ScientificReframingShadowReport | None, str]:
    if not path.exists():
        return None, "shadow artifact absent"
    try:
        report = load_json_model(path, ScientificReframingShadowReport)
    except Exception as exc:  # resume should fail closed
        return None, f"shadow artifact invalid: {type(exc).__name__}"
    if expected_task_id and report.source_task_id != expected_task_id:
        return None, "shadow task lineage mismatch"
    if not report.scientific_trigger_evaluated:
        return None, "shadow lacks scientific trigger lineage"
    if report.source_trigger_report_id != trigger.report_id:
        return None, "shadow trigger lineage is stale"
    if any(run.decision == "generation_failed" for run in report.runs):
        return None, "shadow contains generation_failed run"
    triggered = {
        row.operator_id
        for row in trigger.assessments
        if row.decision == "triggered"
    }
    run_by_operator = {row.operator_id: row for row in report.runs}
    for operator_id in triggered:
        run = run_by_operator.get(operator_id)
        if run is None:
            return None, f"shadow missing triggered operator run: {operator_id}"
        if run.decision in {"skipped_not_triggered", "skipped_not_ready"}:
            return None, f"shadow skipped triggered operator: {operator_id}"
    return report, "valid current trigger lineage"


def resumable_critic_report(
    *,
    path: Path,
    shadow: ScientificReframingShadowReport,
) -> tuple[ScientificReframeCriticReport | None, str]:
    if not path.exists():
        return None, "critic artifact absent"
    try:
        report = load_json_model(path, ScientificReframeCriticReport)
    except Exception as exc:
        return None, f"critic artifact invalid: {type(exc).__name__}"
    if report.source_shadow_report_id != shadow.report_id:
        return None, "critic shadow lineage is stale"
    if report.source_task_id != shadow.source_task_id:
        return None, "critic task lineage mismatch"
    expected = {row.reframe_id for row in shadow.candidates}
    actual = {row.candidate_id for row in report.reviews}
    if actual != expected:
        return None, "critic candidate coverage mismatch"
    if any(not row.llm_review_complete or row.llm_error_type for row in report.reviews):
        return None, "critic contains incomplete or failed LLM review"
    return report, "valid current shadow lineage with complete reviews"


def resumable_portfolio_report(
    *,
    path: Path,
    shadow: ScientificReframingShadowReport,
    critic: ScientificReframeCriticReport,
) -> tuple[ScientificReframeShadowPortfolio | None, str]:
    if not path.exists():
        return None, "portfolio artifact absent"
    try:
        report = load_json_model(path, ScientificReframeShadowPortfolio)
    except Exception as exc:
        return None, f"portfolio artifact invalid: {type(exc).__name__}"
    if report.source_shadow_report_id != shadow.report_id:
        return None, "portfolio shadow lineage is stale"
    if report.source_critic_report_id != critic.report_id:
        return None, "portfolio critic lineage is stale"
    if report.source_task_id != shadow.source_task_id:
        return None, "portfolio task lineage mismatch"
    return report, "valid current shadow/critic lineage"


def stage_from_shadow(
    *,
    report: ScientificReframingShadowReport,
    path: Path,
    status: Literal["executed", "resumed"],
    resume_reason: str | None = None,
) -> SelectiveExecutionStage:
    attempted = report.llm_calls_performed if status == "executed" else 0
    succeeded_total = sum(
        run.decision
        not in {"generation_failed", "skipped_not_ready", "skipped_not_triggered"}
        for run in report.runs
    )
    succeeded = succeeded_total if status == "executed" else 0
    has_error = any(run.decision == "generation_failed" for run in report.runs)
    return SelectiveExecutionStage(
        stage="generation",
        status=("executed_with_errors" if has_error and status == "executed" else status),
        artifact_path=str(path),
        llm_calls_attempted=attempted,
        llm_calls_succeeded=succeeded,
        resume_reason=resume_reason,
    )


def stage_from_critic(
    *,
    report: ScientificReframeCriticReport,
    path: Path,
    status: Literal["executed", "resumed"],
    resume_reason: str | None = None,
) -> SelectiveExecutionStage:
    attempted = report.llm_calls_attempted if status == "executed" else 0
    succeeded = report.llm_calls_succeeded if status == "executed" else 0
    has_error = any(not row.llm_review_complete or row.llm_error_type for row in report.reviews)
    return SelectiveExecutionStage(
        stage="critic",
        status=("executed_with_errors" if has_error and status == "executed" else status),
        artifact_path=str(path),
        llm_calls_attempted=attempted,
        llm_calls_succeeded=succeeded,
        resume_reason=resume_reason,
    )


def stage_from_portfolio(
    *,
    path: Path,
    status: Literal["executed", "resumed"],
    resume_reason: str | None = None,
) -> SelectiveExecutionStage:
    return SelectiveExecutionStage(
        stage="portfolio",
        status=status,
        artifact_path=str(path),
        resume_reason=resume_reason,
    )


def skipped_stage(stage: Literal["generation", "critic", "portfolio"]) -> SelectiveExecutionStage:
    return SelectiveExecutionStage(stage=stage, status="skipped")


def failed_stage(
    stage: Literal["generation", "critic", "portfolio"],
    *,
    exc: Exception,
    artifact_path: Path | None = None,
) -> SelectiveExecutionStage:
    message = str(exc)
    if len(message) > 2000:
        message = message[-2000:]
    return SelectiveExecutionStage(
        stage=stage,
        status="failed",
        artifact_path=str(artifact_path) if artifact_path else None,
        error_type=type(exc).__name__,
        error_message=message,
    )


def blocked_stage(
    stage: Literal["generation", "critic", "portfolio"],
) -> SelectiveExecutionStage:
    return SelectiveExecutionStage(stage=stage, status="blocked")


def summarize_selective_execution(
    *,
    audit: ScientificReframeBenchmarkAuditReport,
    extraction_roots: list[str],
    resume_enabled: bool,
    task_executions: list[BenchmarkTaskSelectiveExecution],
) -> ScientificReframeSelectiveExecutionReport:
    rows = sorted(task_executions, key=lambda row: row.task_key)
    status_counts = Counter(row.status for row in rows)
    payload = {
        "audit_id": audit.audit_id,
        "roots": list(extraction_roots),
        "resume": resume_enabled,
        "tasks": [row.model_dump(mode="json") for row in rows],
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    return ScientificReframeSelectiveExecutionReport(
        execution_id=f"scientific_reframe_selective_execution:{digest}",
        source_benchmark_audit_id=audit.audit_id,
        benchmark_root=audit.benchmark_root,
        extraction_roots=list(extraction_roots),
        resume_enabled=resume_enabled,
        task_executions=rows,
        task_count=len(rows),
        triggered_task_count=sum(row.status != "skipped_not_triggered" for row in rows),
        skipped_task_count=status_counts.get("skipped_not_triggered", 0),
        completed_task_count=status_counts.get("completed", 0),
        completed_with_errors_task_count=status_counts.get("completed_with_errors", 0),
        error_task_count=status_counts.get("error", 0),
        resumed_task_count=sum(row.resume_used for row in rows),
        status_counts=dict(sorted(status_counts.items())),
        llm_calls_attempted=sum(row.llm_calls_attempted for row in rows),
        llm_calls_succeeded=sum(row.llm_calls_succeeded for row in rows),
        generation_calls_attempted=sum(row.generation.llm_calls_attempted for row in rows),
        critic_calls_attempted=sum(row.critic.llm_calls_attempted for row in rows),
    )
