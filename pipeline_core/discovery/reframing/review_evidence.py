from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.evidence_tension import (
    EvidenceLevelTensionWitness,
    ScientificEvidenceTensionReport,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ReframeEvidenceStatement,
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.trigger_calibration import (
    OperatorId,
    TriggerReviewItem,
    TriggerReviewTemplate,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewStatement(StrictModel):
    statement_id: str
    text: str
    epistemic_role: str
    claim_kind: str
    paper_ids: list[str] = Field(default_factory=list)
    requires_verification: bool = False


class ReviewTensionWitness(StrictModel):
    witness_id: str
    source_tension_id: str
    source_tension_type: str
    tension_types: list[str] = Field(default_factory=list)
    side_a_statements: list[ReviewStatement] = Field(default_factory=list)
    side_b_statements: list[ReviewStatement] = Field(default_factory=list)
    focal_statement: ReviewStatement | None = None
    paper_ids: list[str] = Field(default_factory=list)
    claim_kinds: list[str] = Field(default_factory=list)
    classification_bases: list[str] = Field(default_factory=list)
    relevant_condition_signature_count: int = Field(ge=0, default=0)
    relevant_condition_names: list[str] = Field(default_factory=list)
    side_a_response_families: list[str] = Field(default_factory=list)
    side_b_response_families: list[str] = Field(default_factory=list)
    paired_grounded_sides: bool = False
    paired_response_signal: bool = False
    independence_basis: str
    independent_family_signal: bool

    diagnostic_only: Literal[True] = True
    scientific_conflict_authority: Literal[False] = False
    regime_boundary_authority: Literal[False] = False


class ReviewConditionSummary(StrictModel):
    example_count: int = Field(ge=0)
    distinct_signature_count: int = Field(ge=0)
    paper_count: int = Field(ge=0)
    object_kinds: list[str] = Field(default_factory=list)
    condition_names: list[str] = Field(default_factory=list)

    availability_signal_only: Literal[True] = True
    scientific_regime_authority: Literal[False] = False


class OperatorReviewEvidence(StrictModel):
    operator_id: OperatorId
    readiness_status: str | None = None
    current_trigger_decision: str | None = None
    current_trigger_signal: bool | None = None
    readiness_trigger_alignment: str | None = None
    review_question: str
    expert_label: None = None
    expert_rationale: None = None

    current_trigger_is_not_ground_truth: Literal[True] = True
    reviewer_should_ignore_current_decision_when_labeling: Literal[True] = True


class TaskTriggerReviewEvidence(StrictModel):
    task_key: str
    case_key: str
    question: str
    grounded_source_chunk_count: int = Field(ge=0)
    premise_statements: list[ReviewStatement]
    gap_statements: list[ReviewStatement] = Field(default_factory=list)
    tension_witnesses: list[ReviewTensionWitness] = Field(default_factory=list)
    condition_summary: ReviewConditionSummary
    operator_reviews: list[OperatorReviewEvidence]

    @model_validator(mode="after")
    def validate_operator_rows(self) -> "TaskTriggerReviewEvidence":
        ids = [row.operator_id for row in self.operator_reviews]
        if sorted(ids) != ["LATENT_VARIABLE", "REGIME_BOUNDARY"]:
            raise ValueError("task review evidence requires LATENT and REGIME rows")
        return self


class ScientificReframeTriggerReviewEvidencePack(StrictModel):
    schema_version: Literal[
        "scientific-reframe-trigger-review-evidence-v1"
    ] = "scientific-reframe-trigger-review-evidence-v1"
    pack_id: str
    source_audit_id: str
    source_review_template_schema: str
    benchmark_root: str
    extraction_roots: list[str]
    tasks: list[TaskTriggerReviewEvidence]
    task_count: int = Field(ge=0)
    operator_review_row_count: int = Field(ge=0)

    llm_calls_performed: Literal[0] = 0
    deterministic_evidence_pack: Literal[True] = True
    labels_prefilled: Literal[False] = False
    current_trigger_is_ground_truth: Literal[False] = False
    labels_are_scientific_authority: Literal[False] = False
    threshold_selection_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeTriggerReviewEvidencePack":
        if self.task_count != len(self.tasks):
            raise ValueError("task_count must equal tasks length")
        rows = sum(len(task.operator_reviews) for task in self.tasks)
        if self.operator_review_row_count != rows:
            raise ValueError("operator_review_row_count does not match tasks")
        keys = [task.task_key for task in self.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("task review evidence keys must be unique")
        return self


def _statement(row: ReframeEvidenceStatement) -> ReviewStatement:
    return ReviewStatement(
        statement_id=row.statement_id,
        text=row.text,
        epistemic_role=row.epistemic_role,
        claim_kind=row.claim_kind,
        paper_ids=sorted(set(row.paper_ids)),
        requires_verification=row.requires_verification,
    )


def _condition_signature(conditions: list[dict[str, Any]]) -> str:
    normalized = [
        {str(key): row[key] for key in sorted(row)}
        for row in conditions
        if isinstance(row, dict)
    ]
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _condition_summary(
    evidence: ScientificReframeEvidencePacket,
) -> ReviewConditionSummary:
    signatures: set[str] = set()
    papers: set[str] = set()
    kinds: set[str] = set()
    names: set[str] = set()
    for example in evidence.condition_examples:
        signatures.add(_condition_signature(example.conditions))
        papers.add(example.paper_id)
        kinds.add(example.object_kind)
        for condition in example.conditions:
            name = str(condition.get("name", "")).strip()
            if name:
                names.add(name)
    return ReviewConditionSummary(
        example_count=len(evidence.condition_examples),
        distinct_signature_count=len(signatures),
        paper_count=len(papers),
        object_kinds=sorted(kinds),
        condition_names=sorted(names),
    )


def _witness(
    row: EvidenceLevelTensionWitness,
    *,
    by_statement: dict[str, ReviewStatement],
) -> ReviewTensionWitness:
    def statements(ids: list[str]) -> list[ReviewStatement]:
        return [by_statement[sid] for sid in ids if sid in by_statement]

    return ReviewTensionWitness(
        witness_id=row.witness_id,
        source_tension_id=row.source_tension_id,
        source_tension_type=row.source_tension_type,
        tension_types=list(row.tension_types),
        side_a_statements=statements(row.side_a_statement_ids),
        side_b_statements=statements(row.side_b_statement_ids),
        focal_statement=(
            by_statement.get(row.focal_statement_id)
            if row.focal_statement_id is not None
            else None
        ),
        paper_ids=list(row.paper_ids),
        claim_kinds=list(row.claim_kinds),
        classification_bases=list(row.classification_bases),
        relevant_condition_signature_count=row.relevant_condition_signature_count,
        relevant_condition_names=list(row.relevant_condition_names),
        side_a_response_families=list(row.side_a_response_families),
        side_b_response_families=list(row.side_b_response_families),
        paired_grounded_sides=row.paired_grounded_sides,
        paired_response_signal=row.paired_response_signal,
        independence_basis=row.independence_basis,
        independent_family_signal=row.independent_family_signal,
    )


def _review_question(operator_id: OperatorId) -> str:
    if operator_id == "LATENT_VARIABLE":
        return (
            "Ignoring the current trigger decision, do the grounded statements show "
            "an unexplained cross-family contrast or context dependence for which one "
            "hidden-variable/latent-construct shadow generation attempt is scientifically "
            "worthwhile, rather than merely restating known moderators?"
        )
    return (
        "Ignoring the current trigger decision, do the grounded statements support a "
        "plausible qualitative change in response law/regime (threshold, crossover, "
        "saturation, sign/slope change, mechanism switch, or comparable boundary), "
        "rather than ordinary continuous context variation, with condition evidence "
        "local enough to the contrasted observations to justify one shadow attempt?"
    )


def build_task_trigger_review_evidence(
    *,
    template_rows: list[TriggerReviewItem],
    evidence: ScientificReframeEvidencePacket,
    tensions: ScientificEvidenceTensionReport,
) -> TaskTriggerReviewEvidence:
    if not template_rows:
        raise ValueError("task review evidence requires template rows")
    task_keys = {row.task_key for row in template_rows}
    case_keys = {row.case_key for row in template_rows}
    if len(task_keys) != 1 or len(case_keys) != 1:
        raise ValueError("template rows must refer to one task/case")
    operators = {row.operator_id for row in template_rows}
    if operators != {"LATENT_VARIABLE", "REGIME_BOUNDARY"}:
        raise ValueError("template rows must contain exactly LATENT and REGIME")
    questions = {row.question for row in template_rows if row.question is not None}
    if questions and questions != {evidence.question}:
        raise ValueError("review template question does not match grounded evidence")
    if tensions.source_task_id != evidence.task_id:
        raise ValueError("tension report task does not match grounded evidence")
    if tensions.source_context_id != evidence.source_context_id:
        raise ValueError("tension report context does not match grounded evidence")

    premise = [_statement(row) for row in evidence.premise_statements]
    gaps = [_statement(row) for row in evidence.gap_statements]
    by_statement = {row.statement_id: row for row in [*premise, *gaps]}
    witness_rows = [
        _witness(row, by_statement=by_statement)
        for row in tensions.witnesses
    ]
    operator_rows = [
        OperatorReviewEvidence(
            operator_id=row.operator_id,
            readiness_status=row.readiness_status,
            current_trigger_decision=row.trigger_decision,
            current_trigger_signal=row.trigger_signal,
            readiness_trigger_alignment=row.readiness_trigger_alignment,
            review_question=_review_question(row.operator_id),
        )
        for row in sorted(template_rows, key=lambda item: item.operator_id)
    ]
    first = template_rows[0]
    return TaskTriggerReviewEvidence(
        task_key=first.task_key,
        case_key=first.case_key,
        question=evidence.question,
        grounded_source_chunk_count=evidence.grounded_source_chunk_count,
        premise_statements=premise,
        gap_statements=gaps,
        tension_witnesses=witness_rows,
        condition_summary=_condition_summary(evidence),
        operator_reviews=operator_rows,
    )


def build_trigger_review_evidence_pack(
    *,
    template: TriggerReviewTemplate,
    benchmark_root: str,
    extraction_roots: list[str],
    task_evidence: dict[
        str, tuple[ScientificReframeEvidencePacket, ScientificEvidenceTensionReport]
    ],
) -> ScientificReframeTriggerReviewEvidencePack:
    rows_by_task: dict[str, list[TriggerReviewItem]] = {}
    for row in template.rows:
        rows_by_task.setdefault(row.task_key, []).append(row)
    missing = sorted(set(rows_by_task) - set(task_evidence))
    extra = sorted(set(task_evidence) - set(rows_by_task))
    if missing:
        raise ValueError("missing grounded review evidence for tasks: " + ", ".join(missing))
    if extra:
        raise ValueError("unexpected grounded review evidence tasks: " + ", ".join(extra))

    tasks = [
        build_task_trigger_review_evidence(
            template_rows=rows_by_task[task_key],
            evidence=task_evidence[task_key][0],
            tensions=task_evidence[task_key][1],
        )
        for task_key in sorted(rows_by_task)
    ]
    payload = {
        "source_audit_id": template.source_audit_id,
        "benchmark_root": benchmark_root,
        "extraction_roots": list(extraction_roots),
        "tasks": [task.model_dump(mode="json") for task in tasks],
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    return ScientificReframeTriggerReviewEvidencePack(
        pack_id=f"scientific_reframe_trigger_review_evidence:{digest}",
        source_audit_id=template.source_audit_id,
        source_review_template_schema=template.schema_version,
        benchmark_root=benchmark_root,
        extraction_roots=list(extraction_roots),
        tasks=tasks,
        task_count=len(tasks),
        operator_review_row_count=sum(len(task.operator_reviews) for task in tasks),
    )
