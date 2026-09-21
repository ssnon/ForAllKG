from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.critic_contracts import (
    CRITIC_DIMENSIONS,
    ReframeCriticDimension,
    ScientificReframeCriticReport,
)
from pipeline_core.discovery.reframing.portfolio_contracts import (
    ParetoStatus,
    PortfolioRouteKind,
    PortfolioSlotId,
    RiskFlag,
    ScientificReframeShadowPortfolio,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ImplementedReframeOperatorId,
    ScientificReframingShadowReport,
)
from pipeline_core.discovery.reframing.selective_execution import (
    BenchmarkTaskSelectiveExecution,
    ScientificReframeSelectiveExecutionReport,
    load_json_model,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    DirectTriggerSignalKind,
    ScientificReframeTriggerReport,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CriticVectorEntry(StrictModel):
    dimension: ReframeCriticDimension
    rating: int | None = Field(default=None, ge=0, le=3)
    review_status: Literal["reviewed", "missing", "invalid"]
    rationale: str = ""
    concerns: list[str] = Field(default_factory=list)


class BenchmarkReframeCandidateSynthesis(StrictModel):
    task_key: str = Field(min_length=1)
    case_key: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    operator_id: ImplementedReframeOperatorId
    title: str = Field(min_length=1)

    trigger_tension_types: list[str] = Field(default_factory=list)
    trigger_direct_signal_kinds: list[DirectTriggerSignalKind] = Field(default_factory=list)

    challenged_assumption: str = Field(min_length=1)
    proposed_constructs: list[str] = Field(default_factory=list)
    latent_constructs: list[str] = Field(default_factory=list)
    boundary_variables: list[str] = Field(default_factory=list)
    regime_change_kind: str | None = None

    baseline_summary: str = Field(min_length=1)
    alternative_summary: str = Field(min_length=1)
    differential_predictions: list[str] = Field(default_factory=list)
    falsifiers: list[str] = Field(default_factory=list)
    discriminating_test: str = Field(min_length=1)

    grounded_premise_count: int = Field(ge=0)
    alternative_explained_count: int = Field(ge=0)
    speculative_construct_count: int = Field(ge=0)
    critic_vector: list[CriticVectorEntry]
    cross_cutting_concerns: list[str] = Field(default_factory=list)

    pareto_status: ParetoStatus
    route_kind: PortfolioRouteKind
    assigned_slot_id: PortfolioSlotId | None = None
    risk_flags: list[RiskFlag] = Field(default_factory=list)

    lexical_construct_tokens: list[str] = Field(default_factory=list)
    lexical_construct_signature: str = ""

    @model_validator(mode="after")
    def validate_vector(self) -> "BenchmarkReframeCandidateSynthesis":
        if tuple(row.dimension for row in self.critic_vector) != CRITIC_DIMENSIONS:
            raise ValueError("candidate synthesis must preserve canonical critic vector order")
        return self


class LexicalOverlapDiagnostic(StrictModel):
    operator_id: ImplementedReframeOperatorId
    left_candidate_id: str = Field(min_length=1)
    right_candidate_id: str = Field(min_length=1)
    jaccard_overlap: float = Field(ge=0.0, le=1.0)
    shared_tokens: list[str] = Field(default_factory=list)
    left_only_tokens: list[str] = Field(default_factory=list)
    right_only_tokens: list[str] = Field(default_factory=list)
    threshold_exceeded: bool

    diagnostic_only: Literal[True] = True
    semantic_equivalence_asserted: Literal[False] = False
    semantic_collapse_asserted: Literal[False] = False


class OperatorSynthesisSummary(StrictModel):
    operator_id: ImplementedReframeOperatorId
    candidate_count: int = Field(ge=0)
    task_count: int = Field(ge=0)
    slot_counts: dict[str, int] = Field(default_factory=dict)
    pareto_status_counts: dict[str, int] = Field(default_factory=dict)
    risk_flag_counts: dict[str, int] = Field(default_factory=dict)
    dimension_rating_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    repeated_construct_tokens: dict[str, int] = Field(default_factory=dict)
    exact_construct_signature_duplicate_groups: list[list[str]] = Field(default_factory=list)
    lexical_overlap_pair_count: int = Field(ge=0)
    lexical_overlap_threshold_exceeded_count: int = Field(ge=0)


class ScientificReframeBenchmarkSynthesis(StrictModel):
    schema_version: Literal[
        "scientific-reframe-benchmark-synthesis-v1"
    ] = "scientific-reframe-benchmark-synthesis-v1"

    synthesis_id: str = Field(min_length=1)
    source_execution_id: str = Field(min_length=1)
    source_benchmark_audit_id: str = Field(min_length=1)
    benchmark_root: str = Field(min_length=1)

    triggered_task_count: int = Field(ge=0)
    completed_task_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    operator_summaries: list[OperatorSynthesisSummary]
    slot_counts: dict[str, int] = Field(default_factory=dict)
    candidates: list[BenchmarkReframeCandidateSynthesis]
    lexical_overlap_diagnostics: list[LexicalOverlapDiagnostic]
    artifact_lineage_errors: list[str] = Field(default_factory=list)

    lexical_overlap_threshold: float = Field(ge=0.0, le=1.0)
    lexical_overlap_diagnostic_only: Literal[True] = True
    semantic_collapse_evaluated: Literal[False] = False
    semantic_equivalence_evaluated: Literal[False] = False
    aggregate_quality_score_computed: Literal[False] = False
    candidate_ranking_performed: Literal[False] = False
    winner_selected: Literal[False] = False
    shadow_only: Literal[True] = True
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeBenchmarkSynthesis":
        if self.candidate_count != len(self.candidates):
            raise ValueError("candidate_count must equal candidates length")
        candidate_ids = [row.candidate_id for row in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("benchmark synthesis candidate IDs must be unique")
        operators = [row.operator_id for row in self.operator_summaries]
        if len(operators) != len(set(operators)):
            raise ValueError("operator summaries must be unique")
        return self


_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "via", "with", "within",
    "model", "response", "effect", "variable", "construct", "latent",
    "regime", "boundary", "scientific", "reframe", "sers",
}


def _stem_token(token: str) -> str:
    value = token.lower().strip()
    if len(value) > 5 and value.endswith("ies"):
        value = value[:-3] + "y"
    elif len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    elif len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def lexical_construct_tokens(texts: list[str]) -> list[str]:
    tokens: set[str] = set()
    for text in texts:
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9]+", text or ""):
            token = _stem_token(raw)
            if len(token) < 3 or token in _STOPWORDS:
                continue
            tokens.add(token)
    return sorted(tokens)


def _jaccard(left: list[str], right: list[str]) -> tuple[float, list[str], list[str], list[str]]:
    a = set(left)
    b = set(right)
    union = a | b
    score = len(a & b) / len(union) if union else 0.0
    return score, sorted(a & b), sorted(a - b), sorted(b - a)


def _stable_synthesis_id(execution_id: str, candidate_ids: list[str]) -> str:
    payload = json.dumps(
        [execution_id, *sorted(candidate_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"scientific_reframe_benchmark_synthesis:{digest}"


def _validate_task_lineage(
    *,
    task: BenchmarkTaskSelectiveExecution,
    trigger: ScientificReframeTriggerReport,
    shadow: ScientificReframingShadowReport,
    critic: ScientificReframeCriticReport,
    portfolio: ScientificReframeShadowPortfolio,
) -> list[str]:
    errors: list[str] = []
    if task.task_id and trigger.source_task_id != task.task_id:
        errors.append(f"{task.task_key}: trigger task lineage mismatch")
    if shadow.source_trigger_report_id != trigger.report_id:
        errors.append(f"{task.task_key}: shadow trigger lineage mismatch")
    if critic.source_shadow_report_id != shadow.report_id:
        errors.append(f"{task.task_key}: critic shadow lineage mismatch")
    if portfolio.source_shadow_report_id != shadow.report_id:
        errors.append(f"{task.task_key}: portfolio shadow lineage mismatch")
    if portfolio.source_critic_report_id != critic.report_id:
        errors.append(f"{task.task_key}: portfolio critic lineage mismatch")
    return errors


def _candidate_record(
    *,
    task: BenchmarkTaskSelectiveExecution,
    trigger: ScientificReframeTriggerReport,
    shadow: ScientificReframingShadowReport,
    critic: ScientificReframeCriticReport,
    portfolio: ScientificReframeShadowPortfolio,
    candidate_id: str,
) -> BenchmarkReframeCandidateSynthesis:
    candidate_by_id = {row.reframe_id: row for row in shadow.candidates}
    review_by_id = {row.candidate_id: row for row in critic.reviews}
    assessment_by_id = {row.candidate_id: row for row in portfolio.assessments}
    candidate = candidate_by_id[candidate_id]
    review = review_by_id[candidate_id]
    assessment = assessment_by_id[candidate_id]

    assessment_trigger = next(
        row for row in trigger.assessments if row.operator_id == candidate.operator_id
    )
    witness_by_id = {row.witness_id: row for row in trigger.tension_witnesses}
    signal_by_id = {row.signal_id: row for row in trigger.direct_trigger_signals}
    tension_types = sorted(
        {
            tension_type
            for witness_id in assessment_trigger.witness_ids
            if witness_id in witness_by_id
            for tension_type in witness_by_id[witness_id].evidence_tension_types
        }
    )
    direct_signal_kinds = sorted(
        {
            signal_by_id[signal_id].kind
            for signal_id in assessment_trigger.direct_signal_ids
            if signal_id in signal_by_id
        }
    )

    construct_texts = [
        candidate.title,
        *candidate.proposed_constructs,
        *candidate.latent_constructs,
        *candidate.boundary_variables,
    ]
    tokens = lexical_construct_tokens(construct_texts)

    return BenchmarkReframeCandidateSynthesis(
        task_key=task.task_key,
        case_key=task.case_key,
        candidate_id=candidate.reframe_id,
        operator_id=candidate.operator_id,
        title=candidate.title,
        trigger_tension_types=tension_types,
        trigger_direct_signal_kinds=direct_signal_kinds,
        challenged_assumption=candidate.challenged_assumption,
        proposed_constructs=candidate.proposed_constructs,
        latent_constructs=candidate.latent_constructs,
        boundary_variables=candidate.boundary_variables,
        regime_change_kind=candidate.regime_change_kind,
        baseline_summary=candidate.baseline_model.summary,
        alternative_summary=candidate.alternative_model.summary,
        differential_predictions=[
            f"{row.observable}: baseline={row.baseline_expectation} | alternative={row.alternative_expectation}"
            for row in candidate.differential_predictions
        ],
        falsifiers=[row.falsifying_outcome for row in candidate.falsifiers],
        discriminating_test=candidate.discriminating_test.test_design,
        grounded_premise_count=review.structural_audit.grounded_premise_count,
        alternative_explained_count=review.structural_audit.alternative_explained_count,
        speculative_construct_count=review.structural_audit.speculative_construct_count,
        critic_vector=[
            CriticVectorEntry(
                dimension=row.dimension,
                rating=row.rating,
                review_status=row.review_status,
                rationale=row.rationale,
                concerns=row.concerns,
            )
            for row in review.dimensions
        ],
        cross_cutting_concerns=review.cross_cutting_concerns,
        pareto_status=assessment.pareto_status,
        route_kind=assessment.route_kind,
        assigned_slot_id=assessment.assigned_slot_id,
        risk_flags=assessment.risk_flags,
        lexical_construct_tokens=tokens,
        lexical_construct_signature="|".join(tokens),
    )


def _operator_summary(
    operator_id: ImplementedReframeOperatorId,
    candidates: list[BenchmarkReframeCandidateSynthesis],
    overlap_rows: list[LexicalOverlapDiagnostic],
) -> OperatorSynthesisSummary:
    rows = [row for row in candidates if row.operator_id == operator_id]
    slot_counts = Counter(
        row.assigned_slot_id or "UNASSIGNED"
        for row in rows
    )
    pareto_counts = Counter(row.pareto_status for row in rows)
    risk_counts = Counter(flag for row in rows for flag in row.risk_flags)

    dimension_counts: dict[str, Counter[str]] = {
        dimension: Counter() for dimension in CRITIC_DIMENSIONS
    }
    for row in rows:
        for item in row.critic_vector:
            key = "missing" if item.rating is None else str(item.rating)
            dimension_counts[item.dimension][key] += 1

    token_counts = Counter(token for row in rows for token in row.lexical_construct_tokens)
    repeated = {
        token: count
        for token, count in sorted(token_counts.items())
        if count >= 2
    }

    by_signature: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_signature[row.lexical_construct_signature].append(row.candidate_id)
    duplicate_groups = sorted(
        [sorted(candidate_ids) for candidate_ids in by_signature.values() if len(candidate_ids) >= 2],
        key=lambda values: values[0],
    )

    op_overlaps = [row for row in overlap_rows if row.operator_id == operator_id]
    return OperatorSynthesisSummary(
        operator_id=operator_id,
        candidate_count=len(rows),
        task_count=len({row.task_key for row in rows}),
        slot_counts=dict(sorted(slot_counts.items())),
        pareto_status_counts=dict(sorted(pareto_counts.items())),
        risk_flag_counts=dict(sorted(risk_counts.items())),
        dimension_rating_counts={
            dimension: dict(sorted(counts.items()))
            for dimension, counts in dimension_counts.items()
        },
        repeated_construct_tokens=repeated,
        exact_construct_signature_duplicate_groups=duplicate_groups,
        lexical_overlap_pair_count=len(op_overlaps),
        lexical_overlap_threshold_exceeded_count=sum(
            row.threshold_exceeded for row in op_overlaps
        ),
    )


def build_benchmark_reframe_synthesis(
    *,
    execution: ScientificReframeSelectiveExecutionReport,
    lexical_overlap_threshold: float = 0.35,
    fail_on_lineage_error: bool = True,
) -> ScientificReframeBenchmarkSynthesis:
    if not 0.0 <= lexical_overlap_threshold <= 1.0:
        raise ValueError("lexical_overlap_threshold must be between 0 and 1")

    candidates: list[BenchmarkReframeCandidateSynthesis] = []
    lineage_errors: list[str] = []

    for task in execution.task_executions:
        if task.status == "skipped_not_triggered":
            continue
        if task.status not in {"completed", "completed_with_errors"}:
            lineage_errors.append(f"{task.task_key}: task execution status={task.status}")
            continue
        if not all(
            stage.artifact_path
            for stage in (task.generation, task.critic, task.portfolio)
        ):
            lineage_errors.append(f"{task.task_key}: missing stage artifact path")
            continue

        canonical = Path(task.canonical_dir)
        trigger_path = canonical / "scientific_reframing_triggers.json"
        try:
            trigger = load_json_model(trigger_path, ScientificReframeTriggerReport)
            shadow = load_json_model(
                Path(task.generation.artifact_path), ScientificReframingShadowReport
            )
            critic = load_json_model(
                Path(task.critic.artifact_path), ScientificReframeCriticReport
            )
            portfolio = load_json_model(
                Path(task.portfolio.artifact_path), ScientificReframeShadowPortfolio
            )
        except Exception as exc:
            lineage_errors.append(
                f"{task.task_key}: artifact load failed: {type(exc).__name__}: {exc}"
            )
            continue

        task_errors = _validate_task_lineage(
            task=task,
            trigger=trigger,
            shadow=shadow,
            critic=critic,
            portfolio=portfolio,
        )
        lineage_errors.extend(task_errors)
        if task_errors:
            continue

        shadow_ids = {row.reframe_id for row in shadow.candidates}
        critic_ids = {row.candidate_id for row in critic.reviews}
        portfolio_ids = {row.candidate_id for row in portfolio.assessments}
        if not (shadow_ids == critic_ids == portfolio_ids):
            lineage_errors.append(
                f"{task.task_key}: candidate coverage mismatch across shadow/critic/portfolio"
            )
            continue
        if task.candidate_count != len(shadow_ids):
            lineage_errors.append(
                f"{task.task_key}: execution candidate_count does not match shadow artifact"
            )
            continue

        for candidate_id in sorted(shadow_ids):
            candidates.append(
                _candidate_record(
                    task=task,
                    trigger=trigger,
                    shadow=shadow,
                    critic=critic,
                    portfolio=portfolio,
                    candidate_id=candidate_id,
                )
            )

    if lineage_errors and fail_on_lineage_error:
        raise ValueError("benchmark synthesis lineage errors: " + " | ".join(lineage_errors))

    overlap_rows: list[LexicalOverlapDiagnostic] = []
    for operator_id in ("LATENT_VARIABLE", "REGIME_BOUNDARY"):
        rows = sorted(
            (row for row in candidates if row.operator_id == operator_id),
            key=lambda row: row.candidate_id,
        )
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left = rows[left_index]
                right = rows[right_index]
                score, shared, left_only, right_only = _jaccard(
                    left.lexical_construct_tokens,
                    right.lexical_construct_tokens,
                )
                overlap_rows.append(
                    LexicalOverlapDiagnostic(
                        operator_id=operator_id,
                        left_candidate_id=left.candidate_id,
                        right_candidate_id=right.candidate_id,
                        jaccard_overlap=round(score, 6),
                        shared_tokens=shared,
                        left_only_tokens=left_only,
                        right_only_tokens=right_only,
                        threshold_exceeded=(score >= lexical_overlap_threshold and bool(shared)),
                    )
                )

    operator_summaries = [
        _operator_summary(operator_id, candidates, overlap_rows)
        for operator_id in ("LATENT_VARIABLE", "REGIME_BOUNDARY")
    ]
    slot_counts = Counter(
        row.assigned_slot_id or "UNASSIGNED"
        for row in candidates
    )

    synthesis_id = _stable_synthesis_id(
        execution.execution_id,
        [row.candidate_id for row in candidates],
    )
    return ScientificReframeBenchmarkSynthesis(
        synthesis_id=synthesis_id,
        source_execution_id=execution.execution_id,
        source_benchmark_audit_id=execution.source_benchmark_audit_id,
        benchmark_root=execution.benchmark_root,
        triggered_task_count=execution.triggered_task_count,
        completed_task_count=execution.completed_task_count,
        candidate_count=len(candidates),
        operator_summaries=operator_summaries,
        slot_counts=dict(sorted(slot_counts.items())),
        candidates=sorted(candidates, key=lambda row: (row.task_key, row.operator_id, row.candidate_id)),
        lexical_overlap_diagnostics=overlap_rows,
        artifact_lineage_errors=lineage_errors,
        lexical_overlap_threshold=lexical_overlap_threshold,
    )
