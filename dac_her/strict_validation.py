from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from dac_her.draft_schema import KnowledgeGraphDraft
from dac_her.graph_normalization import normalize_graph_vocabularies
from dac_her.graph_validation import collect_graph_issues
from dac_her.measurement_scalarization import (
    format_scalarization_errors,
    measurement_scalarization_issues,
)
from dac_her.schemas import KnowledgeGraph
from dac_her.validation import validate_graph_provenance
from dac_her.validation_issues import (
    IssueCode,
    IssueStage,
    ValidationReport,
    issue,
)
from dac_her.vocab_registry import VocabularyRegistry


@dataclass(frozen=True)
class ValidationContext:
    paper_id: str
    chunk_id: str
    section: str
    document_id: str
    document_role: str
    page_ids: tuple[int, ...] | list[int]
    asset_ids: tuple[str, ...] | list[str]


@dataclass(frozen=True)
class FinalizationResult:
    report: ValidationReport
    graph: KnowledgeGraph | None
    vocabulary_issues: list[Any]


def validate_draft(draft: KnowledgeGraphDraft) -> ValidationReport:
    return collect_graph_issues(draft)


def finalize_draft(
    *,
    draft: KnowledgeGraphDraft,
    context: ValidationContext,
    experiment_registry: VocabularyRegistry,
    metric_registry: VocabularyRegistry,
) -> FinalizationResult:
    report = validate_draft(draft)
    if not report.valid:
        return FinalizationResult(report=report, graph=None, vocabulary_issues=[])

    try:
        graph = KnowledgeGraph.model_validate(draft.model_dump())
    except ValidationError as error:
        fallback = issue(
            code=IssueCode.FINAL_STRICT_VALIDATION_FAILURE,
            stage=IssueStage.FINALIZATION,
            message=str(error),
        )
        return FinalizationResult(
            report=ValidationReport.from_issues([fallback]),
            graph=None,
            vocabulary_issues=[],
        )

    try:
        graph, vocabulary_issues = normalize_graph_vocabularies(
            graph,
            experiment_registry=experiment_registry,
            metric_registry=metric_registry,
        )
    except Exception as error:
        fallback = issue(
            code=IssueCode.VOCABULARY_NORMALIZATION_FAILURE,
            stage=IssueStage.VOCABULARY,
            message=f"{type(error).__name__}: {error}",
        )
        return FinalizationResult(
            report=ValidationReport.from_issues([fallback]),
            graph=None,
            vocabulary_issues=[],
        )

    try:
        graph = KnowledgeGraph.model_validate(graph.model_dump())
    except ValidationError as error:
        fallback = issue(
            code=IssueCode.FINAL_STRICT_VALIDATION_FAILURE,
            stage=IssueStage.FINALIZATION,
            message=(
                "Vocabulary-normalized graph failed strict revalidation: "
                f"{error}"
            ),
        )
        return FinalizationResult(
            report=ValidationReport.from_issues([fallback]),
            graph=None,
            vocabulary_issues=vocabulary_issues,
        )

    scalar_issues = measurement_scalarization_issues(graph)
    if scalar_issues:
        fallback = issue(
            code=IssueCode.SCALARIZATION_FAILURE,
            stage=IssueStage.SCALARIZATION,
            message=format_scalarization_errors(scalar_issues),
        )
        return FinalizationResult(
            report=ValidationReport.from_issues([fallback]),
            graph=None,
            vocabulary_issues=vocabulary_issues,
        )

    try:
        validate_graph_provenance(
            graph,
            paper_id=context.paper_id,
            chunk_id=context.chunk_id,
            section=context.section,
            document_id=context.document_id,
            document_role=context.document_role,
            page_ids=context.page_ids,
            asset_ids=context.asset_ids,
        )
    except Exception as error:
        fallback = issue(
            code=IssueCode.EXTERNAL_PROVENANCE_FAILURE,
            stage=IssueStage.PROVENANCE,
            message=f"{type(error).__name__}: {error}",
        )
        return FinalizationResult(
            report=ValidationReport.from_issues([fallback]),
            graph=None,
            vocabulary_issues=vocabulary_issues,
        )

    return FinalizationResult(
        report=ValidationReport.from_issues([]),
        graph=graph,
        vocabulary_issues=vocabulary_issues,
    )
