from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.reframing.reframe_contracts import (
    ImplementedReframeOperatorId,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ReframeCriticDimension = Literal[
    "operator_validity",
    "premise_fidelity",
    "explanatory_span",
    "construct_burden",
    "task_obligation_coverage",
    "differential_prediction_quality",
    "falsifiability",
    "over_specificity",
    "triviality",
    "reframe_depth",
]

CRITIC_DIMENSIONS: tuple[ReframeCriticDimension, ...] = (
    "operator_validity",
    "premise_fidelity",
    "explanatory_span",
    "construct_burden",
    "task_obligation_coverage",
    "differential_prediction_quality",
    "falsifiability",
    "over_specificity",
    "triviality",
    "reframe_depth",
)


class ReframeCriticDimensionDraft(StrictModel):
    """Loss-tolerant semantic review item.

    The LLM-facing schema deliberately permits unknown dimension names and
    out-of-range ratings. Compilation normalizes these values so one formatting
    mistake cannot discard the entire critic response.
    """

    dimension: str = Field(min_length=1)
    rating: int | str | None = None
    rationale: str = ""
    supporting_statement_ids: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class ReframeCriticDraft(StrictModel):
    candidate_id: str = Field(min_length=1)
    dimensions: list[ReframeCriticDimensionDraft] = Field(default_factory=list)
    cross_cutting_concerns: list[str] = Field(default_factory=list)


class ReframeCriticDimensionReview(StrictModel):
    dimension: ReframeCriticDimension
    rating: int | None = Field(default=None, ge=0, le=3)
    review_status: Literal["reviewed", "missing", "invalid"]
    rationale: str = ""
    supporting_statement_ids: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class ReframeStructuralAudit(StrictModel):
    grounded_premise_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    baseline_explained_count: int = Field(ge=0)
    alternative_explained_count: int = Field(ge=0)
    speculative_construct_count: int = Field(ge=0)
    differential_prediction_count: int = Field(ge=0)
    falsifier_count: int = Field(ge=0)
    primary_observable_count: int = Field(ge=0)
    operator_shape_valid: bool
    all_premise_ids_grounded: bool
    all_gap_ids_grounded: bool


class ScientificReframeCriticReview(StrictModel):
    schema_version: Literal[
        "scientific-reframe-critic-review-v1"
    ] = "scientific-reframe-critic-review-v1"

    review_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    operator_id: ImplementedReframeOperatorId
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    structural_audit: ReframeStructuralAudit
    dimensions: list[ReframeCriticDimensionReview]
    cross_cutting_concerns: list[str] = Field(default_factory=list)
    normalization_notes: list[str] = Field(default_factory=list)
    llm_review_complete: bool = False
    llm_error_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None

    shadow_only: Literal[True] = True
    external_novelty_evaluated: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False
    production_selection_changed: Literal[False] = False
    legacy_portfolio_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_vector(self) -> "ScientificReframeCriticReview":
        dimensions = [row.dimension for row in self.dimensions]
        if tuple(dimensions) != CRITIC_DIMENSIONS:
            raise ValueError("critic dimensions must use the canonical vector order")
        return self


class ScientificReframeCriticReport(StrictModel):
    schema_version: Literal[
        "scientific-reframe-critic-report-v1"
    ] = "scientific-reframe-critic-report-v1"

    report_id: str = Field(min_length=1)
    source_shadow_report_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    backend_name: str
    model_name: str
    reviews: list[ScientificReframeCriticReview]
    llm_calls_attempted: int = Field(ge=0)
    llm_calls_succeeded: int = Field(ge=0)

    shadow_only: Literal[True] = True
    quality_vector_only: Literal[True] = True
    candidate_ranking_performed: Literal[False] = False
    overall_score_computed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    legacy_portfolio_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> "ScientificReframeCriticReport":
        candidate_ids = [row.candidate_id for row in self.reviews]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("critic report candidate IDs must be unique")
        if self.llm_calls_succeeded > self.llm_calls_attempted:
            raise ValueError("successful critic calls cannot exceed attempted calls")
        return self


def stable_critic_review_id(*, candidate_id: str, context_sha256: str) -> str:
    digest = hashlib.sha256(
        f"{candidate_id}|{context_sha256}|reframe-critic-v1".encode("utf-8")
    ).hexdigest()[:20]
    return f"reframe_critic:{digest}"


def stable_critic_report_id(
    *,
    shadow_report_id: str,
    review_ids: list[str],
) -> str:
    payload = json.dumps(
        [shadow_report_id, *sorted(review_ids)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"reframe_critic_report:{digest}"
