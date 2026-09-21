from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ImplementedReframeOperatorId = Literal[
    "LATENT_VARIABLE",
    "REGIME_BOUNDARY",
]

RegimeChangeKind = Literal[
    "threshold",
    "saturation",
    "slope_change",
    "sign_change",
    "mechanism_switch",
    "non_monotonic_transition",
    "qualitative_response_change",
    "other",
]


class ScientificModelDraft(StrictModel):
    summary: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    explained_statement_ids: list[str] = Field(default_factory=list)
    expected_observations: list[str] = Field(default_factory=list)


class DifferentialPredictionDraft(StrictModel):
    local_id: str = Field(min_length=1)
    observable: str = Field(min_length=1)
    baseline_expectation: str = Field(min_length=1)
    alternative_expectation: str = Field(min_length=1)
    discriminating_outcome: str = Field(min_length=1)

class ReframeFalsifierDraft(StrictModel):
    local_id: str = Field(min_length=1)
    falsifying_outcome: str = Field(min_length=1)


class DiscriminatingTestDraft(StrictModel):
    test_design: str = Field(min_length=1)
    primary_observables: list[str] = Field(default_factory=list)
    baseline_favoring_outcome: str = Field(min_length=1)
    alternative_favoring_outcome: str = Field(min_length=1)


class ScientificReframeDraft(StrictModel):
    """Loss-tolerant structured LLM draft.

    Scientific/operator semantics are intentionally compiled after parsing, when
    the grounded evidence scope is available. Keeping those checks out of the
    Instructor response model prevents bookkeeping-only deviations from
    discarding an otherwise usable model contrast before provenance validation.
    """

    local_id: str = Field(min_length=1)
    operator_id: ImplementedReframeOperatorId
    title: str = Field(min_length=1)

    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)

    baseline_model: ScientificModelDraft
    alternative_model: ScientificModelDraft
    challenged_assumption: str = Field(min_length=1)
    proposed_constructs: list[str] = Field(default_factory=list)

    latent_constructs: list[str] = Field(default_factory=list)
    boundary_variables: list[str] = Field(default_factory=list)
    regime_change_kind: RegimeChangeKind | None = None

    differential_predictions: list[DifferentialPredictionDraft] = Field(
        default_factory=list
    )
    falsifiers: list[ReframeFalsifierDraft] = Field(default_factory=list)
    discriminating_test: DiscriminatingTestDraft
    unresolved_questions: list[str] = Field(default_factory=list)


class ScientificReframeBatchDraft(StrictModel):
    """Loss-tolerant generation envelope; semantics are compiled later."""

    operator_id: ImplementedReframeOperatorId
    candidates: list[ScientificReframeDraft] = Field(default_factory=list)
    abstention_reason: str | None = None


class ScientificReframeCandidate(StrictModel):
    schema_version: Literal[
        "scientific-reframe-candidate-v1"
    ] = "scientific-reframe-candidate-v1"

    reframe_id: str = Field(min_length=1)
    operator_id: ImplementedReframeOperatorId
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)

    title: str
    premise_statement_ids: list[str]
    gap_statement_ids: list[str]
    baseline_model: ScientificModelDraft
    alternative_model: ScientificModelDraft
    challenged_assumption: str
    proposed_constructs: list[str]
    latent_constructs: list[str]
    boundary_variables: list[str]
    regime_change_kind: RegimeChangeKind | None
    differential_predictions: list[DifferentialPredictionDraft]
    falsifiers: list[ReframeFalsifierDraft]
    discriminating_test: DiscriminatingTestDraft
    unresolved_questions: list[str]
    provenance_normalizations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_compiled_shape(self) -> "ScientificReframeCandidate":
        premise_ids = set(self.premise_statement_ids)
        if not premise_ids:
            raise ValueError("compiled reframe requires grounded premises")
        if len(self.premise_statement_ids) != len(premise_ids):
            raise ValueError("compiled premise_statement_ids must be unique")
        if len(self.gap_statement_ids) != len(set(self.gap_statement_ids)):
            raise ValueError("compiled gap_statement_ids must be unique")

        if self.baseline_model.summary.strip() == self.alternative_model.summary.strip():
            raise ValueError("baseline and alternative models must differ")
        for model_name, model in (
            ("baseline_model", self.baseline_model),
            ("alternative_model", self.alternative_model),
        ):
            if not model.explained_statement_ids:
                raise ValueError(f"{model_name} must explain at least one premise")
            if not set(model.explained_statement_ids) <= premise_ids:
                raise ValueError(
                    f"{model_name}.explained_statement_ids must be grounded premises"
                )
            if not model.expected_observations:
                raise ValueError(f"{model_name} requires expected_observations")

        if not self.differential_predictions:
            raise ValueError("compiled reframe requires a differential prediction")
        prediction_ids = [row.local_id for row in self.differential_predictions]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("duplicate differential prediction local_id")
        if any(
            row.baseline_expectation.strip() == row.alternative_expectation.strip()
            for row in self.differential_predictions
        ):
            raise ValueError("compiled differential predictions must contrast models")

        if not self.falsifiers:
            raise ValueError("compiled reframe requires a falsifier")
        falsifier_ids = [row.local_id for row in self.falsifiers]
        if len(falsifier_ids) != len(set(falsifier_ids)):
            raise ValueError("duplicate falsifier local_id")
        if not self.discriminating_test.primary_observables:
            raise ValueError("compiled discriminating test requires observables")

        if self.operator_id == "LATENT_VARIABLE":
            if not self.latent_constructs or not self.proposed_constructs:
                raise ValueError("LATENT_VARIABLE requires a proposed latent construct")
            if len(set(self.alternative_model.explained_statement_ids)) < 2:
                raise ValueError(
                    "LATENT_VARIABLE alternative model must explain at least two premises"
                )
            if self.boundary_variables or self.regime_change_kind is not None:
                raise ValueError("compiled LATENT_VARIABLE must not retain regime fields")
        else:
            if not self.boundary_variables or self.regime_change_kind is None:
                raise ValueError("REGIME_BOUNDARY requires explicit regime fields")
            if self.latent_constructs:
                raise ValueError("compiled REGIME_BOUNDARY must not retain latent fields")
        return self

    epistemic_status: Literal["hypothesis_only"] = "hypothesis_only"
    requires_verification: Literal[True] = True
    shadow_only: Literal[True] = True
    scientific_quality_ranking_performed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    production_selection_authority: Literal[False] = False


class ReframeOperatorRunRecord(StrictModel):
    operator_id: ImplementedReframeOperatorId
    readiness_status: str
    decision: Literal[
        "generated",
        "abstained",
        "rejected_invalid_draft",
        "generation_failed",
        "skipped_not_ready",
        "skipped_not_triggered",
    ]
    prompt_sha256: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    reasons: list[str] = Field(default_factory=list)
    rejected_candidate_count: int = Field(default=0, ge=0)
    compile_issues: list[str] = Field(default_factory=list)
    generation_error_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    response_id: str | None = None
    elapsed_seconds: float | None = None


class ScientificReframingShadowReport(StrictModel):
    schema_version: Literal[
        "scientific-reframing-shadow-report-v1"
    ] = "scientific-reframing-shadow-report-v1"

    report_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)
    backend_name: str
    model_name: str
    runs: list[ReframeOperatorRunRecord]
    candidates: list[ScientificReframeCandidate]
    llm_calls_performed: int = Field(ge=0)
    source_trigger_report_id: str | None = None

    shadow_only: Literal[True] = True
    scientific_trigger_evaluated: bool = False
    reframe_quality_evaluated: Literal[False] = False
    scientific_quality_ranking_performed: Literal[False] = False
    external_novelty_evaluated: Literal[False] = False
    n10_run: Literal[False] = False
    production_selection_changed: Literal[False] = False
    legacy_portfolio_mutated: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    novelty_authority: Literal[False] = False
    positive_premise_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_run_lineage(self) -> "ScientificReframingShadowReport":
        candidate_ids = [row.reframe_id for row in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("reframe candidate IDs must be unique")
        emitted = {
            candidate_id
            for run in self.runs
            for candidate_id in run.candidate_ids
        }
        if emitted != set(candidate_ids):
            raise ValueError("run candidate_ids must equal report candidates")
        if self.llm_calls_performed != sum(
            run.decision not in {"skipped_not_ready", "skipped_not_triggered"}
            for run in self.runs
        ):
            raise ValueError("llm_calls_performed must match executed operator runs")
        if self.source_trigger_report_id is None and self.scientific_trigger_evaluated:
            raise ValueError(
                "scientific_trigger_evaluated requires source_trigger_report_id"
            )
        if self.source_trigger_report_id is not None and not self.scientific_trigger_evaluated:
            raise ValueError(
                "source_trigger_report_id requires scientific_trigger_evaluated=true"
            )
        return self


def stable_reframe_id(
    *,
    task_id: str,
    context_sha256: str,
    operator_id: str,
    draft: ScientificReframeDraft,
) -> str:
    payload = json.dumps(
        draft.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        "|".join(
            [task_id, context_sha256, operator_id, payload]
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"scientific_reframe:{digest}"
