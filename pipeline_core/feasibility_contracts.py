from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SemanticGateStatus = Literal[
    "eligible",
    "eligible_with_warnings",
    "human_review_required",
]


class FeasibilityPremise(StrictModel):
    statement_id: str
    text: str
    epistemic_role: str
    claim_kind: str
    paper_ids: list[str] = Field(default_factory=list)
    requires_verification: bool = False


class FeasibilityPrediction(StrictModel):
    observation_id: str
    observable: str
    expected_direction: str
    rationale: str


class FeasibilityFalsifier(StrictModel):
    criterion_id: str
    observable: str
    falsifying_outcome: str


class FeasibilityHypothesis(StrictModel):
    hypothesis_id: str
    title: str
    statement: str
    hypothesis_type: str
    inferential_bridge: str
    assumptions: list[str] = Field(default_factory=list)

    premises: list[FeasibilityPremise] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    candidate_dependency: str

    predictions: list[FeasibilityPrediction] = Field(default_factory=list)
    falsifiers: list[FeasibilityFalsifier] = Field(default_factory=list)

    semantic_gate_status: SemanticGateStatus
    semantic_warning_dimensions: list[str] = Field(default_factory=list)
    semantic_fail_dimensions: list[str] = Field(default_factory=list)


class FeasibilityIntake(StrictModel):
    schema_version: Literal["feasibility-intake-v0"] = "feasibility-intake-v0"
    intake_id: str
    intake_sha256: str

    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_portfolio_sha256: str
    source_semantic_review_id: str

    task_id: str
    question: str
    corpus_id: str
    abstention_reason: str | None = None

    hypotheses: list[FeasibilityHypothesis] = Field(default_factory=list)
