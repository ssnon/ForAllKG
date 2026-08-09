from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FinalDisposition = Literal[
    "requires_validation_design",
    "ready_for_experimental_validation",
    "ready_for_high_fidelity_computation",
    "conditional_candidate",
    "high_complexity_candidate",
    "low_precedent_candidate",
    "human_review_required",
    "insufficient_information",
    "rejected_physical",
    "rejected_experimental",
]


class CandidateDecisionCard(StrictModel):
    schema_version: Literal["candidate-decision-card-v02"] = "candidate-decision-card-v02"
    decision_id: str
    hypothesis_id: str
    hypothesis_statement: str

    source_intake_id: str
    source_scope_id: str
    source_validation_specification_id: str
    source_physics_report_id: str
    source_experimental_report_id: str

    catalyst_class: str
    hypothesis_level: str
    validation_strategy: str
    requires_candidate_concretization: bool

    semantic_status: str
    physics_disposition: str
    experimental_disposition: str
    final_disposition: FinalDisposition

    key_uncertainties: list[str] = Field(default_factory=list)
    required_computations: list[str] = Field(default_factory=list)
    candidate_concretization_requirements: list[str] = Field(default_factory=list)
    required_comparisons: list[str] = Field(default_factory=list)
    not_applicable_physics_checks: list[str] = Field(default_factory=list)
    required_synthesis_capabilities: list[str] = Field(default_factory=list)
    required_characterization: list[str] = Field(default_factory=list)
    required_electrochemical_tests: list[str] = Field(default_factory=list)
    not_applicable_experimental_capabilities: list[str] = Field(default_factory=list)

    synthesis_complexity: str
    characterization_complexity: str
    relative_cost_burden: str
    relative_effort_burden: str
    novelty_status: Literal["not_assessed"] = "not_assessed"


class CandidateDecisionPortfolio(StrictModel):
    schema_version: Literal["candidate-decision-portfolio-v02"] = (
        "candidate-decision-portfolio-v02"
    )
    decision_portfolio_id: str
    source_intake_id: str
    cards: list[CandidateDecisionCard] = Field(default_factory=list)
    abstention_reason: str | None = None
