from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dac_her.hypothesis_semantic_contracts import (
    SEMANTIC_DIMENSIONS,
    SemanticDimension,
    SemanticVerdict,
    StrictModel,
)


E2Scenario = Literal["candidate", "alignment", "partial", "abstention"]
E2ApprovalStatus = Literal["pending", "approved"]


class HypothesisE2ContextCase(StrictModel):
    case_id: str
    scenario: E2Scenario
    description: str
    context_path: str
    review_hint: str


class HypothesisE2ContextManifest(StrictModel):
    schema_version: Literal["hypothesis-e2-context-manifest-v262"] = (
        "hypothesis-e2-context-manifest-v262"
    )
    suite_id: str
    cases: list[HypothesisE2ContextCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> "HypothesisE2ContextManifest":
        ids = [row.case_id for row in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("E2 context manifest contains duplicate case_id")
        return self


class HypothesisE2OutputRecord(StrictModel):
    case_id: str
    scenario: E2Scenario
    description: str
    review_hint: str
    context_path: str
    portfolio_path: str
    run_path: str
    validation_path: str
    hard_evaluation_path: str
    prompt_path: str
    draft_paths: list[str] = Field(default_factory=list)
    generator_version: str
    prompt_version: str
    prompt_sha256: str
    context_file_sha256: str
    portfolio_file_sha256: str
    accepted: Literal[True] = True
    hard_gate_passed: Literal[True] = True
    scenario_postcondition_passed: Literal[True] = True
    abstained: bool


class HypothesisE2OutputManifest(StrictModel):
    schema_version: Literal["hypothesis-e2-output-manifest-v262"] = (
        "hypothesis-e2-output-manifest-v262"
    )
    suite_id: str
    cases: list[HypothesisE2OutputRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> "HypothesisE2OutputManifest":
        ids = [row.case_id for row in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("E2 output manifest contains duplicate case_id")
        return self


class HypothesisE2HumanDimension(StrictModel):
    dimension: SemanticDimension
    critic_verdict: SemanticVerdict
    critic_rationale: str = Field(min_length=1)
    human_allowed_verdicts: list[SemanticVerdict] = Field(default_factory=list)
    critical: bool = True
    human_note: str = ""


class HypothesisE2HumanReviewCase(StrictModel):
    case_id: str
    scenario: E2Scenario
    description: str
    review_hint: str
    source_kind: Literal["controlled_context_live_output"] = (
        "controlled_context_live_output"
    )
    context_path: str
    portfolio_path: str
    review_path: str
    generator_version: str
    approval_status: E2ApprovalStatus = "pending"
    dimensions: list[HypothesisE2HumanDimension]

    @model_validator(mode="after")
    def _complete_dimensions(self) -> "HypothesisE2HumanReviewCase":
        names = [row.dimension for row in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("E2 worksheet contains duplicate semantic dimensions")
        missing = sorted(set(SEMANTIC_DIMENSIONS) - set(names))
        extra = sorted(set(names) - set(SEMANTIC_DIMENSIONS))
        if missing or extra or len(names) != len(SEMANTIC_DIMENSIONS):
            raise ValueError(
                "E2 worksheet must contain all semantic dimensions exactly once; "
                f"missing={missing}, extra={extra}"
            )
        return self


class HypothesisE2HumanReviewWorksheet(StrictModel):
    schema_version: Literal["hypothesis-e2-human-review-worksheet-v262"] = (
        "hypothesis-e2-human-review-worksheet-v262"
    )
    suite_id: str
    cases: list[HypothesisE2HumanReviewCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cases(self) -> "HypothesisE2HumanReviewWorksheet":
        ids = [row.case_id for row in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("E2 worksheet contains duplicate case_id")
        return self
