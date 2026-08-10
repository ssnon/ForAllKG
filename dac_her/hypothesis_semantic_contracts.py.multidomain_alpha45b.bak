from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


SemanticDimension = Literal[
    "premise_fidelity",
    "gap_discipline",
    "candidate_calibration",
    "inferential_proportionality",
    "causal_strengthening",
    "directional_specificity",
    "prediction_linkage",
    "falsifier_informativeness",
    "cross_paper_discipline",
    "hypothesis_distinctness",
    "abstention_appropriateness",
]

SemanticVerdict = Literal["pass", "warning", "fail", "not_applicable"]

SEMANTIC_DIMENSIONS: tuple[str, ...] = (
    "premise_fidelity",
    "gap_discipline",
    "candidate_calibration",
    "inferential_proportionality",
    "causal_strengthening",
    "directional_specificity",
    "prediction_linkage",
    "falsifier_informativeness",
    "cross_paper_discipline",
    "hypothesis_distinctness",
    "abstention_appropriateness",
)


class HypothesisSemanticDimensionDraft(StrictModel):
    dimension: SemanticDimension
    verdict: SemanticVerdict
    rationale: str = Field(min_length=1)
    hypothesis_ids: list[str] = Field(default_factory=list)
    statement_ids: list[str] = Field(default_factory=list)


class HypothesisSemanticReviewDraft(StrictModel):
    dimensions: list[HypothesisSemanticDimensionDraft]
    overall_summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def _complete_unique_dimensions(self) -> "HypothesisSemanticReviewDraft":
        names = [row.dimension for row in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("semantic review contains duplicate dimensions")
        missing = sorted(set(SEMANTIC_DIMENSIONS) - set(names))
        extra = sorted(set(names) - set(SEMANTIC_DIMENSIONS))
        if missing or extra:
            raise ValueError(
                f"semantic review dimension mismatch; missing={missing}, extra={extra}"
            )
        if len(names) != len(SEMANTIC_DIMENSIONS):
            raise ValueError(
                f"semantic review must contain exactly {len(SEMANTIC_DIMENSIONS)} dimensions"
            )
        return self


class HypothesisSemanticReview(StrictModel):
    schema_version: Literal["hypothesis-semantic-review-v262"] = (
        "hypothesis-semantic-review-v262"
    )
    review_id: str
    source_context_id: str
    source_context_sha256: str
    source_portfolio_id: str
    source_portfolio_sha256: str
    source_evaluator_version: str
    source_hard_gate_passed: Literal[True] = True
    critic_prompt_version: str
    critic_prompt_sha256: str
    dimensions: list[HypothesisSemanticDimensionDraft]
    overall_summary: str


class HypothesisSemanticRunRecord(StrictModel):
    schema_version: Literal["hypothesis-semantic-run-v262"] = (
        "hypothesis-semantic-run-v262"
    )
    run_id: str
    context_id: str
    context_sha256: str
    portfolio_id: str
    portfolio_sha256: str
    hard_gate_passed: bool
    review_id: str | None = None
    critic_prompt_version: str
    critic_prompt_sha256: str
    backend: str
    model: str
    generated: bool
    accepted: bool
    failure_stage: Literal["none", "hard_gate", "generation", "review_validation"]
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None
    temperature: float | None = None
    backend_mode: str | None = None
    base_url: str | None = None
    parse_retries: int | None = None
