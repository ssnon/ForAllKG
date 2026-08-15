from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.hypothesis_contracts import (
    ExpectedDirection,
    HypothesisType,
)


HYPOTHESIS_TREND_REFERENCE_CONTRACT_SEMANTICS_ID = (
    "hypothesis_trend_reference_contract_v1_alpha4c5c"
)

TrendReferenceUse = Literal[
    "positive_empirical_support",
    "cross_paper_empirical_support",
    "context_qualification",
    "counterevidence_boundary",
    "replication_gap",
]

VerificationDependency = Literal[
    "none",
    "supporting",
    "essential",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrendReferenceDraft(StrictModel):
    view_id: str = Field(min_length=1)
    use_role: TrendReferenceUse


class TrendAwarePredictedObservationDraft(StrictModel):
    local_id: str
    observable: str = Field(min_length=1)
    expected_direction: ExpectedDirection
    rationale: str = Field(min_length=1)


class TrendAwareFalsificationCriterionDraft(StrictModel):
    local_id: str
    observable: str = Field(min_length=1)
    falsifying_outcome: str = Field(min_length=1)


class TrendAwareHypothesisProposalDraft(StrictModel):
    local_id: str
    title: str = Field(min_length=1)
    hypothesis_statement: str = Field(min_length=1)
    hypothesis_type: HypothesisType

    # Explorer-report namespace.  Unlike the legacy draft, this can be empty
    # when Trend supplies the positive empirical support.
    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)

    # Separate Trend input-view namespace.
    trend_references: list[TrendReferenceDraft] = Field(
        default_factory=list
    )

    inferential_bridge: str = Field(min_length=1)
    predicted_observations: list[
        TrendAwarePredictedObservationDraft
    ] = Field(min_length=1)
    falsification_criteria: list[
        TrendAwareFalsificationCriterionDraft
    ] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(
        self,
    ) -> "TrendAwareHypothesisProposalDraft":
        prediction_ids = [
            row.local_id for row in self.predicted_observations
        ]
        falsifier_ids = [
            row.local_id for row in self.falsification_criteria
        ]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError(
                "duplicate predicted_observation local_id"
            )
        if len(falsifier_ids) != len(set(falsifier_ids)):
            raise ValueError(
                "duplicate falsification_criterion local_id"
            )

        view_ids = [
            row.view_id for row in self.trend_references
        ]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError(
                "duplicate Trend input view_id in one hypothesis"
            )

        positive_trend = any(
            row.use_role
            in {
                "positive_empirical_support",
                "cross_paper_empirical_support",
            }
            for row in self.trend_references
        )
        if (
            not self.premise_statement_ids
            and not positive_trend
        ):
            raise ValueError(
                "A hypothesis requires at least one positive "
                "Explorer premise or positive Trend reference."
            )
        return self


class TrendAwareHypothesisPortfolioDraft(StrictModel):
    hypotheses: list[
        TrendAwareHypothesisProposalDraft
    ] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _abstention_consistency(
        self,
    ) -> "TrendAwareHypothesisPortfolioDraft":
        if (
            not self.hypotheses
            and not (self.abstention_reason or "").strip()
        ):
            raise ValueError(
                "abstention_reason is required when no hypotheses "
                "are proposed"
            )
        if self.hypotheses and self.abstention_reason is not None:
            raise ValueError(
                "abstention_reason must be null when hypotheses "
                "are proposed"
            )
        ids = [row.local_id for row in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate hypothesis local_id")
        return self


class TrendAwarePredictedObservation(StrictModel):
    observation_id: str
    observable: str
    expected_direction: ExpectedDirection
    rationale: str


class TrendAwareFalsificationCriterion(StrictModel):
    criterion_id: str
    observable: str
    falsifying_outcome: str


class CompiledTrendReference(StrictModel):
    reference_id: str
    view_id: str
    grounding_id: str
    relation_id: str
    lane: str
    use_role: TrendReferenceUse
    cross_context_status: str
    paper_ids: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    shapes: list[str] = Field(default_factory=list)
    requires_context_qualification: bool
    requires_verification: bool
    association_only: bool
    directional_cross_paper_premise_allowed: bool

    # These flags are provenance authorization flags, not statements about
    # whether the generated hypothesis itself may contain a proposed causal
    # mechanism in its inferential bridge.
    trend_causal_authorization: Literal[False] = False
    trend_universal_authorization: Literal[False] = False


class TrendAwareHypothesisEvidenceProfile(StrictModel):
    explorer_premise_count: int
    explorer_gap_count: int
    trend_reference_count: int
    trend_positive_support_count: int
    trend_cross_paper_support_count: int
    trend_context_qualification_count: int
    trend_counterevidence_count: int
    trend_gap_count: int
    support_paper_count: int
    verification_required_support_count: int
    association_only_support_count: int
    reported_explorer_premise_count: int
    synthesis_explorer_premise_count: int


class TrendAwareHypothesisCard(StrictModel):
    schema_version: Literal[
        "trend-aware-hypothesis-card-v1"
    ] = "trend-aware-hypothesis-card-v1"

    hypothesis_id: str
    domain_profile_id: str

    source_context_id: str
    source_context_sha256: str
    source_report_id: str
    source_report_sha256: str
    source_trend_input_id: str
    source_trend_input_sha256: str

    title: str
    hypothesis_statement: str
    hypothesis_type: HypothesisType

    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    trend_references: list[CompiledTrendReference] = Field(
        default_factory=list
    )

    inferential_bridge: str
    predicted_observations: list[
        TrendAwarePredictedObservation
    ]
    falsification_criteria: list[
        TrendAwareFalsificationCriterion
    ]
    assumptions: list[str] = Field(default_factory=list)

    explorer_source_paper_ids: list[str] = Field(
        default_factory=list
    )
    trend_positive_source_paper_ids: list[str] = Field(
        default_factory=list
    )
    support_paper_ids: list[str] = Field(default_factory=list)

    explorer_gap_paper_ids: list[str] = Field(
        default_factory=list
    )
    trend_gap_paper_ids: list[str] = Field(
        default_factory=list
    )
    context_and_counterevidence_paper_ids: list[str] = Field(
        default_factory=list
    )

    cross_paper_synthesis: bool = False
    verification_dependency: VerificationDependency = "none"
    evidence_profile: TrendAwareHypothesisEvidenceProfile

    trend_causal_authorization: Literal[False] = False
    trend_universal_authorization: Literal[False] = False

    status: Literal["hypothesized"] = "hypothesized"
    novelty_status: Literal["not_assessed"] = "not_assessed"


class TrendAwareHypothesisPortfolio(StrictModel):
    schema_version: Literal[
        "trend-aware-hypothesis-portfolio-v1"
    ] = "trend-aware-hypothesis-portfolio-v1"

    portfolio_id: str
    domain_profile_id: str

    source_context_id: str
    source_context_sha256: str
    source_report_id: str
    source_report_sha256: str
    source_trend_input_id: str
    source_trend_input_sha256: str

    hypotheses: list[TrendAwareHypothesisCard] = Field(
        default_factory=list
    )
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _abstention_consistency(
        self,
    ) -> "TrendAwareHypothesisPortfolio":
        if (
            not self.hypotheses
            and not (self.abstention_reason or "").strip()
        ):
            raise ValueError(
                "abstention_reason is required when portfolio is empty"
            )
        if self.hypotheses and self.abstention_reason is not None:
            raise ValueError(
                "abstention_reason must be null when hypotheses exist"
            )
        return self
