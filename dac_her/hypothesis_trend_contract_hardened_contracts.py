from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from dac_her.hypothesis_contracts import (
    ExpectedDirection,
    HypothesisType,
)
from dac_her.hypothesis_trend_contracts import (
    StrictModel,
    TrendReferenceDraft,
)


HYPOTHESIS_TREND_HARDENED_CONTRACT_SEMANTICS_ID = (
    "hypothesis_trend_contract_hardened_v1_alpha4c5i"
)

PredictionKind = Literal["trend_bound", "exploratory"]


class ContractHardenedPredictedObservationDraft(StrictModel):
    local_id: str = Field(min_length=1)
    prediction_kind: PredictionKind

    # For trend_bound predictions, the LLM selects only source Trend IDs.
    # Observable and direction are compiler-owned.
    trend_view_ids: list[str] = Field(default_factory=list)

    # Used only for exploratory predictions that are not Trend-bound.
    exploratory_observable: str | None = None
    exploratory_expected_direction: ExpectedDirection | None = None

    # The LLM may propose a mechanism/rationale, but not restate a bound
    # Trend direction using its own IV-direction wording.
    mechanistic_rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _prediction_consistency(
        self,
    ) -> "ContractHardenedPredictedObservationDraft":
        ids = [str(value).strip() for value in self.trend_view_ids]
        if any(not value for value in ids):
            raise ValueError("trend_view_ids may not contain empty IDs")
        if ids != sorted(set(ids)):
            raise ValueError("trend_view_ids must be sorted and unique")

        if self.prediction_kind == "trend_bound":
            if not ids:
                raise ValueError(
                    "trend_bound prediction requires at least one trend_view_id"
                )
            if self.exploratory_observable is not None:
                raise ValueError(
                    "trend_bound prediction may not set exploratory_observable"
                )
            if self.exploratory_expected_direction is not None:
                raise ValueError(
                    "trend_bound prediction may not set "
                    "exploratory_expected_direction"
                )
        else:
            if ids:
                raise ValueError(
                    "exploratory prediction may not bind Trend view IDs"
                )
            if not (self.exploratory_observable or "").strip():
                raise ValueError(
                    "exploratory prediction requires exploratory_observable"
                )
            if self.exploratory_expected_direction is None:
                raise ValueError(
                    "exploratory prediction requires "
                    "exploratory_expected_direction"
                )
        return self


class ContractHardenedFalsificationCriterionDraft(StrictModel):
    local_id: str = Field(min_length=1)
    prediction_local_id: str = Field(min_length=1)
    falsifying_outcome: str = Field(min_length=1)


class ContractHardenedTrendHypothesisProposalDraft(StrictModel):
    local_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    hypothesis_type: HypothesisType

    premise_statement_ids: list[str] = Field(default_factory=list)
    gap_statement_ids: list[str] = Field(default_factory=list)
    trend_references: list[TrendReferenceDraft] = Field(default_factory=list)

    # Scientific synthesis remains LLM-owned.
    mechanistic_proposal: str = Field(min_length=1)
    inferential_bridge: str = Field(min_length=1)

    predicted_observations: list[
        ContractHardenedPredictedObservationDraft
    ] = Field(min_length=1)
    falsification_criteria: list[
        ContractHardenedFalsificationCriterionDraft
    ] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(
        self,
    ) -> "ContractHardenedTrendHypothesisProposalDraft":
        prediction_ids = [
            row.local_id for row in self.predicted_observations
        ]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("duplicate predicted_observation local_id")

        falsifier_ids = [
            row.local_id for row in self.falsification_criteria
        ]
        if len(falsifier_ids) != len(set(falsifier_ids)):
            raise ValueError("duplicate falsification_criterion local_id")

        known_prediction_ids = set(prediction_ids)
        for row in self.falsification_criteria:
            if row.prediction_local_id not in known_prediction_ids:
                raise ValueError(
                    "falsification criterion references unknown "
                    "prediction_local_id"
                )

        view_ids = [row.view_id for row in self.trend_references]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("duplicate Trend input view_id in one hypothesis")

        positive = any(
            row.use_role
            in {
                "positive_empirical_support",
                "cross_paper_empirical_support",
            }
            for row in self.trend_references
        )
        if not self.premise_statement_ids and not positive:
            raise ValueError(
                "A hypothesis requires at least one positive Explorer "
                "premise or positive Trend reference."
            )
        return self


class ContractHardenedTrendHypothesisPortfolioDraft(StrictModel):
    hypotheses: list[
        ContractHardenedTrendHypothesisProposalDraft
    ] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _abstention_consistency(
        self,
    ) -> "ContractHardenedTrendHypothesisPortfolioDraft":
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
                "abstention_reason must be null when hypotheses are proposed"
            )
        ids = [row.local_id for row in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate hypothesis local_id")
        return self
