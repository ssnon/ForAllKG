from __future__ import annotations

from typing import Iterable, Literal

from pydantic import Field, model_validator

from dac_her.hypothesis_trend_contracts import (
    StrictModel,
    TrendAwareFalsificationCriterionDraft,
    TrendAwareHypothesisCard,
    TrendAwareHypothesisPortfolio,
    TrendAwareHypothesisPortfolioDraft,
    TrendAwareHypothesisProposalDraft,
    TrendAwarePredictedObservation,
    TrendAwarePredictedObservationDraft,
)


HYPOTHESIS_TREND_DIRECTIONAL_CONTRACT_SEMANTICS_ID = (
    "hypothesis_trend_directional_prediction_contract_v1_alpha4c5c1"
)

TrendIndependentChange = Literal["increase"]
TrendDependentChange = Literal[
    "increase",
    "decrease",
    "unchanged",
    "non_monotonic",
    "unspecified",
]


def canonical_dependent_change(
    directions: Iterable[str],
) -> TrendDependentChange:
    values = sorted({
        str(value).strip()
        for value in directions
        if str(value).strip()
    })
    if len(values) != 1:
        return "unspecified"
    return {
        "positive": "increase",
        "negative": "decrease",
        "unchanged": "unchanged",
        "non_monotonic": "non_monotonic",
        "unspecified": "unspecified",
    }.get(values[0], "unspecified")  # type: ignore[return-value]


def expected_prediction_direction(
    changes: Iterable[TrendDependentChange],
) -> str:
    mapped = {
        {
            "increase": "increase",
            "decrease": "decrease",
            "unchanged": "unspecified",
            "non_monotonic": "non_monotonic",
            "unspecified": "unspecified",
        }[value]
        for value in changes
    }
    if not mapped:
        return "unspecified"
    if len(mapped) != 1:
        return "unspecified"
    return next(iter(mapped))


class TrendDirectionBindingDraft(StrictModel):
    view_id: str = Field(min_length=1)
    independent_change: TrendIndependentChange = "increase"
    dependent_change: TrendDependentChange


class DirectionAwarePredictedObservationDraft(
    TrendAwarePredictedObservationDraft
):
    trend_direction_bindings: list[
        TrendDirectionBindingDraft
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def _direction_binding_consistency(
        self,
    ) -> "DirectionAwarePredictedObservationDraft":
        ids = [row.view_id for row in self.trend_direction_bindings]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate Trend direction binding view_id in one prediction"
            )
        return self


class DirectionAwareTrendHypothesisProposalDraft(
    TrendAwareHypothesisProposalDraft
):
    predicted_observations: list[
        DirectionAwarePredictedObservationDraft
    ] = Field(min_length=1)


class DirectionAwareTrendHypothesisPortfolioDraft(
    TrendAwareHypothesisPortfolioDraft
):
    hypotheses: list[
        DirectionAwareTrendHypothesisProposalDraft
    ] = Field(default_factory=list)


class CompiledTrendDirectionBinding(StrictModel):
    binding_id: str
    view_id: str
    grounding_id: str
    relation_id: str

    independent_variable_key: str
    dependent_observable_key: str
    source_directions: list[str] = Field(default_factory=list)
    source_shapes: list[str] = Field(default_factory=list)

    independent_change: TrendIndependentChange = "increase"
    dependent_change: TrendDependentChange
    expected_dependent_change: TrendDependentChange
    direction_consistent: Literal[True] = True


class DirectionAwarePredictedObservation(
    TrendAwarePredictedObservation
):
    trend_direction_bindings: list[
        CompiledTrendDirectionBinding
    ] = Field(default_factory=list)


class DirectionAwareTrendHypothesisCard(
    TrendAwareHypothesisCard
):
    schema_version: Literal[
        "direction-aware-trend-hypothesis-card-v1"
    ] = "direction-aware-trend-hypothesis-card-v1"

    base_hypothesis_id: str
    directional_contract_semantics_id: str = (
        HYPOTHESIS_TREND_DIRECTIONAL_CONTRACT_SEMANTICS_ID
    )

    predicted_observations: list[
        DirectionAwarePredictedObservation
    ]


class DirectionAwareTrendHypothesisPortfolio(
    TrendAwareHypothesisPortfolio
):
    schema_version: Literal[
        "direction-aware-trend-hypothesis-portfolio-v1"
    ] = "direction-aware-trend-hypothesis-portfolio-v1"

    base_portfolio_id: str
    directional_contract_semantics_id: str = (
        HYPOTHESIS_TREND_DIRECTIONAL_CONTRACT_SEMANTICS_ID
    )
    directional_compiler_semantics_id: str
    base_compiler_semantics_id: str

    hypotheses: list[
        DirectionAwareTrendHypothesisCard
    ] = Field(default_factory=list)
