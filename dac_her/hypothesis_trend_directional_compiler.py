from __future__ import annotations

import hashlib
from typing import Any

from dac_her.hypothesis_trend_compiler import (
    HYPOTHESIS_TREND_COMPILER_SEMANTICS_ID,
    POSITIVE_USES,
    TrendAwareHypothesisCompiler,
    TrendHypothesisCompileError,
    TrendHypothesisCompileIssue,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareHypothesisPortfolioDraft,
    TrendAwareHypothesisProposalDraft,
    TrendAwarePredictedObservationDraft,
)
from dac_her.hypothesis_trend_directional_contracts import (
    CompiledTrendDirectionBinding,
    DirectionAwarePredictedObservation,
    DirectionAwareTrendHypothesisCard,
    DirectionAwareTrendHypothesisPortfolio,
    DirectionAwareTrendHypothesisPortfolioDraft,
    HYPOTHESIS_TREND_DIRECTIONAL_CONTRACT_SEMANTICS_ID,
    TrendDependentChange,
    canonical_dependent_change,
    expected_prediction_direction,
)
from dac_her.hypothesis_trend_input import (
    HypothesisTrendInputView,
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)


HYPOTHESIS_TREND_DIRECTIONAL_COMPILER_SEMANTICS_ID = (
    "hypothesis_trend_directional_compiler_v1_alpha4c5c1"
)


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _base_draft(
    draft: DirectionAwareTrendHypothesisPortfolioDraft,
) -> TrendAwareHypothesisPortfolioDraft:
    hypotheses: list[TrendAwareHypothesisProposalDraft] = []
    for row in draft.hypotheses:
        payload = row.model_dump(mode="json")
        predictions = []
        for prediction in payload["predicted_observations"]:
            prediction = dict(prediction)
            prediction.pop("trend_direction_bindings", None)
            predictions.append(
                TrendAwarePredictedObservationDraft.model_validate(
                    prediction
                )
            )
        payload["predicted_observations"] = predictions
        hypotheses.append(
            TrendAwareHypothesisProposalDraft.model_validate(payload)
        )
    return TrendAwareHypothesisPortfolioDraft(
        hypotheses=hypotheses,
        abstention_reason=draft.abstention_reason,
    )


def _positive_view_ids(hypothesis: object) -> set[str]:
    return {
        row.view_id
        for row in getattr(hypothesis, "trend_references", [])
        if row.use_role in POSITIVE_USES
    }


def _prediction_required_direction(
    changes: list[TrendDependentChange],
) -> str:
    return expected_prediction_direction(changes)


class DirectionAwareTrendHypothesisCompiler:
    semantics_id = HYPOTHESIS_TREND_DIRECTIONAL_COMPILER_SEMANTICS_ID
    directional_contract_semantics_id = (
        HYPOTHESIS_TREND_DIRECTIONAL_CONTRACT_SEMANTICS_ID
    )
    base_compiler_semantics_id = HYPOTHESIS_TREND_COMPILER_SEMANTICS_ID

    def __init__(
        self,
        *,
        base_compiler: TrendAwareHypothesisCompiler | None = None,
    ) -> None:
        self.base_compiler = base_compiler or TrendAwareHypothesisCompiler()

    def _validate_direction_bindings(
        self,
        source: TrendAwareHypothesisInput,
        draft: DirectionAwareTrendHypothesisPortfolioDraft,
    ) -> list[TrendHypothesisCompileIssue]:
        view_index: dict[str, HypothesisTrendInputView] = {
            row.view_id: row for row in source.trend_views
        }
        issues: list[TrendHypothesisCompileIssue] = []

        for h_index, hypothesis in enumerate(draft.hypotheses):
            base = f"draft.hypotheses[{h_index}]"
            positive_ids = _positive_view_ids(hypothesis)
            bound_ids: set[str] = set()

            for p_index, prediction in enumerate(
                hypothesis.predicted_observations
            ):
                ploc = (
                    base
                    + f".predicted_observations[{p_index}]"
                )
                expected_changes: list[TrendDependentChange] = []

                for b_index, binding in enumerate(
                    prediction.trend_direction_bindings
                ):
                    bloc = (
                        ploc
                        + f".trend_direction_bindings[{b_index}]"
                    )
                    view = view_index.get(binding.view_id)
                    if view is None:
                        issues.append(
                            TrendHypothesisCompileIssue(
                                code="UNKNOWN_TREND_DIRECTION_VIEW",
                                location=bloc + ".view_id",
                                message=(
                                    "Unknown Trend view used in direction "
                                    f"binding: {binding.view_id}"
                                ),
                            )
                        )
                        continue
                    if binding.view_id not in positive_ids:
                        issues.append(
                            TrendHypothesisCompileIssue(
                                code=(
                                    "TREND_DIRECTION_BINDING_NOT_"
                                    "POSITIVE_SUPPORT"
                                ),
                                location=bloc + ".view_id",
                                message=(
                                    "Direction bindings may reference only "
                                    "Trend views selected as positive "
                                    "empirical support in the same hypothesis."
                                ),
                            )
                        )
                        continue

                    expected = canonical_dependent_change(
                        view.directions
                    )
                    if binding.dependent_change != expected:
                        issues.append(
                            TrendHypothesisCompileIssue(
                                code="TREND_DIRECTION_BINDING_MISMATCH",
                                location=bloc + ".dependent_change",
                                message=(
                                    f"Trend view {view.view_id} has "
                                    f"directions={view.directions!r}; under "
                                    "the canonical independent-variable "
                                    "increase frame it requires "
                                    f"dependent_change={expected!r}, not "
                                    f"{binding.dependent_change!r}."
                                ),
                            )
                        )
                    else:
                        expected_changes.append(expected)
                        bound_ids.add(binding.view_id)

                if prediction.trend_direction_bindings:
                    required = _prediction_required_direction(
                        expected_changes
                    )
                    if prediction.expected_direction != required:
                        issues.append(
                            TrendHypothesisCompileIssue(
                                code=(
                                    "TREND_PREDICTION_DIRECTION_"
                                    "MISMATCH"
                                ),
                                location=ploc + ".expected_direction",
                                message=(
                                    "The prediction expected_direction must "
                                    "match its canonical Trend direction "
                                    f"bindings: expected {required!r}, got "
                                    f"{prediction.expected_direction!r}."
                                ),
                            )
                        )

            for view_id in sorted(positive_ids - bound_ids):
                issues.append(
                    TrendHypothesisCompileIssue(
                        code="MISSING_TREND_DIRECTION_BINDING",
                        location=base + ".predicted_observations",
                        message=(
                            "Every selected positive Trend support view must "
                            "be bound to at least one predicted observation "
                            "using the canonical independent-variable "
                            f"increase frame. Missing view: {view_id}"
                        ),
                    )
                )

        return issues

    def compile(
        self,
        source: TrendAwareHypothesisInput,
        draft: DirectionAwareTrendHypothesisPortfolioDraft,
    ) -> DirectionAwareTrendHypothesisPortfolio:
        verify_trend_aware_input_sources(source)
        issues = self._validate_direction_bindings(source, draft)
        if issues:
            raise TrendHypothesisCompileError(issues)

        base_portfolio = self.base_compiler.compile(
            source,
            _base_draft(draft),
        )
        view_index = {
            row.view_id: row for row in source.trend_views
        }

        cards: list[DirectionAwareTrendHypothesisCard] = []
        for draft_h, base_card in zip(
            draft.hypotheses,
            base_portfolio.hypotheses,
            strict=True,
        ):
            predictions: list[DirectionAwarePredictedObservation] = []
            for draft_p, base_p in zip(
                draft_h.predicted_observations,
                base_card.predicted_observations,
                strict=True,
            ):
                compiled_bindings: list[
                    CompiledTrendDirectionBinding
                ] = []
                for binding in draft_p.trend_direction_bindings:
                    view = view_index[binding.view_id]
                    expected = canonical_dependent_change(
                        view.directions
                    )
                    compiled_bindings.append(
                        CompiledTrendDirectionBinding(
                            binding_id=_stable_id(
                                "trend_direction_binding",
                                source.input_sha256,
                                base_card.hypothesis_id,
                                base_p.observation_id,
                                view.view_id,
                                binding.independent_change,
                                binding.dependent_change,
                            ),
                            view_id=view.view_id,
                            grounding_id=view.grounding_id,
                            relation_id=view.relation_id,
                            independent_variable_key=(
                                view.independent_variable_key
                            ),
                            dependent_observable_key=(
                                view.dependent_observable_key
                            ),
                            source_directions=list(view.directions),
                            source_shapes=list(view.shapes),
                            independent_change="increase",
                            dependent_change=binding.dependent_change,
                            expected_dependent_change=expected,
                            direction_consistent=True,
                        )
                    )

                compiled_bindings = sorted(
                    compiled_bindings,
                    key=lambda row: row.view_id,
                )
                binding_signature = ",".join(
                    f"{row.view_id}:{row.dependent_change}"
                    for row in compiled_bindings
                )
                predictions.append(
                    DirectionAwarePredictedObservation(
                        observation_id=_stable_id(
                            "direction_aware_trend_prediction",
                            base_p.observation_id,
                            self.semantics_id,
                            binding_signature,
                        ),
                        observable=base_p.observable,
                        expected_direction=base_p.expected_direction,
                        rationale=base_p.rationale,
                        trend_direction_bindings=compiled_bindings,
                    )
                )

            binding_signature = ",".join(
                f"{binding.view_id}:{binding.dependent_change}"
                for prediction in predictions
                for binding in prediction.trend_direction_bindings
            )
            new_hypothesis_id = _stable_id(
                "direction_aware_trend_hypothesis",
                base_card.hypothesis_id,
                self.semantics_id,
                binding_signature,
            )
            payload: dict[str, Any] = base_card.model_dump(
                mode="json"
            )
            payload.update(
                {
                    "schema_version":
                        "direction-aware-trend-hypothesis-card-v1",
                    "hypothesis_id": new_hypothesis_id,
                    "base_hypothesis_id":
                        base_card.hypothesis_id,
                    "directional_contract_semantics_id":
                        self.directional_contract_semantics_id,
                    "predicted_observations": [
                        row.model_dump(mode="json")
                        for row in predictions
                    ],
                }
            )
            cards.append(
                DirectionAwareTrendHypothesisCard.model_validate(
                    payload
                )
            )

        portfolio_id = _stable_id(
            "direction_aware_trend_hypothesis_portfolio",
            base_portfolio.portfolio_id,
            self.semantics_id,
            ",".join(row.hypothesis_id for row in cards),
        )
        payload = base_portfolio.model_dump(mode="json")
        payload.update(
            {
                "schema_version":
                    "direction-aware-trend-hypothesis-portfolio-v1",
                "portfolio_id": portfolio_id,
                "base_portfolio_id":
                    base_portfolio.portfolio_id,
                "directional_contract_semantics_id":
                    self.directional_contract_semantics_id,
                "directional_compiler_semantics_id":
                    self.semantics_id,
                "base_compiler_semantics_id":
                    self.base_compiler_semantics_id,
                "hypotheses": [
                    row.model_dump(mode="json")
                    for row in cards
                ],
            }
        )
        return DirectionAwareTrendHypothesisPortfolio.model_validate(
            payload
        )
