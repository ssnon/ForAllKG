from __future__ import annotations

import re

from dac_her.hypothesis_trend_compiler import POSITIVE_USES
from dac_her.hypothesis_trend_contracts import (
    TrendAwareHypothesisCard,
    TrendAwareHypothesisPortfolio,
    TrendAwarePredictedObservation,
)
from dac_her.hypothesis_trend_directional_contracts import (
    DirectionAwareTrendHypothesisPortfolio,
    HYPOTHESIS_TREND_DIRECTIONAL_CONTRACT_SEMANTICS_ID,
    TrendDependentChange,
    canonical_dependent_change,
    expected_prediction_direction,
)
from dac_her.hypothesis_trend_input import (
    TrendAwareHypothesisInput,
    verify_trend_aware_input_sources,
)
from dac_her.hypothesis_trend_validator import (
    TrendAwareHypothesisValidator,
    TrendHypothesisValidationIssue,
    TrendHypothesisValidationResult,
)


HYPOTHESIS_TREND_DIRECTIONAL_VALIDATOR_SEMANTICS_ID = (
    "hypothesis_trend_directional_validator_v1_alpha4c5c1"
)

_DECREASE_MARKER = re.compile(
    r"\b(?:decreas\w*|reduc\w*|smaller|lower|less|shrink\w*)\b",
    re.I,
)


def _base_portfolio(
    portfolio: DirectionAwareTrendHypothesisPortfolio,
) -> TrendAwareHypothesisPortfolio:
    payload = portfolio.model_dump(mode="json")
    payload["schema_version"] = "trend-aware-hypothesis-portfolio-v1"
    payload.pop("base_portfolio_id", None)
    payload.pop("directional_contract_semantics_id", None)
    payload.pop("directional_compiler_semantics_id", None)
    payload.pop("base_compiler_semantics_id", None)

    hypotheses = []
    for card_payload in payload["hypotheses"]:
        card_payload = dict(card_payload)
        card_payload["schema_version"] = "trend-aware-hypothesis-card-v1"
        card_payload.pop("base_hypothesis_id", None)
        card_payload.pop("directional_contract_semantics_id", None)
        predictions = []
        for pred_payload in card_payload["predicted_observations"]:
            pred_payload = dict(pred_payload)
            pred_payload.pop("trend_direction_bindings", None)
            predictions.append(
                TrendAwarePredictedObservation.model_validate(
                    pred_payload
                ).model_dump(mode="json")
            )
        card_payload["predicted_observations"] = predictions
        hypotheses.append(
            TrendAwareHypothesisCard.model_validate(
                card_payload
            ).model_dump(mode="json")
        )
    payload["hypotheses"] = hypotheses
    return TrendAwareHypothesisPortfolio.model_validate(payload)


def _key_phrase(key: str) -> str:
    return " ".join(str(key).replace("_", " ").split()).lower()


def _uses_decrease_frame(
    text: str,
    independent_variable_key: str,
) -> bool:
    normalized = " ".join(str(text).lower().split())
    phrase = _key_phrase(independent_variable_key)
    if not phrase or phrase not in normalized:
        return False
    start = 0
    while True:
        index = normalized.find(phrase, start)
        if index < 0:
            return False
        left = max(0, index - 48)
        right = min(
            len(normalized),
            index + len(phrase) + 48,
        )
        if _DECREASE_MARKER.search(normalized[left:right]):
            return True
        start = index + len(phrase)


def _required_prediction_direction(
    changes: list[TrendDependentChange],
) -> str:
    return expected_prediction_direction(changes)


class DirectionAwareTrendHypothesisValidator:
    semantics_id = (
        HYPOTHESIS_TREND_DIRECTIONAL_VALIDATOR_SEMANTICS_ID
    )
    directional_contract_semantics_id = (
        HYPOTHESIS_TREND_DIRECTIONAL_CONTRACT_SEMANTICS_ID
    )

    def __init__(
        self,
        *,
        base_validator: TrendAwareHypothesisValidator | None = None,
    ) -> None:
        self.base_validator = (
            base_validator or TrendAwareHypothesisValidator()
        )

    def validate(
        self,
        source: TrendAwareHypothesisInput,
        portfolio: DirectionAwareTrendHypothesisPortfolio,
    ) -> TrendHypothesisValidationResult:
        verify_trend_aware_input_sources(source)
        base_result = self.base_validator.validate(
            source,
            _base_portfolio(portfolio),
        )
        issues = list(base_result.issues)
        views = {
            row.view_id: row for row in source.trend_views
        }

        def error(
            code: str,
            location: str,
            message: str,
        ) -> None:
            issues.append(
                TrendHypothesisValidationIssue(
                    severity="error",
                    code=code,
                    location=location,
                    message=message,
                )
            )

        if portfolio.directional_contract_semantics_id != (
            self.directional_contract_semantics_id
        ):
            error(
                "DIRECTIONAL_CONTRACT_SEMANTICS_MISMATCH",
                "portfolio.directional_contract_semantics_id",
                "Directional prediction contract semantics drifted.",
            )

        for h_index, card in enumerate(portfolio.hypotheses):
            hloc = f"hypotheses[{h_index}]"
            positive_ids = {
                row.view_id
                for row in card.trend_references
                if row.use_role in POSITIVE_USES
            }
            bound_ids: set[str] = set()

            for p_index, prediction in enumerate(
                card.predicted_observations
            ):
                ploc = (
                    hloc
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
                    view = views.get(binding.view_id)
                    if view is None:
                        error(
                            "UNKNOWN_COMPILED_TREND_DIRECTION_VIEW",
                            bloc + ".view_id",
                            binding.view_id,
                        )
                        continue
                    if binding.view_id not in positive_ids:
                        error(
                            "COMPILED_DIRECTION_BINDING_NOT_POSITIVE",
                            bloc + ".view_id",
                            (
                                "Compiled direction binding does not "
                                "correspond to positive Trend support."
                            ),
                        )
                        continue

                    expected = canonical_dependent_change(
                        view.directions
                    )
                    checks = {
                        "grounding_id": view.grounding_id,
                        "relation_id": view.relation_id,
                        "independent_variable_key":
                            view.independent_variable_key,
                        "dependent_observable_key":
                            view.dependent_observable_key,
                        "source_directions": list(view.directions),
                        "source_shapes": list(view.shapes),
                        "independent_change": "increase",
                        "expected_dependent_change": expected,
                        "dependent_change": expected,
                        "direction_consistent": True,
                    }
                    for field, wanted in checks.items():
                        actual = getattr(binding, field)
                        if actual != wanted:
                            error(
                                "COMPILED_DIRECTION_BINDING_MISMATCH",
                                bloc + f".{field}",
                                (
                                    f"expected={wanted!r}, "
                                    f"actual={actual!r}"
                                ),
                            )
                    expected_changes.append(expected)
                    bound_ids.add(binding.view_id)

                    text_scope = "\n".join(
                        [
                            card.title,
                            card.hypothesis_statement,
                            card.inferential_bridge,
                            prediction.observable,
                            prediction.rationale,
                        ]
                    )
                    if _uses_decrease_frame(
                        text_scope,
                        view.independent_variable_key,
                    ):
                        error(
                            "NONCANONICAL_TREND_DIRECTION_FRAME",
                            ploc,
                            (
                                "Trend-grounded directional language must "
                                "use the canonical independent-variable "
                                "increase frame. The generated text "
                                f"reframed {view.independent_variable_key!r} "
                                "using decrease/smaller/lower language, "
                                "which can invert the frozen Trend sign."
                            ),
                        )

                if prediction.trend_direction_bindings:
                    required = _required_prediction_direction(
                        expected_changes
                    )
                    if prediction.expected_direction != required:
                        error(
                            "COMPILED_TREND_PREDICTION_DIRECTION_MISMATCH",
                            ploc + ".expected_direction",
                            (
                                f"expected={required!r}, "
                                f"actual={prediction.expected_direction!r}"
                            ),
                        )

            for view_id in sorted(positive_ids - bound_ids):
                error(
                    "MISSING_COMPILED_TREND_DIRECTION_BINDING",
                    hloc + ".predicted_observations",
                    (
                        "Selected positive Trend support is not bound to "
                        f"any prediction: {view_id}"
                    ),
                )

        errors = sum(
            row.severity == "error" for row in issues
        )
        warnings = sum(
            row.severity == "warning" for row in issues
        )
        return TrendHypothesisValidationResult(
            semantics_id=self.semantics_id,
            passes=errors == 0,
            errors=errors,
            warnings=warnings,
            issues=issues,
        )
