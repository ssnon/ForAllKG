from __future__ import annotations

import unicodedata
from typing import Literal


SourceReferenceStatus = Literal["READY", "INCOMPLETE", "INVALID"]


def _surface(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def _normalized(value: object) -> str:
    return _surface(value).casefold()


def resolve_atomic_source_reference(
    *,
    hypothesis: object,
    prediction_observation_id: object,
    falsification_criterion_id: object,
    expected_observable: object | None = None,
    expected_falsification_condition: object | None = None,
) -> tuple[
    SourceReferenceStatus,
    list[str],
    object | None,
    object | None,
]:
    """Resolve stable prediction/falsifier source IDs without text reconstruction.

    Optional expected surfaces are integrity checks only. They do not establish
    source identity; identity comes from the stable IDs.
    """

    prediction_id = _surface(prediction_observation_id)
    falsifier_id = _surface(falsification_criterion_id)
    reasons: list[str] = []

    if not prediction_id:
        reasons.append("missing_prediction_source_id")
    if not falsifier_id:
        reasons.append("missing_falsifier_source_id")
    if reasons:
        return "INCOMPLETE", reasons, None, None

    predictions = [
        row
        for row in getattr(hypothesis, "predicted_observations", [])
        if getattr(row, "observation_id", None) == prediction_id
    ]
    falsifiers = [
        row
        for row in getattr(hypothesis, "falsification_criteria", [])
        if getattr(row, "criterion_id", None) == falsifier_id
    ]

    if len(predictions) != 1:
        reasons.append(
            "prediction_source_id_cardinality:" + str(len(predictions))
        )
    if len(falsifiers) != 1:
        reasons.append(
            "falsifier_source_id_cardinality:" + str(len(falsifiers))
        )
    if reasons:
        return "INVALID", reasons, None, None

    prediction = predictions[0]
    falsifier = falsifiers[0]
    prediction_observable = _normalized(
        getattr(prediction, "observable", "")
    )
    falsifier_observable = _normalized(
        getattr(falsifier, "observable", "")
    )

    if not prediction_observable:
        reasons.append("source_observable_empty")
    elif prediction_observable != falsifier_observable:
        reasons.append("source_observable_identity_mismatch")

    if expected_observable is not None:
        expected = _normalized(expected_observable)
        if not expected:
            reasons.append("expected_source_observable_empty")
        elif prediction_observable and expected != prediction_observable:
            reasons.append("expected_source_observable_mismatch")

    if expected_falsification_condition is not None:
        expected_falsifier = _normalized(
            expected_falsification_condition
        )
        source_falsifier = _normalized(
            getattr(falsifier, "falsifying_outcome", "")
        )
        if not expected_falsifier:
            reasons.append("expected_falsification_condition_empty")
        elif expected_falsifier != source_falsifier:
            reasons.append(
                "expected_falsification_condition_mismatch"
            )

    if reasons:
        return "INVALID", list(dict.fromkeys(reasons)), prediction, falsifier
    return "READY", [], prediction, falsifier


__all__ = [
    "SourceReferenceStatus",
    "resolve_atomic_source_reference",
]
