from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import inspect

import pytest

from pipeline_core.discovery.discovery_axis_context_runtime import (
    AxisContextReviewRecord,
    validate_context_review_shape,
)
from pipeline_core.discovery.discovery_axis_runtime import (
    AcceptedAxisDraft,
    DiscoveryAxisSynthesisOutcome,
    DiscoveryAxisSynthesisRuntime,
)


@dataclass(frozen=True)
class _Axis:
    axis_id: str = "axis:test"


@dataclass(frozen=True)
class _Card:
    hypothesis_id: str = "hypothesis:test"


@dataclass(frozen=True)
class _Review:
    review_id: str = "context-review:test"
    hypothesis_id: str = "hypothesis:test"
    status: str = "reframe_required"


def test_context_shape_accepts_reframe_as_observation() -> None:
    assert (
        validate_context_review_shape(
            axis=_Axis(),  # type: ignore[arg-type]
            card=_Card(),
            review=_Review(),
        )
        == "reframe_required"
    )


def test_context_shape_fails_closed_on_wrong_hypothesis() -> None:
    with pytest.raises(
        RuntimeError,
        match="hypothesis_id mismatch",
    ):
        validate_context_review_shape(
            axis=_Axis(),  # type: ignore[arg-type]
            card=_Card(),
            review=_Review(
                hypothesis_id="hypothesis:wrong"
            ),
        )


def test_context_capability_is_optional_for_legacy_callers() -> None:
    signature = inspect.signature(
        DiscoveryAxisSynthesisRuntime.__init__
    )

    assert (
        signature.parameters[
            "context_reviewer"
        ].default
        is None
    )

    assert (
        AcceptedAxisDraft
        .__dataclass_fields__[
            "context_review"
        ].default
        is None
    )

    fields = (
        DiscoveryAxisSynthesisOutcome
        .__dataclass_fields__
    )

    assert (
        fields[
            "context_reviews"
        ].default
        == ()
    )

    assert (
        fields[
            "context_review_history"
        ].default
        == ()
    )


def test_context_history_record_preserves_stage() -> None:
    review = _Review()

    row = AxisContextReviewRecord(
        axis_id="axis:test",
        generation_index=4,
        stage="novelty_repair",
        review=review,
    )

    assert row.generation_index == 4
    assert row.stage == "novelty_repair"
    assert row.review is review


def test_runtime_context_reviews_are_between_inference_and_novelty() -> None:
    source = Path(
        "pipeline_core/discovery/"
        "discovery_axis_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source
    )

    runtime = next(
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.ClassDef,
            )
            and node.name
            == "DiscoveryAxisSynthesisRuntime"
        )
    )

    run_fn = next(
        node
        for node in runtime.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name == "run"
        )
    )

    calls = []

    for node in ast.walk(
        run_fn
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        try:
            name = ast.unparse(
                node.func
            )
        except Exception:
            continue

        if name in {
            "self._review_inference",
            "self._review_context",
            "self._novelty_card",
        }:
            calls.append(
                (
                    node.lineno,
                    name,
                )
            )

    calls.sort()

    inference_lines = [
        line
        for line, name in calls
        if name
        == "self._review_inference"
    ]

    context_lines = [
        line
        for line, name in calls
        if name
        == "self._review_context"
    ]

    novelty_lines = [
        line
        for line, name in calls
        if name
        == "self._novelty_card"
    ]

    assert len(
        context_lines
    ) == 2

    assert len(
        novelty_lines
    ) == 2

    for context_line, novelty_line in zip(
        context_lines,
        novelty_lines,
        strict=True,
    ):
        previous_inference = max(
            (
                line
                for line
                in inference_lines
                if line < context_line
            ),
            default=None,
        )

        # A configured D1 reviewer is optional, but structurally,
        # where inference exists its latest review precedes S1.
        if previous_inference is not None:
            assert (
                previous_inference
                < context_line
                < novelty_line
            )
        else:
            assert (
                context_line
                < novelty_line
            )


def test_context_history_capture_is_centralized() -> None:
    source = Path(
        "pipeline_core/discovery/"
        "discovery_axis_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    assert source.count(
        "AxisContextReviewRecord("
    ) == 1

    assert source.count(
        "history=context_review_history"
    ) == 2

    assert (
        "context_review_history=tuple("
        in source
    )


def test_s1_does_not_consume_reframe_as_action_policy() -> None:
    source = Path(
        "pipeline_core/discovery/"
        "discovery_axis_runtime.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        'context_review.status == "reframe_required"'
        not in source
    )

    assert (
        'context_review.status != "pass"'
        not in source
    )
