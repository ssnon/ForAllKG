from __future__ import annotations

from dac_her.hypothesis_trend_directional_contracts import (
    canonical_dependent_change,
    expected_prediction_direction,
)
from dac_her.hypothesis_trend_directional_validator import (
    _uses_decrease_frame,
)


def test_positive_maps_to_increase():
    assert canonical_dependent_change(["positive"]) == "increase"
    assert expected_prediction_direction(["increase"]) == "increase"


def test_negative_maps_to_decrease():
    assert canonical_dependent_change(["negative"]) == "decrease"
    assert expected_prediction_direction(["decrease"]) == "decrease"


def test_mixed_direction_fails_closed_to_unspecified():
    assert (
        canonical_dependent_change(["positive", "negative"])
        == "unspecified"
    )
    assert (
        expected_prediction_direction(
            ["increase", "decrease"]
        )
        == "unspecified"
    )


def test_decrease_frame_detector_catches_seen_failure_mode():
    assert _uses_decrease_frame(
        (
            "decreasing particle size is hypothesized to improve "
            "qualitative SERS performance"
        ),
        "particle_size",
    )
    assert _uses_decrease_frame(
        "qualitative SERS performance as particle size decreases",
        "particle_size",
    )


def test_canonical_increase_frame_is_not_flagged():
    assert not _uses_decrease_frame(
        (
            "increasing particle size is hypothesized to increase "
            "qualitative SERS performance"
        ),
        "particle_size",
    )
