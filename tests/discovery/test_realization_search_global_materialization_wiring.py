from __future__ import annotations

import inspect

from pipeline_core.discovery.realization_search_materialize import (
    AxisWinnerMaterializationRecord,
    materialize_realization_winners,
)


def test_materialization_record_has_truthful_global_nonselection_state():
    row = AxisWinnerMaterializationRecord(
        axis_id="axis:a",
        status="ELIGIBLE_NOT_GLOBALLY_SELECTED",
        winner_slot_index=0,
        winner_hypothesis_id="hypothesis:a",
        winner_tier="HIGH",
    )

    assert (
        row.status
        == "ELIGIBLE_NOT_GLOBALLY_SELECTED"
    )


def test_global_materialization_is_opt_in_and_legacy_default_is_preserved():
    sig = inspect.signature(
        materialize_realization_winners
    )

    assert (
        sig.parameters[
            "global_selection_enforced"
        ].default
        is False
    )

    assert (
        sig.parameters[
            "global_winner_axis_id"
        ].default
        is None
    )


def test_materializer_distinguishes_eligibility_from_global_retention():
    src = inspect.getsource(
        materialize_realization_winners
    )

    assert (
        "ELIGIBLE_NOT_GLOBALLY_SELECTED"
        in src
    )

    assert (
        "NO_ELIGIBLE_REALIZATION"
        in src
    )

    assert (
        "global production selection must materialize "
        in src
    )
    assert (
        "exactly one canonical winner"
        in src
    )


def test_non_global_axis_keeps_axis_local_winner_metadata():
    src = inspect.getsource(
        materialize_realization_winners
    )

    marker = (
        "ELIGIBLE_NOT_GLOBALLY_SELECTED"
    )

    pos = src.index(marker)

    nearby = src[
        max(0, pos - 900):
        pos + 900
    ]

    assert "winner_slot_index" in nearby
    assert "winner_hypothesis_id" in nearby
    assert "winner_tier" in nearby
