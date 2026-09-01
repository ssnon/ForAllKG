from __future__ import annotations

import inspect

import pytest

from pipeline_core.discovery.realization_search_global import (
    GlobalProductionSelectionReport,
    select_global_axis_production_winner,
)
from pipeline_core.discovery.realization_search_production import (
    ProductionRealizationSelectionReport,
)
from pipeline_core.discovery.realization_search_task_aware import (
    TaskAwareProductionSelectionReport,
)


def axis_report(
    axis_id: str,
    *,
    tier: str | None,
    hypothesis_id: str | None = None,
    slot_index: int = 0,
) -> TaskAwareProductionSelectionReport:
    if tier is None:
        selection = (
            ProductionRealizationSelectionReport.model_construct(
                search_width=1,
                status=(
                    "NO_STABLE_DETERMINATE_CANDIDATE"
                ),
                winner_slot_index=None,
                winner_hypothesis_id=None,
                winner_tier=None,
                candidates=[],
            )
        )
    else:
        selection = (
            ProductionRealizationSelectionReport.model_construct(
                search_width=1,
                status="WINNER_SELECTED",
                winner_slot_index=slot_index,
                winner_hypothesis_id=(
                    hypothesis_id
                    or f"hypothesis:{axis_id}"
                ),
                winner_tier=tier,
                candidates=[],
            )
        )

    return (
        TaskAwareProductionSelectionReport.model_construct(
            axis_id=axis_id,
            search_width=1,
            task_eligible_slot_count=(
                0
                if tier is None
                else 1
            ),
            task_ineligible_slot_count=0,
            task_ineligible_slot_indices=[],
            slot_decisions=[],
            selection=selection,
        )
    )


def test_high_beats_moderate_and_low():
    reports = {
        "axis:a": axis_report(
            "axis:a",
            tier="MODERATE",
        ),
        "axis:b": axis_report(
            "axis:b",
            tier="HIGH",
        ),
        "axis:c": axis_report(
            "axis:c",
            tier="LOW",
        ),
    }

    result = (
        select_global_axis_production_winner(
            axis_order=[
                "axis:a",
                "axis:b",
                "axis:c",
            ],
            task_aware_selections_by_axis=(
                reports
            ),
        )
    )

    assert result.status == "WINNER_SELECTED"
    assert result.winner_axis_id == "axis:b"
    assert result.winner_tier == "HIGH"
    assert result.eligible_axis_winner_count == 3


def test_moderate_beats_low():
    reports = {
        "axis:a": axis_report(
            "axis:a",
            tier="LOW",
        ),
        "axis:b": axis_report(
            "axis:b",
            tier="MODERATE",
        ),
    }

    result = (
        select_global_axis_production_winner(
            axis_order=[
                "axis:a",
                "axis:b",
            ],
            task_aware_selections_by_axis=(
                reports
            ),
        )
    )

    assert result.winner_axis_id == "axis:b"
    assert result.winner_tier == "MODERATE"


def test_tier_tie_uses_frozen_axis_order():
    # Deliberately reverse dict insertion order.
    reports = {
        "axis:c": axis_report(
            "axis:c",
            tier="HIGH",
        ),
        "axis:b": axis_report(
            "axis:b",
            tier="HIGH",
        ),
        "axis:a": axis_report(
            "axis:a",
            tier="HIGH",
        ),
    }

    result = (
        select_global_axis_production_winner(
            axis_order=[
                "axis:b",
                "axis:a",
                "axis:c",
            ],
            task_aware_selections_by_axis=(
                reports
            ),
        )
    )

    assert result.winner_axis_id == "axis:b"
    assert result.winner_axis_plan_index == 0


def test_axis_without_local_winner_is_skipped():
    reports = {
        "axis:a": axis_report(
            "axis:a",
            tier=None,
        ),
        "axis:b": axis_report(
            "axis:b",
            tier="LOW",
        ),
        "axis:c": axis_report(
            "axis:c",
            tier=None,
        ),
    }

    result = (
        select_global_axis_production_winner(
            axis_order=[
                "axis:a",
                "axis:b",
                "axis:c",
            ],
            task_aware_selections_by_axis=(
                reports
            ),
        )
    )

    assert result.status == "WINNER_SELECTED"
    assert result.winner_axis_id == "axis:b"
    assert result.eligible_axis_winner_count == 1
    assert [
        row.axis_id
        for row in result.candidates
    ] == ["axis:b"]


def test_zero_axis_local_winners_fails_closed():
    reports = {
        "axis:a": axis_report(
            "axis:a",
            tier=None,
        ),
        "axis:b": axis_report(
            "axis:b",
            tier=None,
        ),
    }

    result = (
        select_global_axis_production_winner(
            axis_order=[
                "axis:a",
                "axis:b",
            ],
            task_aware_selections_by_axis=(
                reports
            ),
        )
    )

    assert (
        result.status
        == "NO_ELIGIBLE_AXIS_WINNER"
    )
    assert result.winner_axis_id is None
    assert result.winner_hypothesis_id is None
    assert result.winner_tier is None
    assert result.candidates == []


def test_empty_axis_set_is_valid_fail_closed():
    result = (
        select_global_axis_production_winner(
            axis_order=[],
            task_aware_selections_by_axis={},
        )
    )

    assert result.axis_count == 0
    assert result.status == "NO_ELIGIBLE_AXIS_WINNER"


def test_axis_order_must_be_unique():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        select_global_axis_production_winner(
            axis_order=[
                "axis:a",
                "axis:a",
            ],
            task_aware_selections_by_axis={
                "axis:a": axis_report(
                    "axis:a",
                    tier="HIGH",
                ),
            },
        )


def test_axis_sets_must_match_exactly():
    with pytest.raises(
        ValueError,
        match="axis set",
    ):
        select_global_axis_production_winner(
            axis_order=[
                "axis:a",
                "axis:b",
            ],
            task_aware_selections_by_axis={
                "axis:a": axis_report(
                    "axis:a",
                    tier="HIGH",
                ),
            },
        )


def test_report_axis_id_must_match_mapping_key():
    with pytest.raises(
        ValueError,
        match="axis_id mismatch",
    ):
        select_global_axis_production_winner(
            axis_order=[
                "axis:a",
            ],
            task_aware_selections_by_axis={
                "axis:a": axis_report(
                    "axis:WRONG",
                    tier="HIGH",
                ),
            },
        )


def test_global_selector_does_not_reimplement_upstream_semantics():
    src = inspect.getsource(
        select_global_axis_production_winner
    )

    # It may only consume the already-selected winner tier.
    assert "task_class" not in src
    assert "quality_eligible" not in src
    assert "decision_stable" not in src
    assert "semantic_observation" not in src
    assert "overall_tier" not in src
    assert "exploration" not in src
    assert "novelty" not in src


def test_report_contract_exposes_zero_llm_calls():
    result = (
        select_global_axis_production_winner(
            axis_order=[
                "axis:a",
            ],
            task_aware_selections_by_axis={
                "axis:a": axis_report(
                    "axis:a",
                    tier="HIGH",
                ),
            },
        )
    )

    assert isinstance(
        result,
        GlobalProductionSelectionReport,
    )
    assert result.llm_calls == 0
    assert result.task_eligibility_reused is True
    assert result.semantic_tier_ranking_reused is True
    assert (
        result.semantic_aggregation_reimplemented
        is False
    )
    assert (
        result.task_classification_reimplemented
        is False
    )
