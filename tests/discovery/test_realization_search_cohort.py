from __future__ import annotations

import pytest

from pipeline_core.discovery.realization_search_cohort import (
    build_axis_realization_cohort,
)
from pipeline_core.discovery.realization_search_shadow import (
    RealizationSemanticObservation,
)


AGG = (
    "semantic-distinctiveness-aggregation-v2.1"
)

MODEL = "openai/gpt-5.6-luna"


def obs(
    slot: int,
    hypothesis_id: str,
    tier1: str,
    tier2: str,
):
    return RealizationSemanticObservation(
        slot_index=slot,
        hypothesis_id=hypothesis_id,
        pass_tiers=(
            tier1,
            tier2,
        ),
        pass_aggregation_versions=(
            AGG,
            AGG,
        ),
        pass_served_models=(
            MODEL,
            MODEL,
        ),
        pass_diagnostic_only=(
            True,
            True,
        ),
        pass_action_policy_applied=(
            False,
            False,
        ),
        pass_scientific_selection_changed=(
            False,
            False,
        ),
    )


def test_three_realizations_group_by_axis():
    slot_payloads = [
        {
            "slot_index": 0,
            "alpha4_empty": False,
            "hypothesis_by_axis": {
                "axis:a":
                    "hypothesis:a0",
            },
            "semantic_by_hypothesis": {
                "hypothesis:a0":
                    obs(
                        0,
                        "hypothesis:a0",
                        "MODERATE",
                        "MODERATE",
                    ),
            },
        },
        {
            "slot_index": 1,
            "alpha4_empty": False,
            "hypothesis_by_axis": {
                "axis:a":
                    "hypothesis:a1",
            },
            "semantic_by_hypothesis": {
                "hypothesis:a1":
                    obs(
                        1,
                        "hypothesis:a1",
                        "HIGH",
                        "HIGH",
                    ),
            },
        },
        {
            "slot_index": 2,
            "alpha4_empty": False,
            "hypothesis_by_axis": {
                "axis:a":
                    "hypothesis:a2",
            },
            "semantic_by_hypothesis": {
                "hypothesis:a2":
                    obs(
                        2,
                        "hypothesis:a2",
                        "INDETERMINATE",
                        "INDETERMINATE",
                    ),
            },
        },
    ]

    report = build_axis_realization_cohort(
        axis_ids=["axis:a"],
        search_width=3,
        slot_payloads=slot_payloads,
    )

    axis = report.axes[0]

    assert report.axis_count == 1

    assert (
        axis.semantic_observation_count
        == 3
    )

    assert axis.missing_slot_count == 0

    assert [
        row.status
        for row in axis.slots
    ] == [
        "SEMANTIC_EVALUATED",
        "SEMANTIC_EVALUATED",
        "SEMANTIC_EVALUATED",
    ]

    assert (
        report.production_selection_applied
        is False
    )


def test_missing_axis_hypothesis_is_explicit():
    slot_payloads = [
        {
            "slot_index": 0,
            "alpha4_empty": False,
            "hypothesis_by_axis": {
                "axis:a":
                    "hypothesis:a0",
            },
            "semantic_by_hypothesis": {
                "hypothesis:a0":
                    obs(
                        0,
                        "hypothesis:a0",
                        "HIGH",
                        "HIGH",
                    ),
            },
        },
        {
            "slot_index": 1,
            "alpha4_empty": False,
            "hypothesis_by_axis": {},
            "semantic_by_hypothesis": {},
        },
        {
            "slot_index": 2,
            "alpha4_empty": True,
            "hypothesis_by_axis": {},
            "semantic_by_hypothesis": {},
        },
    ]

    report = build_axis_realization_cohort(
        axis_ids=["axis:a"],
        search_width=3,
        slot_payloads=slot_payloads,
    )

    axis = report.axes[0]

    assert (
        axis.semantic_observation_count
        == 1
    )

    assert axis.missing_slot_count == 2

    assert [
        row.status
        for row in axis.slots
    ] == [
        "SEMANTIC_EVALUATED",
        "NO_HYPOTHESIS_FOR_AXIS",
        "ALPHA4_EMPTY",
    ]


def test_each_axis_is_grouped_independently():
    slot_payloads = [
        {
            "slot_index": 0,
            "alpha4_empty": False,
            "hypothesis_by_axis": {
                "axis:a":
                    "hypothesis:a0",
                "axis:b":
                    "hypothesis:b0",
            },
            "semantic_by_hypothesis": {
                "hypothesis:a0":
                    obs(
                        0,
                        "hypothesis:a0",
                        "HIGH",
                        "HIGH",
                    ),
                "hypothesis:b0":
                    obs(
                        0,
                        "hypothesis:b0",
                        "MODERATE",
                        "MODERATE",
                    ),
            },
        },
        {
            "slot_index": 1,
            "alpha4_empty": False,
            "hypothesis_by_axis": {
                "axis:b":
                    "hypothesis:b1",
            },
            "semantic_by_hypothesis": {
                "hypothesis:b1":
                    obs(
                        1,
                        "hypothesis:b1",
                        "HIGH",
                        "HIGH",
                    ),
            },
        },
        {
            "slot_index": 2,
            "alpha4_empty": True,
            "hypothesis_by_axis": {},
            "semantic_by_hypothesis": {},
        },
    ]

    report = build_axis_realization_cohort(
        axis_ids=[
            "axis:a",
            "axis:b",
        ],
        search_width=3,
        slot_payloads=slot_payloads,
    )

    by_axis = {
        row.axis_id: row
        for row in report.axes
    }

    assert (
        by_axis[
            "axis:a"
        ].semantic_observation_count
        == 1
    )

    assert (
        by_axis[
            "axis:b"
        ].semantic_observation_count
        == 2
    )


def test_hypothesis_without_semantic_observation_fails_closed():
    with pytest.raises(
        ValueError,
        match="without a two-pass semantic observation",
    ):
        build_axis_realization_cohort(
            axis_ids=["axis:a"],
            search_width=3,
            slot_payloads=[
                {
                    "slot_index": 0,
                    "alpha4_empty": False,
                    "hypothesis_by_axis": {
                        "axis:a":
                            "hypothesis:a0",
                    },
                    "semantic_by_hypothesis": {},
                },
                {
                    "slot_index": 1,
                    "alpha4_empty": True,
                    "hypothesis_by_axis": {},
                    "semantic_by_hypothesis": {},
                },
                {
                    "slot_index": 2,
                    "alpha4_empty": True,
                    "hypothesis_by_axis": {},
                    "semantic_by_hypothesis": {},
                },
            ],
        )


def test_requires_exact_realization_slot_set():
    with pytest.raises(
        ValueError,
        match="slots must be exactly",
    ):
        build_axis_realization_cohort(
            axis_ids=["axis:a"],
            search_width=3,
            slot_payloads=[
                {
                    "slot_index": 0,
                },
                {
                    "slot_index": 2,
                },
                {
                    "slot_index": 3,
                },
            ],
        )
