from __future__ import annotations

from pipeline_core.discovery.realization_search_cohort import (
    build_axis_realization_cohort,
)
from pipeline_core.discovery.realization_search_production import (
    select_axis_realization_production_winner,
)
from pipeline_core.discovery.realization_search_shadow import (
    RealizationSearchPolicy,
    RealizationSemanticObservation,
)


AGG = (
    "semantic-distinctiveness-aggregation-v2.1"
)

MODEL = "openai/gpt-5.6-luna"


def obs(
    slot: int,
    hid: str,
    first: str,
    second: str,
):
    return RealizationSemanticObservation(
        slot_index=slot,
        hypothesis_id=hid,
        pass_tiers=(
            first,
            second,
        ),
        pass_aggregation_versions=(
            AGG,
            AGG,
        ),
        pass_served_models=(
            MODEL,
            MODEL,
        ),
    )


def cohort(
    slot_payloads,
):
    return (
        build_axis_realization_cohort(
            axis_ids=["axis:a"],
            search_width=3,
            slot_payloads=slot_payloads,
        )
        .axes[0]
    )


def test_partial_cohort_keeps_stable_high_despite_failed_slot():
    axis = cohort(
        [
            {
                "slot_index": 0,
                "alpha4_empty": False,
                "hypothesis_by_axis": {
                    "axis:a":
                        "hypothesis:r0",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r0":
                        obs(
                            0,
                            "hypothesis:r0",
                            "HIGH",
                            "HIGH",
                        ),
                },
            },
            {
                "slot_index": 1,
                "alpha4_empty": True,
                "hypothesis_by_axis": {},
                "semantic_by_hypothesis": {},
            },
            {
                "slot_index": 2,
                "alpha4_empty": False,
                "hypothesis_by_axis": {
                    "axis:a":
                        "hypothesis:r2",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r2":
                        obs(
                            2,
                            "hypothesis:r2",
                            "MODERATE",
                            "MODERATE",
                        ),
                },
            },
        ]
    )

    report = (
        select_axis_realization_production_winner(
            axis,
            policy=RealizationSearchPolicy(),
        )
    )

    assert report.status == (
        "WINNER_SELECTED"
    )

    assert (
        report.winner_slot_index
        == 0
    )

    assert (
        report.winner_hypothesis_id
        == "hypothesis:r0"
    )

    assert report.winner_tier == "HIGH"

    assert (
        report.semantic_observation_count
        == 2
    )

    assert report.missing_slot_count == 1

    assert (
        report.missing_slot_indices
        == [1]
    )


def test_partial_tie_break_uses_original_slot_order():
    axis = cohort(
        [
            {
                "slot_index": 0,
                "alpha4_empty": True,
                "hypothesis_by_axis": {},
                "semantic_by_hypothesis": {},
            },
            {
                "slot_index": 1,
                "alpha4_empty": False,
                "hypothesis_by_axis": {
                    "axis:a":
                        "hypothesis:r1",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r1":
                        obs(
                            1,
                            "hypothesis:r1",
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
                        "hypothesis:r2",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r2":
                        obs(
                            2,
                            "hypothesis:r2",
                            "HIGH",
                            "HIGH",
                        ),
                },
            },
        ]
    )

    report = (
        select_axis_realization_production_winner(
            axis,
            policy=RealizationSearchPolicy(),
        )
    )

    assert (
        report.winner_slot_index
        == 1
    )


def test_partial_unstable_high_does_not_beat_stable_moderate():
    axis = cohort(
        [
            {
                "slot_index": 0,
                "alpha4_empty": False,
                "hypothesis_by_axis": {
                    "axis:a":
                        "hypothesis:r0",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r0":
                        obs(
                            0,
                            "hypothesis:r0",
                            "HIGH",
                            "MODERATE",
                        ),
                },
            },
            {
                "slot_index": 1,
                "alpha4_empty": False,
                "hypothesis_by_axis": {
                    "axis:a":
                        "hypothesis:r1",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r1":
                        obs(
                            1,
                            "hypothesis:r1",
                            "MODERATE",
                            "MODERATE",
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
    )

    report = (
        select_axis_realization_production_winner(
            axis,
            policy=RealizationSearchPolicy(),
        )
    )

    assert (
        report.winner_slot_index
        == 1
    )

    assert (
        report.winner_tier
        == "MODERATE"
    )


def test_all_missing_fails_closed():
    axis = cohort(
        [
            {
                "slot_index": 0,
                "alpha4_empty": True,
                "hypothesis_by_axis": {},
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
        ]
    )

    report = (
        select_axis_realization_production_winner(
            axis,
            policy=RealizationSearchPolicy(),
        )
    )

    assert report.status == (
        "NO_STABLE_DETERMINATE_CANDIDATE"
    )

    assert (
        report.winner_hypothesis_id
        is None
    )

    assert (
        report.semantic_observation_count
        == 0
    )

    assert report.missing_slot_count == 3

    assert (
        report.missing_slot_indices
        == [0, 1, 2]
    )


def test_only_indeterminate_fails_closed():
    axis = cohort(
        [
            {
                "slot_index": 0,
                "alpha4_empty": False,
                "hypothesis_by_axis": {
                    "axis:a":
                        "hypothesis:r0",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r0":
                        obs(
                            0,
                            "hypothesis:r0",
                            "INDETERMINATE",
                            "INDETERMINATE",
                        ),
                },
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
        ]
    )

    report = (
        select_axis_realization_production_winner(
            axis,
            policy=RealizationSearchPolicy(),
        )
    )

    assert report.status == (
        "NO_STABLE_DETERMINATE_CANDIDATE"
    )

    assert (
        report.winner_hypothesis_id
        is None
    )


def test_restored_candidate_slots_are_original_slots():
    axis = cohort(
        [
            {
                "slot_index": 0,
                "alpha4_empty": True,
                "hypothesis_by_axis": {},
                "semantic_by_hypothesis": {},
            },
            {
                "slot_index": 1,
                "alpha4_empty": False,
                "hypothesis_by_axis": {
                    "axis:a":
                        "hypothesis:r1",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r1":
                        obs(
                            1,
                            "hypothesis:r1",
                            "LOW",
                            "LOW",
                        ),
                },
            },
            {
                "slot_index": 2,
                "alpha4_empty": False,
                "hypothesis_by_axis": {
                    "axis:a":
                        "hypothesis:r2",
                },
                "semantic_by_hypothesis": {
                    "hypothesis:r2":
                        obs(
                            2,
                            "hypothesis:r2",
                            "MODERATE",
                            "MODERATE",
                        ),
                },
            },
        ]
    )

    report = (
        select_axis_realization_production_winner(
            axis,
            policy=RealizationSearchPolicy(),
        )
    )

    assert [
        row.slot_index
        for row in report.candidates
    ] == [
        1,
        2,
    ]

    assert (
        report.winner_slot_index
        == 2
    )
