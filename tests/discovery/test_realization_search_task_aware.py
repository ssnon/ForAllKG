from __future__ import annotations

import pytest

from pipeline_core.discovery.question_task_preservation_policy import (
    TaskPreservationAssessment,
)
from pipeline_core.discovery.realization_search_cohort import (
    build_axis_realization_cohort,
)
from pipeline_core.discovery.realization_search_shadow import (
    RealizationSearchPolicy,
    RealizationSemanticObservation,
)
from pipeline_core.discovery.realization_search_task_aware import (
    select_axis_task_aware_production_winner,
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


def task(
    hid: str,
    task_class: str,
    *,
    stable: bool = True,
):
    return TaskPreservationAssessment(
        candidate_id=hid,
        quality_eligible=True,
        task_class=task_class,
        decision_stable=stable,
        source_decision_stable=stable,
    )


def cohort():
    return (
        build_axis_realization_cohort(
            axis_ids=[
                "axis:a"
            ],
            search_width=3,
            slot_payloads=[
                {
                    "slot_index":
                        0,

                    "alpha4_empty":
                        False,

                    "hypothesis_by_axis":
                        {
                            "axis:a":
                                "hypothesis:r0",
                        },

                    "semantic_by_hypothesis":
                        {
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
                    "slot_index":
                        1,

                    "alpha4_empty":
                        False,

                    "hypothesis_by_axis":
                        {
                            "axis:a":
                                "hypothesis:r1",
                        },

                    "semantic_by_hypothesis":
                        {
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
                    "slot_index":
                        2,

                    "alpha4_empty":
                        False,

                    "hypothesis_by_axis":
                        {
                            "axis:a":
                                "hypothesis:r2",
                        },

                    "semantic_by_hypothesis":
                        {
                            "hypothesis:r2":
                                obs(
                                    2,
                                    "hypothesis:r2",
                                    "LOW",
                                    "LOW",
                                ),
                        },
                },
            ],
        )
        .axes[0]
    )


def test_task_replacing_high_cannot_beat_direct_moderate():
    original = cohort()

    report = (
        select_axis_task_aware_production_winner(
            original,
            task_assessments_by_slot_hypothesis={
                (
                    0,
                    "hypothesis:r0",
                ):
                    task(
                        "hypothesis:r0",
                        "TASK_REPLACING",
                    ),

                (
                    1,
                    "hypothesis:r1",
                ):
                    task(
                        "hypothesis:r1",
                        "DIRECT",
                    ),

                (
                    2,
                    "hypothesis:r2",
                ):
                    task(
                        "hypothesis:r2",
                        "SUBORDINATE",
                    ),
            },
            policy=(
                RealizationSearchPolicy()
            ),
        )
    )

    assert (
        report.selection.winner_slot_index
        == 1
    )

    assert (
        report.selection.winner_tier
        == "MODERATE"
    )

    assert (
        report.task_ineligible_slot_indices
        == [0]
    )


def test_direct_high_still_wins_normally():
    report = (
        select_axis_task_aware_production_winner(
            cohort(),
            task_assessments_by_slot_hypothesis={
                (
                    0,
                    "hypothesis:r0",
                ):
                    task(
                        "hypothesis:r0",
                        "DIRECT",
                    ),

                (
                    1,
                    "hypothesis:r1",
                ):
                    task(
                        "hypothesis:r1",
                        "DIRECT",
                    ),

                (
                    2,
                    "hypothesis:r2",
                ):
                    task(
                        "hypothesis:r2",
                        "SUBORDINATE",
                    ),
            },
            policy=(
                RealizationSearchPolicy()
            ),
        )
    )

    assert (
        report.selection.winner_slot_index
        == 0
    )

    assert (
        report.selection.winner_tier
        == "HIGH"
    )


def test_unresolved_and_task_replacing_all_fail_closed():
    report = (
        select_axis_task_aware_production_winner(
            cohort(),
            task_assessments_by_slot_hypothesis={
                (
                    0,
                    "hypothesis:r0",
                ):
                    task(
                        "hypothesis:r0",
                        "TASK_REPLACING",
                    ),

                (
                    1,
                    "hypothesis:r1",
                ):
                    task(
                        "hypothesis:r1",
                        "UNRESOLVED",
                        stable=False,
                    ),

                (
                    2,
                    "hypothesis:r2",
                ):
                    task(
                        "hypothesis:r2",
                        "TASK_REPLACING",
                    ),
            },
            policy=(
                RealizationSearchPolicy()
            ),
        )
    )

    assert (
        report.selection.status
        == "NO_STABLE_DETERMINATE_CANDIDATE"
    )

    assert (
        report.selection.winner_hypothesis_id
        is None
    )

    assert (
        report.task_eligible_slot_count
        == 0
    )


def test_original_semantic_cohort_is_not_mutated():
    original = cohort()

    (
        select_axis_task_aware_production_winner(
            original,
            task_assessments_by_slot_hypothesis={
                (
                    0,
                    "hypothesis:r0",
                ):
                    task(
                        "hypothesis:r0",
                        "TASK_REPLACING",
                    ),

                (
                    1,
                    "hypothesis:r1",
                ):
                    task(
                        "hypothesis:r1",
                        "DIRECT",
                    ),

                (
                    2,
                    "hypothesis:r2",
                ):
                    task(
                        "hypothesis:r2",
                        "SUBORDINATE",
                    ),
            },
            policy=(
                RealizationSearchPolicy()
            ),
        )
    )

    assert (
        original.semantic_observation_count
        == 3
    )

    assert [
        row.status
        for row
        in original.slots
    ] == [
        "SEMANTIC_EVALUATED",
        "SEMANTIC_EVALUATED",
        "SEMANTIC_EVALUATED",
    ]


def test_missing_task_assessment_fails_closed():
    with pytest.raises(
        ValueError,
        match=(
            "lacks task-preservation assessment"
        ),
    ):
        select_axis_task_aware_production_winner(
            cohort(),
            task_assessments_by_slot_hypothesis={},
            policy=(
                RealizationSearchPolicy()
            ),
        )
