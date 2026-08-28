from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.discovery.realization_search_shadow import (
    RealizationSemanticObservation,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


RealizationSlotStatus = Literal[
    "SEMANTIC_EVALUATED",
    "NO_HYPOTHESIS_FOR_AXIS",
    "ALPHA4_EMPTY",
]


class RealizationSlotResult(
    StrictModel
):
    slot_index: int = Field(
        ge=0
    )

    status: RealizationSlotStatus

    axis_id: str

    hypothesis_id: str | None = None

    semantic_observation: (
        RealizationSemanticObservation
        | None
    ) = None


class AxisRealizationCohort(
    StrictModel
):
    schema_version: Literal[
        "axis-realization-cohort-v1"
    ] = "axis-realization-cohort-v1"

    axis_id: str

    search_width: int = Field(
        ge=1,
        le=4,
    )

    slots: list[
        RealizationSlotResult
    ]

    semantic_observation_count: int = Field(
        ge=0
    )

    missing_slot_count: int = Field(
        ge=0
    )


class RealizationSearchCohortReport(
    StrictModel
):
    schema_version: Literal[
        "realization-search-cohort-report-v1"
    ] = (
        "realization-search-cohort-report-v1"
    )

    search_width: int = Field(
        ge=1,
        le=4,
    )

    axis_count: int = Field(
        ge=0
    )

    axes: list[
        AxisRealizationCohort
    ]

    production_selection_applied: Literal[
        False
    ] = False


def build_axis_realization_cohort(
    *,
    axis_ids: list[str],
    search_width: int,
    slot_payloads: list[
        dict[str, object]
    ],
) -> RealizationSearchCohortReport:
    """Aggregate realization trajectories by frozen discovery axis.

    ``slot_payloads`` are deterministic orchestration records with:

      slot_index
      alpha4_empty
      hypothesis_by_axis
      semantic_by_hypothesis

    Missing hypotheses are preserved explicitly rather than silently
    shrinking the realization-search width.
    """

    if search_width < 1 or search_width > 4:
        raise ValueError(
            "search_width must be between 1 and 4"
        )

    if len(slot_payloads) != search_width:
        raise ValueError(
            "slot payload count must equal search_width"
        )

    expected_slots = list(
        range(search_width)
    )

    actual_slots = sorted(
        int(
            row["slot_index"]
        )
        for row in slot_payloads
    )

    if actual_slots != expected_slots:
        raise ValueError(
            "realization slots must be exactly "
            f"{expected_slots}; got {actual_slots}"
        )

    if len(set(axis_ids)) != len(axis_ids):
        raise ValueError(
            "axis_ids must be unique"
        )

    axes: list[
        AxisRealizationCohort
    ] = []

    by_slot = {
        int(row["slot_index"]):
            row
        for row in slot_payloads
    }

    for axis_id in axis_ids:
        slot_results: list[
            RealizationSlotResult
        ] = []

        for slot_index in expected_slots:
            payload = by_slot[
                slot_index
            ]

            alpha4_empty = bool(
                payload.get(
                    "alpha4_empty",
                    False,
                )
            )

            hypothesis_by_axis = (
                payload.get(
                    "hypothesis_by_axis",
                    {}
                )
            )

            semantic_by_hypothesis = (
                payload.get(
                    "semantic_by_hypothesis",
                    {}
                )
            )

            if not isinstance(
                hypothesis_by_axis,
                dict,
            ):
                raise ValueError(
                    "hypothesis_by_axis must be a dict"
                )

            if not isinstance(
                semantic_by_hypothesis,
                dict,
            ):
                raise ValueError(
                    "semantic_by_hypothesis must be a dict"
                )

            hypothesis_id = (
                hypothesis_by_axis.get(
                    axis_id
                )
            )

            if alpha4_empty:
                if hypothesis_id is not None:
                    raise ValueError(
                        "ALPHA4_EMPTY slot may not contain "
                        "an axis hypothesis"
                    )

                slot_results.append(
                    RealizationSlotResult(
                        slot_index=slot_index,
                        status="ALPHA4_EMPTY",
                        axis_id=axis_id,
                    )
                )

                continue

            if hypothesis_id is None:
                slot_results.append(
                    RealizationSlotResult(
                        slot_index=slot_index,
                        status=(
                            "NO_HYPOTHESIS_FOR_AXIS"
                        ),
                        axis_id=axis_id,
                    )
                )

                continue

            hypothesis_id = str(
                hypothesis_id
            )

            observation = (
                semantic_by_hypothesis.get(
                    hypothesis_id
                )
            )

            if observation is None:
                raise ValueError(
                    "Axis hypothesis reached the realization "
                    "cohort without a two-pass semantic "
                    "observation: "
                    f"slot={slot_index}, "
                    f"axis={axis_id}, "
                    f"hypothesis={hypothesis_id}"
                )

            if not isinstance(
                observation,
                RealizationSemanticObservation,
            ):
                observation = (
                    RealizationSemanticObservation
                    .model_validate(
                        observation
                    )
                )

            if (
                observation.slot_index
                != slot_index
            ):
                raise ValueError(
                    "semantic observation slot mismatch"
                )

            if (
                observation.hypothesis_id
                != hypothesis_id
            ):
                raise ValueError(
                    "semantic observation hypothesis mismatch"
                )

            slot_results.append(
                RealizationSlotResult(
                    slot_index=slot_index,
                    status=(
                        "SEMANTIC_EVALUATED"
                    ),
                    axis_id=axis_id,
                    hypothesis_id=(
                        hypothesis_id
                    ),
                    semantic_observation=(
                        observation
                    ),
                )
            )

        semantic_count = sum(
            row.status
            == "SEMANTIC_EVALUATED"
            for row in slot_results
        )

        missing_count = (
            search_width
            - semantic_count
        )

        axes.append(
            AxisRealizationCohort(
                axis_id=axis_id,
                search_width=(
                    search_width
                ),
                slots=slot_results,
                semantic_observation_count=(
                    semantic_count
                ),
                missing_slot_count=(
                    missing_count
                ),
            )
        )

    return RealizationSearchCohortReport(
        search_width=search_width,
        axis_count=len(axes),
        axes=axes,
    )
