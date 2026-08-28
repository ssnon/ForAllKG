from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from pipeline_core.discovery.question_task_preservation_policy import (
    TaskPreservationAssessment,
)
from pipeline_core.discovery.realization_search_cohort import (
    AxisRealizationCohort,
)
from pipeline_core.discovery.realization_search_production import (
    ProductionRealizationSelectionReport,
    select_axis_realization_production_winner,
)
from pipeline_core.discovery.realization_search_shadow import (
    RealizationSearchPolicy,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class TaskAwareRealizationSlotDecision(
    StrictModel
):
    slot_index: int = Field(
        ge=0
    )

    hypothesis_id: str | None = None

    task_assessed: bool

    task_class: Literal[
        "DIRECT",
        "SUBORDINATE",
        "TASK_REPLACING",
        "UNRESOLVED",
    ] | None = None

    task_decision_stable: bool | None = None

    task_source_decision_stable: bool | None = None

    task_quality_eligible: bool | None = None

    winner_ranking_eligible: bool


class TaskAwareProductionSelectionReport(
    StrictModel
):
    schema_version: Literal[
        "realization-search-task-aware-production-v1"
    ] = (
        "realization-search-task-aware-production-v1"
    )

    axis_id: str

    search_width: int = Field(
        ge=1,
        le=4,
    )

    task_eligible_slot_count: int = Field(
        ge=0
    )

    task_ineligible_slot_count: int = Field(
        ge=0
    )

    task_ineligible_slot_indices: list[int] = Field(
        default_factory=list
    )

    slot_decisions: list[
        TaskAwareRealizationSlotDecision
    ]

    selection: ProductionRealizationSelectionReport

    production_task_eligibility_applied: Literal[
        True
    ] = True

    semantic_tier_ranking_changed: Literal[
        False
    ] = False

    production_selection_changed: Literal[
        True
    ] = True


def select_axis_task_aware_production_winner(
    cohort: AxisRealizationCohort,
    *,
    task_assessments_by_slot_hypothesis: dict[
        tuple[int, str],
        TaskPreservationAssessment,
    ],
    policy: RealizationSearchPolicy,
) -> TaskAwareProductionSelectionReport:
    """Filter winner eligibility by the frozen task-preservation policy.

    Scientific/semantic evaluation is NOT removed from task-ineligible
    realizations.  This wrapper changes only which already-evaluated
    realizations may enter production winner ranking.

    Frozen production task rule:
        DIRECT       -> eligible
        SUBORDINATE  -> eligible
        TASK_REPLACING -> ineligible
        UNRESOLVED     -> ineligible

    The surviving candidates are ranked by the existing realization
    selector without changing HIGH > MODERATE > LOW semantics.
    """

    if (
        cohort.search_width
        != policy.search_width
    ):
        raise ValueError(
            "task-aware cohort search_width does not match policy"
        )

    synthetic_slots = []
    slot_decisions = []

    task_eligible_count = 0
    task_ineligible_indices = []

    for row in sorted(
        cohort.slots,
        key=lambda item:
            item.slot_index,
    ):
        # Missing/empty realization attempts never enter ranking and
        # therefore do not require a task review.
        if (
            row.status
            != "SEMANTIC_EVALUATED"
        ):
            synthetic_slots.append(
                row
            )

            slot_decisions.append(
                TaskAwareRealizationSlotDecision(
                    slot_index=(
                        row.slot_index
                    ),
                    hypothesis_id=(
                        row.hypothesis_id
                    ),
                    task_assessed=False,
                    winner_ranking_eligible=False,
                )
            )

            continue

        hypothesis_id = (
            row.hypothesis_id
        )

        if not hypothesis_id:
            raise ValueError(
                "SEMANTIC_EVALUATED realization lacks hypothesis_id"
            )

        key = (
            row.slot_index,
            hypothesis_id,
        )

        assessment = (
            task_assessments_by_slot_hypothesis.get(
                key
            )
        )

        if assessment is None:
            raise ValueError(
                "Semantically evaluated realization lacks "
                "task-preservation assessment: "
                f"slot={row.slot_index}, "
                f"hypothesis={hypothesis_id}"
            )

        if (
            assessment.candidate_id
            != hypothesis_id
        ):
            raise ValueError(
                "Task-preservation candidate_id mismatch"
            )

        eligible = (
            assessment.quality_eligible
            and assessment.decision_stable
            and assessment.task_class
            in {
                "DIRECT",
                "SUBORDINATE",
            }
        )

        slot_decisions.append(
            TaskAwareRealizationSlotDecision(
                slot_index=(
                    row.slot_index
                ),
                hypothesis_id=(
                    hypothesis_id
                ),
                task_assessed=True,
                task_class=(
                    assessment.task_class
                ),
                task_decision_stable=(
                    assessment.decision_stable
                ),
                task_source_decision_stable=(
                    assessment.source_decision_stable
                ),
                task_quality_eligible=(
                    assessment.quality_eligible
                ),
                winner_ranking_eligible=(
                    eligible
                ),
            )
        )

        if eligible:
            task_eligible_count += 1

            synthetic_slots.append(
                row
            )

            continue

        task_ineligible_indices.append(
            row.slot_index
        )

        # Preserve the original cohort unchanged.  Only the synthetic
        # ranking cohort marks this slot unavailable to the downstream
        # frozen semantic-tier selector.
        synthetic_slots.append(
            row.model_copy(
                update={
                    "status":
                        "NO_HYPOTHESIS_FOR_AXIS",

                    "hypothesis_id":
                        None,

                    "semantic_observation":
                        None,
                }
            )
        )

    synthetic_cohort = (
        cohort.model_copy(
            update={
                "slots":
                    synthetic_slots,

                "semantic_observation_count":
                    task_eligible_count,

                "missing_slot_count":
                    (
                        cohort.search_width
                        - task_eligible_count
                    ),
            }
        )
    )

    selection = (
        select_axis_realization_production_winner(
            synthetic_cohort,
            policy=policy,
        )
    )

    return (
        TaskAwareProductionSelectionReport(
            axis_id=cohort.axis_id,
            search_width=(
                cohort.search_width
            ),
            task_eligible_slot_count=(
                task_eligible_count
            ),
            task_ineligible_slot_count=(
                len(
                    task_ineligible_indices
                )
            ),
            task_ineligible_slot_indices=(
                task_ineligible_indices
            ),
            slot_decisions=(
                slot_decisions
            ),
            selection=selection,
        )
    )
