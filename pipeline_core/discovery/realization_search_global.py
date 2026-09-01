from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.realization_search_task_aware import (
    TaskAwareProductionSelectionReport,
)


DeterminateTier = Literal[
    "LOW",
    "MODERATE",
    "HIGH",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class GlobalAxisWinnerCandidate(StrictModel):
    """One already-selected, task-eligible axis-local winner.

    This layer does not inspect raw realizations, task classes, semantic
    passes, exploration scores, or novelty evidence.  Those decisions are
    upstream authorities.  It only compares authoritative axis-local
    winners across the frozen discovery-axis plan.
    """

    axis_plan_index: int = Field(
        ge=0
    )

    axis_id: str = Field(
        min_length=1
    )

    winner_slot_index: int = Field(
        ge=0
    )

    winner_hypothesis_id: str = Field(
        min_length=1
    )

    winner_tier: DeterminateTier


class GlobalProductionSelectionReport(StrictModel):
    schema_version: Literal[
        "realization-search-global-production-selection-v1"
    ] = (
        "realization-search-global-production-selection-v1"
    )

    axis_count: int = Field(
        ge=0
    )

    eligible_axis_winner_count: int = Field(
        ge=0
    )

    status: Literal[
        "WINNER_SELECTED",
        "NO_ELIGIBLE_AXIS_WINNER",
    ]

    candidates: list[
        GlobalAxisWinnerCandidate
    ]

    winner_axis_plan_index: int | None = None
    winner_axis_id: str | None = None
    winner_slot_index: int | None = None
    winner_hypothesis_id: str | None = None
    winner_tier: DeterminateTier | None = None

    selection_objective: Literal[
        "TASK_ELIGIBLE_STABLE_DETERMINATE_SEMANTIC_TIER"
    ] = (
        "TASK_ELIGIBLE_STABLE_DETERMINATE_SEMANTIC_TIER"
    )

    tie_break: Literal[
        "EARLIEST_FROZEN_AXIS_PLAN_ORDER"
    ] = (
        "EARLIEST_FROZEN_AXIS_PLAN_ORDER"
    )

    production_authority: Literal[
        True
    ] = True

    task_eligibility_reused: Literal[
        True
    ] = True

    semantic_tier_ranking_reused: Literal[
        True
    ] = True

    semantic_aggregation_reimplemented: Literal[
        False
    ] = False

    task_classification_reimplemented: Literal[
        False
    ] = False

    llm_calls: Literal[
        0
    ] = 0

    cross_axis_selection_applied: Literal[
        True
    ] = True

    production_selection_changed: Literal[
        True
    ] = True

    @model_validator(
        mode="after"
    )
    def _validate_contract(
        self,
    ) -> "GlobalProductionSelectionReport":
        if (
            self.eligible_axis_winner_count
            != len(self.candidates)
        ):
            raise ValueError(
                "eligible_axis_winner_count must match candidates"
            )

        if (
            self.eligible_axis_winner_count
            > self.axis_count
        ):
            raise ValueError(
                "eligible axis winner count exceeds axis count"
            )

        winner_fields = (
            self.winner_axis_plan_index,
            self.winner_axis_id,
            self.winner_slot_index,
            self.winner_hypothesis_id,
            self.winner_tier,
        )

        if self.status == "WINNER_SELECTED":
            if any(
                value is None
                for value in winner_fields
            ):
                raise ValueError(
                    "WINNER_SELECTED requires complete winner fields"
                )

            if not self.candidates:
                raise ValueError(
                    "WINNER_SELECTED requires at least one candidate"
                )

        else:
            if any(
                value is not None
                for value in winner_fields
            ):
                raise ValueError(
                    "NO_ELIGIBLE_AXIS_WINNER requires null winner fields"
                )

            if self.candidates:
                raise ValueError(
                    "NO_ELIGIBLE_AXIS_WINNER requires zero candidates"
                )

        return self


_TIER_RANK: dict[
    DeterminateTier,
    int,
] = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
}


def select_global_axis_production_winner(
    *,
    axis_order: list[str],
    task_aware_selections_by_axis: dict[
        str,
        TaskAwareProductionSelectionReport,
    ],
) -> GlobalProductionSelectionReport:
    """Collapse authoritative axis-local winners to zero or one winner.

    Frozen N7-A cross-axis rule:

      1. Inputs are the existing task-aware axis-local production reports.
      2. Axes without an axis-local winner are unavailable.
      3. The global layer does not re-run or reinterpret task eligibility.
      4. The global layer does not re-run semantic aggregation.
      5. Existing determinate semantic tiers rank HIGH > MODERATE > LOW.
      6. Tier ties resolve to earliest frozen discovery-axis plan order.
      7. No axis-local winner means fail closed with no global winner.

    Dictionary iteration order is never authoritative.
    """

    if len(set(axis_order)) != len(axis_order):
        raise ValueError(
            "axis_order must contain unique axis IDs"
        )

    expected = set(axis_order)
    actual = set(
        task_aware_selections_by_axis
    )

    if actual != expected:
        missing = sorted(
            expected - actual
        )

        extra = sorted(
            actual - expected
        )

        raise ValueError(
            "task-aware selection axis set does not match "
            "frozen axis order: "
            f"missing={missing}, extra={extra}"
        )

    candidates: list[
        GlobalAxisWinnerCandidate
    ] = []

    for axis_plan_index, axis_id in enumerate(
        axis_order
    ):
        report = (
            task_aware_selections_by_axis[
                axis_id
            ]
        )

        if report.axis_id != axis_id:
            raise ValueError(
                "task-aware report axis_id mismatch: "
                f"key={axis_id}, report={report.axis_id}"
            )

        selection = report.selection

        if (
            selection.status
            == "NO_STABLE_DETERMINATE_CANDIDATE"
        ):
            continue

        if (
            selection.status
            != "WINNER_SELECTED"
        ):
            raise ValueError(
                "unsupported axis-local selection status: "
                f"{selection.status}"
            )

        winner_slot_index = (
            selection.winner_slot_index
        )

        winner_hypothesis_id = (
            selection.winner_hypothesis_id
        )

        winner_tier = (
            selection.winner_tier
        )

        if (
            winner_slot_index is None
            or not winner_hypothesis_id
            or winner_tier is None
        ):
            raise ValueError(
                "axis-local WINNER_SELECTED report lacks "
                "complete winner fields"
            )

        candidates.append(
            GlobalAxisWinnerCandidate(
                axis_plan_index=(
                    axis_plan_index
                ),
                axis_id=axis_id,
                winner_slot_index=(
                    winner_slot_index
                ),
                winner_hypothesis_id=(
                    winner_hypothesis_id
                ),
                winner_tier=(
                    winner_tier
                ),
            )
        )

    if not candidates:
        return (
            GlobalProductionSelectionReport(
                axis_count=len(
                    axis_order
                ),
                eligible_axis_winner_count=0,
                status=(
                    "NO_ELIGIBLE_AXIS_WINNER"
                ),
                candidates=[],
            )
        )

    winner = max(
        candidates,
        key=lambda row: (
            _TIER_RANK[
                row.winner_tier
            ],
            -row.axis_plan_index,
        ),
    )

    return (
        GlobalProductionSelectionReport(
            axis_count=len(
                axis_order
            ),
            eligible_axis_winner_count=(
                len(candidates)
            ),
            status="WINNER_SELECTED",
            candidates=candidates,
            winner_axis_plan_index=(
                winner.axis_plan_index
            ),
            winner_axis_id=(
                winner.axis_id
            ),
            winner_slot_index=(
                winner.winner_slot_index
            ),
            winner_hypothesis_id=(
                winner.winner_hypothesis_id
            ),
            winner_tier=(
                winner.winner_tier
            ),
        )
    )
