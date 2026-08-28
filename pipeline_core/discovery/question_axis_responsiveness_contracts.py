from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


QuestionAxisPreservation = Literal[
    "YES",
    "PARTIAL",
    "NO",
]


QuestionAxisRole = Literal[
    "DIRECT_ANSWER",
    "SUBORDINATE_EXTENSION",
    "TASK_REPLACEMENT",
    "UNRELATED",
]


QuestionAxisResponsivenessStatus = Literal[
    "PASS",
    "WARNING",
    "FAIL",
]


class QuestionAxisResponsivenessDraft(
    StrictModel
):
    requested_variable_preservation: (
        QuestionAxisPreservation
    )

    requested_outcome_preservation: (
        QuestionAxisPreservation
    )

    relation_nucleus_preservation: (
        QuestionAxisPreservation
    )

    axis_role: QuestionAxisRole

    overall_status: (
        QuestionAxisResponsivenessStatus
    )

    rationale: str = Field(
        min_length=1,
        max_length=1800,
    )

    @model_validator(mode="after")
    def _role_status_contract(
        self,
    ) -> "QuestionAxisResponsivenessDraft":

        if self.axis_role in {
            "TASK_REPLACEMENT",
            "UNRELATED",
        }:
            if self.overall_status != "FAIL":
                raise ValueError(
                    "TASK_REPLACEMENT or UNRELATED "
                    "must have overall_status=FAIL"
                )

        elif self.axis_role == "DIRECT_ANSWER":
            if self.overall_status != "PASS":
                raise ValueError(
                    "DIRECT_ANSWER must have "
                    "overall_status=PASS"
                )

        elif self.axis_role == "SUBORDINATE_EXTENSION":
            if self.overall_status not in {
                "PASS",
                "WARNING",
            }:
                raise ValueError(
                    "SUBORDINATE_EXTENSION must be "
                    "PASS or WARNING"
                )

        return self


class QuestionAxisTwoPassStability(
    StrictModel
):
    schema_version: Literal[
        "question-axis-two-pass-stability-v1"
    ] = (
        "question-axis-two-pass-stability-v1"
    )

    pass_1_status: (
        QuestionAxisResponsivenessStatus
    )

    pass_2_status: (
        QuestionAxisResponsivenessStatus
    )

    pass_1_role: QuestionAxisRole
    pass_2_role: QuestionAxisRole

    decision_stable: bool

    stable_status: (
        QuestionAxisResponsivenessStatus
        | None
    ) = None

    stable_role: (
        QuestionAxisRole
        | None
    ) = None

    reason_codes: list[str] = Field(
        default_factory=list,
        max_length=8,
    )

    # N2C remains diagnostic only.
    shadow_only: Literal[True] = True

    action_policy_applied: Literal[
        False
    ] = False

    planner_changed: Literal[
        False
    ] = False

    production_selection_changed: Literal[
        False
    ] = False
