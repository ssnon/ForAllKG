from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


InferenceSourceClass = Literal[
    "G_GROUNDED",
    "A_AXIS",
    "S_BOUNDED_SYNTHESIS",
    "X_UNSUPPORTED_SPECIFICITY",
]

InferenceAction = Literal[
    "KEEP",
    "KEEP_HYPOTHETICAL",
    "OPEN_DIRECTION",
    "REFRAME",
    "REMOVE",
]

InferenceReviewStatus = Literal[
    "pass",
    "reframe_required",
]

InferenceOverallRisk = Literal[
    "low",
    "moderate",
    "high",
]

InferenceAssertionKind = Literal[
    "central_hypothesis",
    "prediction",
]


_KEEP_ACTIONS = {
    "KEEP",
    "KEEP_HYPOTHETICAL",
}

_REPAIR_ACTIONS = {
    "OPEN_DIRECTION",
    "REFRAME",
    "REMOVE",
}


def allowed_inference_actions(
    source_class: str,
) -> set[str]:
    return {
        "G_GROUNDED": {
            "KEEP",
        },
        "A_AXIS": {
            "KEEP_HYPOTHETICAL",
        },
        "S_BOUNDED_SYNTHESIS": {
            "KEEP",
            "OPEN_DIRECTION",
        },
        "X_UNSUPPORTED_SPECIFICITY": {
            "REFRAME",
            "REMOVE",
        },
    }[source_class]


class AxisInferenceAssertionDraft(StrictModel):
    """One assertion-level epistemic-strength judgment.

    This contract distinguishes four sources of scientific content:

    G
        Directly grounded positive-premise content.

    A
        Content explicitly supplied by the discovery axis. The axis remains
        inspiration-only and is not promoted to positive evidence.

    S
        A bounded, testable synthesis needed to connect grounded premises
        with the assigned discovery axis.

    X
        Additional specificity that is neither supplied by the positive
        premises nor by the axis and is not required merely to connect them.
    """

    assertion_id: str = Field(min_length=1)
    assertion_kind: InferenceAssertionKind
    assertion_text: str = Field(min_length=1)

    source_class: InferenceSourceClass
    action: InferenceAction

    grounded_statement_ids: list[str] = Field(
        default_factory=list
    )

    axis_basis: list[str] = Field(
        default_factory=list
    )

    specificity_tags: list[str] = Field(
        default_factory=list
    )

    rationale: str = Field(min_length=1)




class AxisInferenceReviewDraft(StrictModel):
    """LLM-facing structured review before deterministic lineage attachment."""

    assertions: list[AxisInferenceAssertionDraft] = Field(
        min_length=1
    )

    overall_risk: InferenceOverallRisk
    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _assertion_structure(
        self,
    ) -> "AxisInferenceReviewDraft":
        ids = [
            row.assertion_id
            for row in self.assertions
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate inference assertion_id"
            )

        central = [
            row
            for row in self.assertions
            if row.assertion_kind
            == "central_hypothesis"
        ]

        if len(central) != 1:
            raise ValueError(
                "inference review must contain "
                "exactly one central_hypothesis assertion"
            )

        return self


class AxisInferenceReview(StrictModel):
    """Compiled and provenance-bound inference-strength review."""

    schema_version: Literal[
        "axis-inference-review-v1"
    ] = "axis-inference-review-v1"

    review_id: str = Field(min_length=1)

    axis_id: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)

    source_context_id: str = Field(min_length=1)
    source_context_sha256: str = Field(min_length=1)

    critic_prompt_version: str = Field(min_length=1)
    critic_prompt_sha256: str = Field(min_length=1)

    status: InferenceReviewStatus

    assertions: list[AxisInferenceAssertionDraft] = Field(
        min_length=1
    )

    overall_risk: InferenceOverallRisk
    reason_codes: list[str] = Field(
        default_factory=list
    )

    interpretation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _review_consistency(
        self,
    ) -> "AxisInferenceReview":
        ids = [
            row.assertion_id
            for row in self.assertions
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate inference assertion_id"
            )

        central = [
            row
            for row in self.assertions
            if row.assertion_kind
            == "central_hypothesis"
        ]

        if len(central) != 1:
            raise ValueError(
                "compiled inference review must contain "
                "exactly one central_hypothesis assertion"
            )

        for row in self.assertions:
            allowed = allowed_inference_actions(
                row.source_class
            )

            if row.action not in allowed:
                raise ValueError(
                    "inference source/action mismatch: "
                    f"source_class={row.source_class!r}, "
                    f"action={row.action!r}, "
                    f"allowed={sorted(allowed)!r}"
                )

        actions = {
            row.action
            for row in self.assertions
        }

        requires_repair = bool(
            actions & _REPAIR_ACTIONS
        )

        expected_status = (
            "reframe_required"
            if requires_repair
            else "pass"
        )

        if self.status != expected_status:
            raise ValueError(
                "inference review status/action mismatch: "
                f"expected={expected_status!r}, "
                f"actual={self.status!r}, "
                f"actions={sorted(actions)!r}"
            )

        return self


def inference_review_status(
    assertions: list[
        AxisInferenceAssertionDraft
    ],
) -> InferenceReviewStatus:
    """Deterministically derive the overall action status."""

    if any(
        row.action in _REPAIR_ACTIONS
        for row in assertions
    ):
        return "reframe_required"

    return "pass"
