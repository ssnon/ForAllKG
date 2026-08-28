from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)


AxisContextReviewStatus = Literal[
    "pass",
    "pass_with_unknowns",
    "reframe_required",
]


class AxisContextReviewUnavailableError(RuntimeError):
    """A domain reviewer cannot produce a valid review for one axis."""


class DiscoveryAxisContextReviewer(Protocol):
    """Optional domain-aware scientific-context review capability.

    Implementations may be domain-specific. The generic discovery
    runtime only requires a review object exposing:

        review_id: str
        hypothesis_id: str
        status: pass | pass_with_unknowns | reframe_required

    The reviewer does not decide KEEP/REFRAME/REJECT actions here.
    """

    def review(
        self,
        *,
        dual: DualHypothesisContext,
        axis: DiscoveryAxis,
        card: Any,
    ) -> Any:
        ...


@dataclass(frozen=True)
class AxisContextReviewRecord:
    axis_id: str
    generation_index: int
    stage: str
    review: Any


def validate_context_review_shape(
    *,
    axis: DiscoveryAxis,
    card: Any,
    review: Any,
) -> AxisContextReviewStatus:
    status = str(
        getattr(
            review,
            "status",
            "",
        )
    )

    allowed = {
        "pass",
        "pass_with_unknowns",
        "reframe_required",
    }

    if status not in allowed:
        raise RuntimeError(
            "configured context reviewer returned invalid "
            f"status {status!r}"
        )

    hypothesis_id = str(
        getattr(
            review,
            "hypothesis_id",
            "",
        )
    )

    expected_hypothesis_id = str(
        getattr(
            card,
            "hypothesis_id",
            "",
        )
    )

    if (
        not hypothesis_id
        or hypothesis_id
        != expected_hypothesis_id
    ):
        raise RuntimeError(
            "configured context reviewer returned "
            "hypothesis_id mismatch: "
            f"expected={expected_hypothesis_id!r}, "
            f"actual={hypothesis_id!r}"
        )

    review_id = str(
        getattr(
            review,
            "review_id",
            "",
        )
    )

    if not review_id:
        raise RuntimeError(
            "configured context reviewer returned "
            "review without review_id"
        )

    return status  # type: ignore[return-value]
