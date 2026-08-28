from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisTwoPassStability,
)


TaskPreservationClass = Literal[
    "DIRECT",
    "SUBORDINATE",
    "TASK_REPLACING",
    "UNRESOLVED",
]


@dataclass(frozen=True)
class TaskPreservationAssessment:
    """
    Shadow-only task-preservation classification.

    source_decision_stable:
        Exact status+role stability from the original two-pass critic.

    decision_stable:
        Stability at the coarser task-preservation class used only for
        semantic-conflict arbitration.

    This distinction intentionally allows, for example,

        FAIL/TASK_REPLACEMENT
        FAIL/UNRELATED

    to be stable TASK_REPLACING for arbitration while preserving the
    fact that the original exact role judgment was unstable.
    """

    candidate_id: str
    quality_eligible: bool
    task_class: TaskPreservationClass
    decision_stable: bool
    source_decision_stable: bool


@dataclass(frozen=True)
class SemanticConflictArbitration:
    incumbent_id: str
    challenger_id: str

    decision: Literal[
        "KEEP_INCUMBENT",
        "REPLACE_WITH_CHALLENGER",
    ]

    reason: Literal[
        "CHALLENGER_NOT_QUALITY_ELIGIBLE",
        "INCUMBENT_NOT_QUALITY_ELIGIBLE",
        "UNRESOLVED_TASK_ASSESSMENT",
        "CHALLENGER_MORE_TASK_PRESERVING",
        "INCUMBENT_MORE_TASK_PRESERVING",
        "EQUAL_TASK_PRESERVATION_KEEP_ORDER",
    ]

    shadow_only: Literal[True] = True
    production_selection_changed: Literal[False] = False


_TASK_RANK: dict[TaskPreservationClass, int] = {
    "UNRESOLVED": 0,
    "TASK_REPLACING": 1,
    "SUBORDINATE": 2,
    "DIRECT": 3,
}


def _coarse_task_class(
    *,
    status: str,
    role: str,
) -> TaskPreservationClass:
    if (
        status == "PASS"
        and role == "DIRECT_ANSWER"
    ):
        return "DIRECT"

    if (
        status == "WARNING"
        and role == "SUBORDINATE_EXTENSION"
    ):
        return "SUBORDINATE"

    if (
        status == "FAIL"
        and role in {
            "TASK_REPLACEMENT",
            "UNRELATED",
        }
    ):
        return "TASK_REPLACING"

    return "UNRESOLVED"


def classify_task_preservation(
    *,
    candidate_id: str,
    quality_eligible: bool,
    stability: QuestionAxisTwoPassStability,
) -> TaskPreservationAssessment:
    """
    Convert the existing exact two-pass responsiveness judgment into a
    coarse task-preservation class for shadow arbitration.

    The arbitration class is stable iff both review passes independently
    map to the same non-UNRESOLVED coarse class.

    This does NOT weaken the original critic's exact stability contract.
    The original decision_stable value is retained separately as
    source_decision_stable.
    """

    pass_1_class = _coarse_task_class(
        status=stability.pass_1_status,
        role=stability.pass_1_role,
    )

    pass_2_class = _coarse_task_class(
        status=stability.pass_2_status,
        role=stability.pass_2_role,
    )

    coarse_stable = (
        pass_1_class != "UNRESOLVED"
        and pass_1_class == pass_2_class
    )

    task_class: TaskPreservationClass = (
        pass_1_class
        if coarse_stable
        else "UNRESOLVED"
    )

    return TaskPreservationAssessment(
        candidate_id=candidate_id,
        quality_eligible=quality_eligible,
        task_class=task_class,
        decision_stable=coarse_stable,
        source_decision_stable=stability.decision_stable,
    )


def arbitrate_semantic_conflict(
    *,
    incumbent: TaskPreservationAssessment,
    challenger: TaskPreservationAssessment,
) -> SemanticConflictArbitration:
    """
    Generic shadow invariant:

    Among scientific-quality-eligible candidates that conflict under
    the existing semantic-diversity rule, a more task-preserving
    candidate must not lose solely because it arrived later in the
    existing Bundle ordering.

    This function does NOT:
    - alter semantic thresholds;
    - alter reserve floors;
    - alter exploration weights;
    - bypass scientific-quality eligibility;
    - optimize over unresolved task assessments;
    - change production selection.
    """

    if not challenger.quality_eligible:
        return SemanticConflictArbitration(
            incumbent_id=incumbent.candidate_id,
            challenger_id=challenger.candidate_id,
            decision="KEEP_INCUMBENT",
            reason="CHALLENGER_NOT_QUALITY_ELIGIBLE",
        )

    if not incumbent.quality_eligible:
        return SemanticConflictArbitration(
            incumbent_id=incumbent.candidate_id,
            challenger_id=challenger.candidate_id,
            decision="REPLACE_WITH_CHALLENGER",
            reason="INCUMBENT_NOT_QUALITY_ELIGIBLE",
        )

    if (
        not incumbent.decision_stable
        or not challenger.decision_stable
        or incumbent.task_class == "UNRESOLVED"
        or challenger.task_class == "UNRESOLVED"
    ):
        return SemanticConflictArbitration(
            incumbent_id=incumbent.candidate_id,
            challenger_id=challenger.candidate_id,
            decision="KEEP_INCUMBENT",
            reason="UNRESOLVED_TASK_ASSESSMENT",
        )

    incumbent_rank = _TASK_RANK[
        incumbent.task_class
    ]

    challenger_rank = _TASK_RANK[
        challenger.task_class
    ]

    if challenger_rank > incumbent_rank:
        return SemanticConflictArbitration(
            incumbent_id=incumbent.candidate_id,
            challenger_id=challenger.candidate_id,
            decision="REPLACE_WITH_CHALLENGER",
            reason="CHALLENGER_MORE_TASK_PRESERVING",
        )

    if incumbent_rank > challenger_rank:
        return SemanticConflictArbitration(
            incumbent_id=incumbent.candidate_id,
            challenger_id=challenger.candidate_id,
            decision="KEEP_INCUMBENT",
            reason="INCUMBENT_MORE_TASK_PRESERVING",
        )

    return SemanticConflictArbitration(
        incumbent_id=incumbent.candidate_id,
        challenger_id=challenger.candidate_id,
        decision="KEEP_INCUMBENT",
        reason="EQUAL_TASK_PRESERVATION_KEEP_ORDER",
    )
