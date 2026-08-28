from __future__ import annotations

from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisResponsivenessDraft,
    QuestionAxisTwoPassStability,
)


def summarize_question_axis_two_pass(
    pass_1: QuestionAxisResponsivenessDraft,
    pass_2: QuestionAxisResponsivenessDraft,
) -> QuestionAxisTwoPassStability:
    """Summarize only decision-relevant two-pass stability.

    Dimension-level YES/PARTIAL variation is intentionally tolerated.

    N2B established that the production-relevant quantities are:
    - overall_status
    - axis_role

    This function does not change planner or production behavior.
    """

    stable = (
        pass_1.overall_status
        == pass_2.overall_status
        and
        pass_1.axis_role
        == pass_2.axis_role
    )

    if stable:
        return QuestionAxisTwoPassStability(
            pass_1_status=(
                pass_1.overall_status
            ),
            pass_2_status=(
                pass_2.overall_status
            ),
            pass_1_role=pass_1.axis_role,
            pass_2_role=pass_2.axis_role,
            decision_stable=True,
            stable_status=(
                pass_1.overall_status
            ),
            stable_role=pass_1.axis_role,
            reason_codes=[
                "OVERALL_STATUS_STABLE",
                "AXIS_ROLE_STABLE",
            ],
        )

    reasons = []

    if (
        pass_1.overall_status
        != pass_2.overall_status
    ):
        reasons.append(
            "OVERALL_STATUS_UNSTABLE"
        )

    if (
        pass_1.axis_role
        != pass_2.axis_role
    ):
        reasons.append(
            "AXIS_ROLE_UNSTABLE"
        )

    return QuestionAxisTwoPassStability(
        pass_1_status=(
            pass_1.overall_status
        ),
        pass_2_status=(
            pass_2.overall_status
        ),
        pass_1_role=pass_1.axis_role,
        pass_2_role=pass_2.axis_role,
        decision_stable=False,
        stable_status=None,
        stable_role=None,
        reason_codes=reasons,
    )
