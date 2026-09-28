from __future__ import annotations

from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisTwoPassStability,
)
from pipeline_core.discovery.question_task_preservation_policy import (
    classify_task_preservation,
)


def _stability(
    *,
    status: str,
    role: str,
) -> QuestionAxisTwoPassStability:
    return QuestionAxisTwoPassStability(
        pass_1_status=status,
        pass_2_status=status,
        pass_1_role=role,
        pass_2_role=role,
        decision_stable=True,
        stable_status=status,
        stable_role=role,
    )


def test_pass_subordinate_maps_to_subordinate():
    result = classify_task_preservation(
        candidate_id="candidate:pass-subordinate",
        quality_eligible=True,
        stability=_stability(
            status="PASS",
            role="SUBORDINATE_EXTENSION",
        ),
    )

    assert result.task_class == "SUBORDINATE"
    assert result.decision_stable is True
    assert result.source_decision_stable is True


def test_warning_subordinate_still_maps_to_subordinate():
    result = classify_task_preservation(
        candidate_id="candidate:warning-subordinate",
        quality_eligible=True,
        stability=_stability(
            status="WARNING",
            role="SUBORDINATE_EXTENSION",
        ),
    )

    assert result.task_class == "SUBORDINATE"
    assert result.decision_stable is True
    assert result.source_decision_stable is True


def test_pass_direct_mapping_is_unchanged():
    result = classify_task_preservation(
        candidate_id="candidate:direct",
        quality_eligible=True,
        stability=_stability(
            status="PASS",
            role="DIRECT_ANSWER",
        ),
    )

    assert result.task_class == "DIRECT"
    assert result.decision_stable is True


def test_fail_task_replacement_mapping_is_unchanged():
    result = classify_task_preservation(
        candidate_id="candidate:replacement",
        quality_eligible=True,
        stability=_stability(
            status="FAIL",
            role="TASK_REPLACEMENT",
        ),
    )

    assert result.task_class == "TASK_REPLACING"
    assert result.decision_stable is True
