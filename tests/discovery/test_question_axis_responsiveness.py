from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.question_axis_responsiveness import (
    summarize_question_axis_two_pass,
)
from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisResponsivenessDraft,
)
from pipeline_core.discovery.question_axis_responsiveness_prompt import (
    QuestionAxisResponsivenessPromptAssembler,
)


def _axis(
    *,
    label: str,
    subject: str,
    relation: str,
    obj: str,
) -> DiscoveryAxis:
    return DiscoveryAxis(
        axis_id="discovery_axis:test",
        axis_rank=1,
        inspiration_id="inspiration:test",
        source_path_id="path:test",
        candidate_unit_id="candidate:test",
        label=label,
        proposed_subject=subject,
        proposed_relation=relation,
        proposed_object=obj,
        rendered_path=label,
        source_mode="test",
        exploration_score=0.8,
        candidate_unit_score=0.8,
        planner_score=0.8,
        mechanistic_continuity_band="high",
    )


def _draft(
    *,
    status: str,
    role: str,
    variable: str = "YES",
    outcome: str = "YES",
    relation: str = "YES",
) -> QuestionAxisResponsivenessDraft:
    return QuestionAxisResponsivenessDraft(
        requested_variable_preservation=(
            variable
        ),
        requested_outcome_preservation=(
            outcome
        ),
        relation_nucleus_preservation=(
            relation
        ),
        axis_role=role,
        overall_status=status,
        rationale="test rationale",
    )


def test_task_replacement_must_fail():
    with pytest.raises(
        ValidationError
    ):
        _draft(
            status="PASS",
            role="TASK_REPLACEMENT",
        )


def test_direct_answer_must_pass():
    with pytest.raises(
        ValidationError
    ):
        _draft(
            status="WARNING",
            role="DIRECT_ANSWER",
        )


def test_subordinate_extension_can_warn():
    result = _draft(
        status="WARNING",
        role="SUBORDINATE_EXTENSION",
        variable="PARTIAL",
        outcome="YES",
        relation="PARTIAL",
    )

    assert (
        result.overall_status
        == "WARNING"
    )


def test_n2b_t01_two_pass_is_stable_failure():
    pass_1 = _draft(
        status="FAIL",
        role="TASK_REPLACEMENT",
        variable="NO",
        outcome="PARTIAL",
        relation="NO",
    )

    pass_2 = _draft(
        status="FAIL",
        role="TASK_REPLACEMENT",
        variable="NO",
        outcome="PARTIAL",
        relation="NO",
    )

    result = (
        summarize_question_axis_two_pass(
            pass_1,
            pass_2,
        )
    )

    assert result.decision_stable is True
    assert result.stable_status == "FAIL"
    assert (
        result.stable_role
        == "TASK_REPLACEMENT"
    )

    assert result.shadow_only is True
    assert (
        result.production_selection_changed
        is False
    )


def test_n2b_t03_dimension_variation_is_still_stable():
    pass_1 = _draft(
        status="PASS",
        role="DIRECT_ANSWER",
        variable="YES",
        outcome="YES",
        relation="YES",
    )

    # N2B pass 2 differed only at the outcome dimension.
    pass_2 = _draft(
        status="PASS",
        role="DIRECT_ANSWER",
        variable="YES",
        outcome="PARTIAL",
        relation="YES",
    )

    result = (
        summarize_question_axis_two_pass(
            pass_1,
            pass_2,
        )
    )

    assert result.decision_stable is True
    assert result.stable_status == "PASS"
    assert (
        result.stable_role
        == "DIRECT_ANSWER"
    )


def test_role_flip_is_unstable():
    pass_1 = _draft(
        status="PASS",
        role="DIRECT_ANSWER",
    )

    pass_2 = _draft(
        status="WARNING",
        role="SUBORDINATE_EXTENSION",
        relation="PARTIAL",
    )

    result = (
        summarize_question_axis_two_pass(
            pass_1,
            pass_2,
        )
    )

    assert result.decision_stable is False
    assert result.stable_status is None
    assert result.stable_role is None


def test_prompt_contains_question_and_relation_nucleus():
    axis = _axis(
        label=(
            "SERS signal varies with "
            "excitation wavelength"
        ),
        subject="Raman/SERS signal intensity",
        relation="VARIES_WITH",
        obj="excitation wavelength",
    )

    question = (
        "Which excitation wavelength and "
        "Raman reporter combination provides "
        "the most appropriate SERS conditions?"
    )

    prompt = (
        QuestionAxisResponsivenessPromptAssembler()
        .build(
            question=question,
            axis=axis,
        )
    )

    assert question in prompt.user_prompt
    assert (
        "Raman/SERS signal intensity"
        in prompt.user_prompt
    )
    assert (
        "VARIES_WITH"
        in prompt.user_prompt
    )
    assert (
        "excitation wavelength"
        in prompt.user_prompt
    )


def test_prompt_has_no_external_novelty_task():
    axis = _axis(
        label="test axis",
        subject="signal",
        relation="VARIES_WITH",
        obj="wavelength",
    )

    prompt = (
        QuestionAxisResponsivenessPromptAssembler()
        .build(
            question="Which wavelength is appropriate?",
            axis=axis,
        )
    )

    lower = prompt.system_prompt.lower()

    assert "do not judge" in lower
    assert "external novelty" in lower
    assert "question -> axis" in lower


def test_empty_question_fails_closed():
    axis = _axis(
        label="test axis",
        subject="signal",
        relation="VARIES_WITH",
        obj="wavelength",
    )

    with pytest.raises(
        ValueError,
        match="question must be non-empty",
    ):
        (
            QuestionAxisResponsivenessPromptAssembler()
            .build(
                question="   ",
                axis=axis,
            )
        )
