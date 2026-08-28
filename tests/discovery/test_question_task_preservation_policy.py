from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisTwoPassStability,
)
from pipeline_core.discovery.question_task_preservation_policy import (
    arbitrate_semantic_conflict,
    classify_task_preservation,
)


def stability(
    *,
    pass_1_status="PASS",
    pass_1_role="DIRECT_ANSWER",
    pass_2_status=None,
    pass_2_role=None,
    exact_stable=True,
    stable_status=None,
    stable_role=None,
):
    pass_2_status = pass_2_status or pass_1_status
    pass_2_role = pass_2_role or pass_1_role

    if exact_stable:
        stable_status = stable_status or pass_1_status
        stable_role = stable_role or pass_1_role
    else:
        stable_status = None
        stable_role = None

    return QuestionAxisTwoPassStability(
        pass_1_status=pass_1_status,
        pass_2_status=pass_2_status,
        pass_1_role=pass_1_role,
        pass_2_role=pass_2_role,
        decision_stable=exact_stable,
        stable_status=stable_status,
        stable_role=stable_role,
    )


def assess(
    candidate_id,
    *,
    quality=True,
    **kwargs,
):
    return classify_task_preservation(
        candidate_id=candidate_id,
        quality_eligible=quality,
        stability=stability(**kwargs),
    )


def test_direct_replaces_task_replacing():
    incumbent = assess(
        "old",
        pass_1_status="FAIL",
        pass_1_role="TASK_REPLACEMENT",
    )
    challenger = assess("new")

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "REPLACE_WITH_CHALLENGER"
    assert result.reason == "CHALLENGER_MORE_TASK_PRESERVING"


def test_direct_replaces_unrelated():
    incumbent = assess(
        "old",
        pass_1_status="FAIL",
        pass_1_role="UNRELATED",
    )
    challenger = assess("new")

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "REPLACE_WITH_CHALLENGER"


def test_subordinate_replaces_task_replacing():
    incumbent = assess(
        "old",
        pass_1_status="FAIL",
        pass_1_role="TASK_REPLACEMENT",
    )

    challenger = assess(
        "new",
        pass_1_status="WARNING",
        pass_1_role="SUBORDINATE_EXTENSION",
    )

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "REPLACE_WITH_CHALLENGER"


def test_task_replacing_does_not_replace_direct():
    incumbent = assess("old")

    challenger = assess(
        "new",
        pass_1_status="FAIL",
        pass_1_role="TASK_REPLACEMENT",
    )

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "KEEP_INCUMBENT"


def test_equal_task_class_preserves_existing_order():
    incumbent = assess("old")
    challenger = assess("new")

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "KEEP_INCUMBENT"
    assert result.reason == "EQUAL_TASK_PRESERVATION_KEEP_ORDER"


def test_exact_role_instability_can_be_coarse_task_replacing_stable():
    assessment = assess(
        "x",
        pass_1_status="FAIL",
        pass_1_role="TASK_REPLACEMENT",
        pass_2_status="FAIL",
        pass_2_role="UNRELATED",
        exact_stable=False,
    )

    assert assessment.source_decision_stable is False
    assert assessment.decision_stable is True
    assert assessment.task_class == "TASK_REPLACING"


def test_coarse_task_replacing_incumbent_can_be_replaced_by_direct():
    incumbent = assess(
        "old",
        pass_1_status="FAIL",
        pass_1_role="TASK_REPLACEMENT",
        pass_2_status="FAIL",
        pass_2_role="UNRELATED",
        exact_stable=False,
    )

    challenger = assess("new")

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "REPLACE_WITH_CHALLENGER"
    assert result.reason == "CHALLENGER_MORE_TASK_PRESERVING"


def test_cross_class_instability_remains_unresolved():
    assessment = assess(
        "x",
        pass_1_status="WARNING",
        pass_1_role="SUBORDINATE_EXTENSION",
        pass_2_status="FAIL",
        pass_2_role="TASK_REPLACEMENT",
        exact_stable=False,
    )

    assert assessment.source_decision_stable is False
    assert assessment.decision_stable is False
    assert assessment.task_class == "UNRESOLVED"


def test_unresolved_challenger_cannot_replace():
    incumbent = assess(
        "old",
        pass_1_status="FAIL",
        pass_1_role="TASK_REPLACEMENT",
    )

    challenger = assess(
        "new",
        pass_1_status="PASS",
        pass_1_role="DIRECT_ANSWER",
        pass_2_status="FAIL",
        pass_2_role="TASK_REPLACEMENT",
        exact_stable=False,
    )

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "KEEP_INCUMBENT"
    assert result.reason == "UNRESOLVED_TASK_ASSESSMENT"


def test_quality_ineligible_challenger_never_replaces():
    incumbent = assess(
        "old",
        pass_1_status="FAIL",
        pass_1_role="TASK_REPLACEMENT",
    )

    challenger = assess(
        "new",
        quality=False,
    )

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "KEEP_INCUMBENT"
    assert result.reason == "CHALLENGER_NOT_QUALITY_ELIGIBLE"


def test_quality_ineligible_incumbent_yields_to_quality_eligible_challenger():
    incumbent = assess(
        "old",
        quality=False,
    )

    challenger = assess(
        "new",
        pass_1_status="WARNING",
        pass_1_role="SUBORDINATE_EXTENSION",
    )

    result = arbitrate_semantic_conflict(
        incumbent=incumbent,
        challenger=challenger,
    )

    assert result.decision == "REPLACE_WITH_CHALLENGER"
    assert result.reason == "INCUMBENT_NOT_QUALITY_ELIGIBLE"


def test_unexpected_stable_combination_is_unresolved():
    assessment = assess(
        "x",
        pass_1_status="WARNING",
        pass_1_role="DIRECT_ANSWER",
    )

    assert assessment.task_class == "UNRESOLVED"
    assert assessment.decision_stable is False
