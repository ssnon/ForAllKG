from pipeline_core.discovery.question_task_preservation_policy import (
    TaskPreservationAssessment,
)
from pipeline_core.discovery.question_task_preservation_shadow import (
    SemanticConflictObservation,
    evaluate_semantic_conflicts_shadow,
)


def assessment(
    candidate_id,
    task_class,
    *,
    quality=True,
    stable=True,
):
    return TaskPreservationAssessment(
        candidate_id=candidate_id,
        quality_eligible=quality,
        task_class=task_class,
        decision_stable=stable,
        source_decision_stable=stable,
    )


def observation(
    incumbent="old",
    challenger="new",
    overlap=0.90,
    phase="GENERAL_STRICT",
):
    return SemanticConflictObservation(
        incumbent_id=incumbent,
        challenger_id=challenger,
        semantic_overlap=overlap,
        phase=phase,
    )


def test_direct_challenger_generates_replacement_proposal():
    report = evaluate_semantic_conflicts_shadow(
        observations=[
            observation()
        ],
        assessments={
            "old": assessment(
                "old",
                "TASK_REPLACING",
            ),
            "new": assessment(
                "new",
                "DIRECT",
            ),
        },
    )

    assert report.observations_seen == 1
    assert report.observations_assessed == 1
    assert report.replacement_proposal_count == 1
    assert report.keep_proposal_count == 0

    proposal = report.proposals[0]

    assert (
        proposal.proposed_action
        == "REPLACE_WITH_CHALLENGER"
    )
    assert (
        proposal.reason
        == "CHALLENGER_MORE_TASK_PRESERVING"
    )
    assert proposal.shadow_only is True
    assert (
        proposal.production_selection_changed
        is False
    )


def test_less_responsive_challenger_is_kept_out():
    report = evaluate_semantic_conflicts_shadow(
        observations=[
            observation()
        ],
        assessments={
            "old": assessment(
                "old",
                "DIRECT",
            ),
            "new": assessment(
                "new",
                "TASK_REPLACING",
            ),
        },
    )

    assert report.replacement_proposal_count == 0
    assert report.keep_proposal_count == 1

    assert (
        report.proposals[0].proposed_action
        == "KEEP_INCUMBENT"
    )


def test_equal_task_class_preserves_bundle_order():
    report = evaluate_semantic_conflicts_shadow(
        observations=[
            observation()
        ],
        assessments={
            "old": assessment(
                "old",
                "DIRECT",
            ),
            "new": assessment(
                "new",
                "DIRECT",
            ),
        },
    )

    assert report.keep_proposal_count == 1

    assert (
        report.proposals[0].reason
        == "EQUAL_TASK_PRESERVATION_KEEP_ORDER"
    )


def test_unstable_candidate_does_not_trigger_replacement():
    report = evaluate_semantic_conflicts_shadow(
        observations=[
            observation()
        ],
        assessments={
            "old": assessment(
                "old",
                "TASK_REPLACING",
            ),
            "new": assessment(
                "new",
                "UNRESOLVED",
                stable=False,
            ),
        },
    )

    assert report.replacement_proposal_count == 0

    assert (
        report.proposals[0].reason
        == "UNRESOLVED_TASK_ASSESSMENT"
    )


def test_quality_ineligible_direct_challenger_does_not_replace():
    report = evaluate_semantic_conflicts_shadow(
        observations=[
            observation()
        ],
        assessments={
            "old": assessment(
                "old",
                "TASK_REPLACING",
            ),
            "new": assessment(
                "new",
                "DIRECT",
                quality=False,
            ),
        },
    )

    assert report.replacement_proposal_count == 0

    assert (
        report.proposals[0].reason
        == "CHALLENGER_NOT_QUALITY_ELIGIBLE"
    )


def test_missing_assessment_is_reported_without_guessing():
    report = evaluate_semantic_conflicts_shadow(
        observations=[
            observation()
        ],
        assessments={
            "old": assessment(
                "old",
                "TASK_REPLACING",
            ),
        },
    )

    assert report.observations_seen == 1
    assert report.observations_assessed == 0
    assert report.missing_assessment_count == 1

    proposal = report.proposals[0]

    assert (
        proposal.proposed_action
        == "NO_ASSESSMENT"
    )
    assert (
        proposal.reason
        == "TASK_ASSESSMENT_MISSING"
    )


def test_multiple_observations_are_not_collapsed_or_executed():
    observations = [
        observation(
            incumbent="old1",
            challenger="new1",
            overlap=0.91,
        ),
        observation(
            incumbent="old2",
            challenger="new2",
            overlap=0.93,
        ),
    ]

    report = evaluate_semantic_conflicts_shadow(
        observations=observations,
        assessments={
            "old1": assessment(
                "old1",
                "TASK_REPLACING",
            ),
            "new1": assessment(
                "new1",
                "DIRECT",
            ),
            "old2": assessment(
                "old2",
                "DIRECT",
            ),
            "new2": assessment(
                "new2",
                "SUBORDINATE",
            ),
        },
    )

    assert report.observations_seen == 2
    assert report.observations_assessed == 2
    assert report.replacement_proposal_count == 1
    assert report.keep_proposal_count == 1

    assert report.shadow_only is True
    assert (
        report.production_selection_changed
        is False
    )
