from pipeline_core.discovery.question_task_preservation_policy import (
    TaskPreservationAssessment,
)
from pipeline_core.discovery.question_task_preservation_shadow import (
    SemanticConflictObservation,
)
from pipeline_core.discovery.question_task_preservation_shadow_pairs import (
    aggregate_semantic_conflict_observations,
    evaluate_semantic_conflict_pairs_shadow,
)


def assessment(
    candidate_id,
    task_class,
):
    return TaskPreservationAssessment(
        candidate_id=candidate_id,
        quality_eligible=True,
        task_class=task_class,
        decision_stable=True,
        source_decision_stable=True,
    )


def observation(
    *,
    incumbent="old",
    challenger="new",
    overlap=0.90,
    phase="GENERAL_STRICT",
    incumbent_rank=2,
):
    return SemanticConflictObservation(
        incumbent_id=incumbent,
        challenger_id=challenger,
        semantic_overlap=overlap,
        phase=phase,
        incumbent_bundle_rank=incumbent_rank,
    )


def test_repeated_observations_collapse_to_one_pair():
    observations = [
        observation(
            overlap=0.91,
            phase="CANDIDATE_RESERVE",
        ),
        observation(
            overlap=0.93,
            phase="GENERAL_STRICT",
        ),
        observation(
            overlap=0.92,
            phase="GENERAL_RELAXED",
        ),
    ]

    pairs = (
        aggregate_semantic_conflict_observations(
            observations
        )
    )

    assert len(pairs) == 1

    pair = pairs[0]

    assert pair.observation_count == 3
    assert pair.max_semantic_overlap == 0.93

    assert pair.phases == (
        "CANDIDATE_RESERVE",
        "GENERAL_STRICT",
        "GENERAL_RELAXED",
    )


def test_distinct_ordered_pairs_remain_distinct():
    observations = [
        observation(
            incumbent="a",
            challenger="b",
        ),
        observation(
            incumbent="a",
            challenger="c",
        ),
    ]

    pairs = (
        aggregate_semantic_conflict_observations(
            observations
        )
    )

    assert len(pairs) == 2


def test_pair_level_arbitration_emits_one_replacement():
    observations = [
        observation(
            overlap=0.91,
            phase="CANDIDATE_RESERVE",
        ),
        observation(
            overlap=0.93,
            phase="GENERAL_STRICT",
        ),
        observation(
            overlap=0.92,
            phase="GENERAL_RELAXED",
        ),
    ]

    report = (
        evaluate_semantic_conflict_pairs_shadow(
            observations=observations,
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
    )

    assert report.raw_observation_count == 3
    assert report.unique_pair_count == 1
    assert report.assessed_pair_count == 1

    assert (
        report.replacement_pair_count
        == 1
    )

    assert len(report.proposals) == 1

    proposal = report.proposals[0]

    assert proposal.observation_count == 3

    assert (
        proposal.proposed_action
        == "REPLACE_WITH_CHALLENGER"
    )


def test_missing_assessment_is_counted_once_per_pair():
    observations = [
        observation(),
        observation(
            phase="GENERAL_RELAXED",
        ),
    ]

    report = (
        evaluate_semantic_conflict_pairs_shadow(
            observations=observations,
            assessments={
                "old": assessment(
                    "old",
                    "TASK_REPLACING",
                ),
            },
        )
    )

    assert report.raw_observation_count == 2
    assert report.unique_pair_count == 1

    assert (
        report.missing_assessment_pair_count
        == 1
    )

    assert len(report.proposals) == 1

    assert (
        report.proposals[0].proposed_action
        == "NO_ASSESSMENT"
    )


def test_equal_task_class_keeps_incumbent_once():
    observations = [
        observation(),
        observation(
            phase="GENERAL_RELAXED",
        ),
    ]

    report = (
        evaluate_semantic_conflict_pairs_shadow(
            observations=observations,
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
    )

    assert report.unique_pair_count == 1
    assert report.keep_pair_count == 1
    assert report.replacement_pair_count == 0
