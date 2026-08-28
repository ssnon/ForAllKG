from pipeline_core.discovery.question_task_preservation_shadow_artifact import (
    build_task_preservation_shadow_artifact,
)


def audit_row(
    unit_id,
    *,
    group="A01",
    p1_status="PASS",
    p1_role="DIRECT_ANSWER",
    p2_status=None,
    p2_role=None,
    stable=True,
):
    p2_status = p2_status or p1_status
    p2_role = p2_role or p1_role

    return {
        "group":
            group,

        "unit_id":
            unit_id,

        "pass_1_status":
            p1_status,

        "pass_2_status":
            p2_status,

        "pass_1_role":
            p1_role,

        "pass_2_role":
            p2_role,

        "decision_stable":
            stable,

        "stable_status":
            p1_status
            if stable
            else None,

        "stable_role":
            p1_role
            if stable
            else None,
    }


def raw_observation(
    incumbent,
    challenger,
    *,
    overlap=0.91,
    phase="GENERAL_STRICT",
):
    return {
        "incumbent_id":
            incumbent,

        "challenger_id":
            challenger,

        "semantic_overlap":
            overlap,

        "phase":
            phase,

        "incumbent_bundle_rank":
            2,

        "challenger_candidate_rank":
            None,
    }


def test_artifact_deduplicates_repeated_pair_and_replaces():
    raw = {
        "observations": [
            raw_observation(
                "old",
                "new",
                overlap=0.91,
            ),
            raw_observation(
                "old",
                "new",
                overlap=0.93,
                phase="GENERAL_RELAXED",
            ),
        ]
    }

    audit = {
        "candidate_results": [
            audit_row(
                "old",
                p1_status="FAIL",
                p1_role="TASK_REPLACEMENT",
            ),
            audit_row(
                "new"
            ),
        ]
    }

    result = (
        build_task_preservation_shadow_artifact(
            raw_conflict_payload=raw,
            responsiveness_payload=audit,
            group="A01",
        )
    )

    assert result[
        "raw_observation_count"
    ] == 2

    assert result[
        "unique_pair_count"
    ] == 1

    assert result[
        "replacement_pair_count"
    ] == 1

    assert len(
        result["proposals"]
    ) == 1

    assert (
        result[
            "proposals"
        ][0]["observation_count"]
        == 2
    )


def test_artifact_counts_missing_assessment_per_pair():
    raw = {
        "observations": [
            raw_observation(
                "old",
                "unknown",
            ),
            raw_observation(
                "old",
                "unknown",
                phase="GENERAL_RELAXED",
            ),
        ]
    }

    audit = {
        "candidate_results": [
            audit_row(
                "old",
                p1_status="FAIL",
                p1_role="TASK_REPLACEMENT",
            )
        ]
    }

    result = (
        build_task_preservation_shadow_artifact(
            raw_conflict_payload=raw,
            responsiveness_payload=audit,
            group="A01",
        )
    )

    assert result[
        "unique_pair_count"
    ] == 1

    assert result[
        "missing_assessment_pair_count"
    ] == 1


def test_artifact_filters_responsiveness_by_group():
    raw = {
        "observations": [
            raw_observation(
                "old",
                "new",
            )
        ]
    }

    audit = {
        "candidate_results": [
            audit_row(
                "old",
                group="A02",
                p1_status="FAIL",
                p1_role="TASK_REPLACEMENT",
            ),
            audit_row(
                "new",
                group="A02",
            ),
        ]
    }

    result = (
        build_task_preservation_shadow_artifact(
            raw_conflict_payload=raw,
            responsiveness_payload=audit,
            group="A01",
        )
    )

    assert result[
        "assessment_count"
    ] == 0

    assert result[
        "missing_assessment_pair_count"
    ] == 1


def test_cross_class_instability_remains_unresolved():
    raw = {
        "observations": [
            raw_observation(
                "old",
                "new",
            )
        ]
    }

    audit = {
        "candidate_results": [
            audit_row(
                "old",
                p1_status="FAIL",
                p1_role="TASK_REPLACEMENT",
            ),
            audit_row(
                "new",
                p1_status="PASS",
                p1_role="DIRECT_ANSWER",
                p2_status="FAIL",
                p2_role="TASK_REPLACEMENT",
                stable=False,
            ),
        ]
    }

    result = (
        build_task_preservation_shadow_artifact(
            raw_conflict_payload=raw,
            responsiveness_payload=audit,
            group="A01",
        )
    )

    assert result[
        "replacement_pair_count"
    ] == 0

    assert (
        result[
            "proposals"
        ][0]["reason"]
        == "UNRESOLVED_TASK_ASSESSMENT"
    )


def test_artifact_is_explicitly_shadow_only():
    result = (
        build_task_preservation_shadow_artifact(
            raw_conflict_payload={
                "observations": []
            },
            responsiveness_payload={
                "candidate_results": []
            },
            group="A01",
        )
    )

    assert result["shadow_only"] is True

    assert (
        result[
            "production_selection_changed"
        ]
        is False
    )
