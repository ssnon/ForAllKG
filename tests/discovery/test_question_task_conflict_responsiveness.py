from dataclasses import dataclass

from pipeline_core.discovery.question_axis_responsiveness_contracts import (
    QuestionAxisResponsivenessDraft,
)
from pipeline_core.discovery.question_task_conflict_responsiveness import (
    conflict_candidate_ids,
    conflict_responsiveness_artifact,
    evaluate_conflict_candidates_shadow,
    recover_candidate_units,
)


@dataclass(frozen=True)
class FakeGeneration:
    draft: QuestionAxisResponsivenessDraft


class FakeBackend:
    def __init__(self):
        self.calls = []

    def review(
        self,
        prompt,
        *,
        review_pass_index,
        debug_path=None,
    ):
        self.calls.append(
            (
                review_pass_index,
                debug_path,
            )
        )

        return FakeGeneration(
            draft=QuestionAxisResponsivenessDraft(
                requested_variable_preservation="YES",
                requested_outcome_preservation="YES",
                relation_nucleus_preservation="YES",
                axis_role="DIRECT_ANSWER",
                overall_status="PASS",
                rationale=(
                    "The candidate directly preserves "
                    "the requested relation."
                ),
            )
        )


def traversal_payload():
    return {
        "candidate_paths": [
            {
                "path_id":
                    "p1",

                "candidate_unit": {
                    "unit_id":
                        "candidate_unit:u1",

                    "label":
                        "Geometry promotes hotspots",

                    "proposed_subject":
                        "geometry",

                    "proposed_relation":
                        "PROMOTES",

                    "proposed_object":
                        "hotspots",
                },
            },
            {
                # duplicate route for same scientific unit
                "path_id":
                    "p2",

                "candidate_unit": {
                    "unit_id":
                        "candidate_unit:u1",

                    "label":
                        "Geometry promotes hotspots",

                    "proposed_subject":
                        "geometry",

                    "proposed_relation":
                        "PROMOTES",

                    "proposed_object":
                        "hotspots",
                },
            },
            {
                "path_id":
                    "p3",

                "candidate_unit": {
                    "unit_id":
                        "candidate_unit:u2",

                    "label":
                        "Air exposure changes signal",

                    "proposed_subject":
                        "air exposure",

                    "proposed_relation":
                        "CHANGES",

                    "proposed_object":
                        "signal",
                },
            },
        ]
    }


def raw_conflicts():
    return {
        "observations": [
            {
                "incumbent_id":
                    "candidate_unit:u2",

                "challenger_id":
                    "candidate_unit:u1",

                "semantic_overlap":
                    0.91,

                "phase":
                    "GENERAL_STRICT",
            },
            {
                "incumbent_id":
                    "candidate_unit:u2",

                "challenger_id":
                    "candidate_unit:u1",

                "semantic_overlap":
                    0.92,

                "phase":
                    "GENERAL_RELAXED",
            },
        ]
    }


def test_conflict_candidate_ids_are_unique_in_first_seen_order():
    assert conflict_candidate_ids(
        raw_conflicts()
    ) == (
        "candidate_unit:u2",
        "candidate_unit:u1",
    )


def test_recover_candidate_units_deduplicates_routes():
    result = recover_candidate_units(
        traversal_payloads=[
            traversal_payload()
        ],
        wanted_ids=[
            "candidate_unit:u1",
            "candidate_unit:u2",
        ],
    )

    assert set(result) == {
        "candidate_unit:u1",
        "candidate_unit:u2",
    }

    assert (
        result[
            "candidate_unit:u1"
        ].proposed_subject
        == "geometry"
    )


def test_two_pass_review_runs_once_per_unique_candidate():
    backend = FakeBackend()

    results = evaluate_conflict_candidates_shadow(
        group="A01",
        question=(
            "How does geometry affect hotspots?"
        ),
        raw_conflict_payload=(
            raw_conflicts()
        ),
        traversal_payloads=[
            traversal_payload()
        ],
        backend=backend,
    )

    assert len(results) == 2
    assert len(backend.calls) == 4

    assert all(
        result.decision_stable
        for result in results
    )

    assert all(
        result.stable_status
        == "PASS"
        for result in results
    )

    assert all(
        result.stable_role
        == "DIRECT_ANSWER"
        for result in results
    )


def test_missing_conflict_candidate_fails_closed():
    backend = FakeBackend()

    raw = {
        "observations": [
            {
                "incumbent_id":
                    "candidate_unit:missing",

                "challenger_id":
                    "candidate_unit:u1",

                "semantic_overlap":
                    0.9,

                "phase":
                    "GENERAL_STRICT",
            }
        ]
    }

    try:
        evaluate_conflict_candidates_shadow(
            group="A01",
            question="q",
            raw_conflict_payload=raw,
            traversal_payloads=[
                traversal_payload()
            ],
            backend=backend,
        )
    except ValueError as exc:
        assert (
            "candidate units missing"
            in str(exc)
        )
    else:
        raise AssertionError(
            "expected missing candidate failure"
        )


def test_artifact_remains_c7c_compatible():
    backend = FakeBackend()

    results = evaluate_conflict_candidates_shadow(
        group="A01",
        question="q",
        raw_conflict_payload=(
            raw_conflicts()
        ),
        traversal_payloads=[
            traversal_payload()
        ],
        backend=backend,
    )

    artifact = conflict_responsiveness_artifact(
        group="A01",
        question="q",
        results=results,
    )

    assert artifact[
        "candidate_count"
    ] == 2

    assert len(
        artifact[
            "candidate_results"
        ]
    ) == 2

    row = artifact[
        "candidate_results"
    ][0]

    for key in (
        "group",
        "unit_id",
        "pass_1_status",
        "pass_2_status",
        "pass_1_role",
        "pass_2_role",
        "decision_stable",
        "stable_status",
        "stable_role",
    ):
        assert key in row

    assert (
        artifact[
            "production_selection_changed"
        ]
        is False
    )
