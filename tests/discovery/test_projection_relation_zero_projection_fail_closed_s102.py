from __future__ import annotations

from pipeline_core.discovery.projection_relation_adjudication import (
    ProjectionRelationClaimCandidateSet,
    ProjectionRelationClaimReviewDraft,
    ProjectionRelationWorkCandidate,
    compile_projection_relation_claim_review,
)


def _candidate() -> ProjectionRelationWorkCandidate:
    return ProjectionRelationWorkCandidate(
        review_work_id="w1",
        source_work_ids=["w1"],
        title="Synthetic work",
        abstract="Synthetic abstract.",
        typed_compatibility_state="DOMAIN_COMPATIBLE",
        selection_score=1.0,
    )


def _unprojected_set(
    *,
    candidates: list[ProjectionRelationWorkCandidate] | None = None,
) -> ProjectionRelationClaimCandidateSet:
    rows = list(candidates or [])
    return ProjectionRelationClaimCandidateSet(
        hypothesis_id="h1",
        claim_id="claim:unprojected",
        relation_ir_id="relation:unprojected",
        claim_text="A relation whose typed projection is not ready.",
        endpoint_terms=["endpoint A", "endpoint B"],
        identity_terms=["synthetic identity"],
        projection_ids=[],
        full_projection_ids=[],
        lower_order_projection_ids=[],
        candidates=rows,
        candidate_count=len(rows),
        abstract_candidate_count=sum(bool(row.abstract) for row in rows),
        typed_identity_excluded_work_count=0,
        source_unique_work_count=len(rows),
        max_review_works=20,
    )


def test_unprojected_relation_fails_closed_as_insufficient_metadata():
    candidate_set = _unprojected_set()

    review = compile_projection_relation_claim_review(
        candidate_set=candidate_set,
        draft=ProjectionRelationClaimReviewDraft(
            matches=[],
            interpretation=(
                "The typed relation was not ready for grounded projection, "
                "so bounded relation adjudication could not be performed."
            ),
        ),
        projection_by_id={},
    )

    assert review.relation_state == "INSUFFICIENT_METADATA"
    assert review.presented_work_count == 0
    assert review.classified_work_count == 0
    assert review.unclassified_work_ids == []


def test_unprojected_relation_cannot_carry_review_candidates():
    try:
        _unprojected_set(candidates=[_candidate()])
    except ValueError as exc:
        assert (
            "unprojected relation cannot carry bounded review candidates"
            in str(exc)
        )
    else:
        raise AssertionError(
            "expected unprojected relation with review candidates to fail"
        )


def test_zero_projection_state_is_not_no_material_signal():
    review = compile_projection_relation_claim_review(
        candidate_set=_unprojected_set(),
        draft=ProjectionRelationClaimReviewDraft(
            matches=[],
            interpretation="Typed relation was not projection-ready.",
        ),
        projection_by_id={},
    )

    assert review.relation_state != "NO_MATERIAL_RELATION_SIGNAL"
    assert review.relation_state == "INSUFFICIENT_METADATA"
