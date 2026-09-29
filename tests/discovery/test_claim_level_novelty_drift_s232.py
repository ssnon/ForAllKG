from types import SimpleNamespace

from scripts.discovery.run_s232_claim_level_novelty_drift_audit import (
    classify_claim_drift,
)


def match(
    work_id,
    relationship,
    *,
    confidence=0.9,
    abstract=True,
):
    return SimpleNamespace(
        work_id=work_id,
        relationship=relationship,
        confidence=confidence,
        abstract_available=abstract,
        title="t",
        doi="10.1/x",
    )


def review(
    status,
    matches,
    *,
    abstracts=4,
    unique=8,
):
    return SimpleNamespace(
        status=status,
        claim_text="claim",
        importance="core",
        matches=matches,
        coverage=SimpleNamespace(
            query_count=2,
            successful_query_count=2,
            unique_work_count=unique,
            abstract_work_count=abstracts,
            reviewed_work_count=len(matches),
        ),
        reason_codes=[],
        reviewer_unknown_work_ids=[],
    )


POLICY = SimpleNamespace(
    direct_match_confidence=0.70,
    min_match_confidence=0.65,
)


def test_same_relation_backed_work_relabeled_is_reviewer_drift():
    before = review(
        "PARTIAL_PRIOR_ART",
        [
            match(
                "w1",
                "PARTIAL_PRIOR_ART",
            )
        ],
    )
    after = review(
        "COMPONENTS_ONLY",
        [
            match(
                "w1",
                "COMPONENT_ONLY",
            )
        ],
        abstracts=5,
    )

    classification, detail = (
        classify_claim_drift(
            before_review=before,
            after_review=after,
            before_policy=POLICY,
            after_policy=POLICY,
        )
    )

    assert (
        classification
        == "RELATION_BACKED_LOST_SAME_WORK_RELABELED"
    )
    assert len(
        detail[
            "lost_relation_backed_same_work_relabeled"
        ]
    ) == 1


def test_lost_match_with_non_decreasing_coverage_is_separate():
    before = review(
        "DIRECT_PRIOR_ART",
        [
            match(
                "w1",
                "DIRECT_PRIOR_ART",
            )
        ],
    )
    after = review(
        "NO_DIRECT_MATCH_FOUND",
        [],
        abstracts=5,
        unique=8,
    )

    classification, _ = (
        classify_claim_drift(
            before_review=before,
            after_review=after,
            before_policy=POLICY,
            after_policy=POLICY,
        )
    )

    assert (
        classification
        == "RELATION_BACKED_LOST_MATCH_DISAPPEARED_WITH_NONDECREASING_COVERAGE"
    )


def test_relation_backed_gain_is_recorded():
    before = review(
        "COMPONENTS_ONLY",
        [
            match(
                "w1",
                "COMPONENT_ONLY",
            )
        ],
    )
    after = review(
        "PARTIAL_PRIOR_ART",
        [
            match(
                "w1",
                "PARTIAL_PRIOR_ART",
            )
        ],
    )

    classification, _ = (
        classify_claim_drift(
            before_review=before,
            after_review=after,
            before_policy=POLICY,
            after_policy=POLICY,
        )
    )

    assert (
        classification
        == "RELATION_BACKED_GAINED"
    )
