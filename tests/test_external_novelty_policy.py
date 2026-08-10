from __future__ import annotations

from types import SimpleNamespace

from dac_her.external_novelty import ExternalNoveltyAssessor
from dac_her.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    ExternalNoveltyPolicy,
    HypothesisSearchCoverage,
)


class Dummy:
    pass


def review(status: str, *, importance: str = "core") -> ClaimPriorArtReview:
    return ClaimPriorArtReview(
        hypothesis_id="h1",
        claim_id="c1:" + status,
        claim_text=status,
        importance=importance,
        status=status,
        matches=[],
        coverage=ClaimSearchCoverage(
            claim_id="c1:" + status,
            query_count=2,
            successful_query_count=2,
            unique_work_count=8,
            abstract_work_count=5,
            reviewed_work_count=5,
        ),
        interpretation="fixture",
    )


def assessor() -> ExternalNoveltyAssessor:
    return ExternalNoveltyAssessor(
        decomposer=Dummy(),
        ranker=Dummy(),
        review_backend=Dummy(),
        policy=ExternalNoveltyPolicy(),
    )


def coverage(sufficient: bool = True) -> HypothesisSearchCoverage:
    return HypothesisSearchCoverage(
        hypothesis_id="h1",
        query_count=5,
        successful_query_count=5,
        provider_success_count=2,
        unique_work_count=20,
        abstract_work_count=12,
        core_claim_count=2,
        core_claims_with_minimum_abstract_coverage=2,
        sufficient_for_absence_based_novelty=sufficient,
    )


def test_direct_core_claims_are_well_established() -> None:
    status, _, _ = assessor()._status(
        [review("DIRECT_PRIOR_ART"), review("DIRECT_PRIOR_ART")],
        coverage(),
    )
    assert status == "WELL_ESTABLISHED"


def test_known_components_plus_unmatched_core_is_new_combination() -> None:
    status, _, _ = assessor()._status(
        [review("COMPONENTS_ONLY"), review("NO_DIRECT_MATCH_FOUND")],
        coverage(),
    )
    assert status == "NEW_COMBINATION_OF_KNOWN_EFFECTS"


def test_absence_based_status_fails_closed_under_low_coverage() -> None:
    status, _, _ = assessor()._status(
        [review("NO_DIRECT_MATCH_FOUND")],
        coverage(False),
    )
    assert status == "INSUFFICIENT_SEARCH_EVIDENCE"


def test_title_only_core_needs_absence_coverage_and_does_not_count_as_supported_extension() -> None:
    status, _, _ = assessor()._status(
        [review("PARTIAL_PRIOR_ART"), review("TITLE_ONLY_NEIGHBORS")],
        coverage(False),
    )
    assert status == "INSUFFICIENT_SEARCH_EVIDENCE"
