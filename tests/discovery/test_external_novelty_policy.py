from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.external_novelty import ExternalNoveltyAssessor
from pipeline_core.discovery.external_novelty_contracts import (
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
        compiler=Dummy(),
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


def test_known_components_plus_unmatched_core_is_relational_gap() -> None:
    status, _, _ = assessor()._status(
        [review("COMPONENTS_ONLY"), review("NO_DIRECT_MATCH_FOUND")],
        coverage(),
    )
    assert status == "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP"


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


def test_n1_2_strict_majority_relation_backed_is_supported_extension() -> None:
    status, reasons, _ = assessor()._status(
        [
            review("COMPONENTS_ONLY"),
            review("PARTIAL_PRIOR_ART"),
            review("PARTIAL_PRIOR_ART"),
        ],
        coverage(),
    )
    assert status == "LITERATURE_SUPPORTED_EXTENSION"
    assert (
        "majority_core_relations_have_direct_or_partial_prior_art"
        in reasons
    )


def test_n1_2_relation_backed_minority_remains_new_combination() -> None:
    status, reasons, _ = assessor()._status(
        [
            review("COMPONENTS_ONLY"),
            review("PARTIAL_PRIOR_ART"),
            review("COMPONENTS_ONLY"),
        ],
        coverage(),
    )
    assert status == "NEW_COMBINATION_OF_KNOWN_EFFECTS"
    assert "known_relations_with_component_supported_gap" in reasons


def test_n1_2_tied_relation_support_does_not_count_as_majority() -> None:
    status, reasons, _ = assessor()._status(
        [
            review("PARTIAL_PRIOR_ART"),
            review("COMPONENTS_ONLY"),
        ],
        coverage(),
    )
    assert status == "NEW_COMBINATION_OF_KNOWN_EFFECTS"
    assert "known_relations_with_component_supported_gap" in reasons


def test_n1_2_actual_no_direct_gap_stays_new_combination_even_with_majority_support() -> None:
    status, reasons, _ = assessor()._status(
        [
            review("PARTIAL_PRIOR_ART"),
            review("PARTIAL_PRIOR_ART"),
            review("NO_DIRECT_MATCH_FOUND"),
        ],
        coverage(),
    )
    assert status == "NEW_COMBINATION_OF_KNOWN_EFFECTS"
    assert "known_relations_with_unmatched_composite_relation" in reasons


def test_relation_backed_strict_majority_bypasses_absence_gate_for_supported_extension() -> None:
    status, reasons, _ = assessor()._status(
        [
            review("COMPONENTS_ONLY"),
            review("PARTIAL_PRIOR_ART"),
            review("DIRECT_PRIOR_ART"),
        ],
        coverage(False),
    )

    assert status == "LITERATURE_SUPPORTED_EXTENSION"
    assert (
        "majority_core_relations_have_direct_or_partial_prior_art"
        in reasons
    )
    assert (
        "insufficient_coverage_for_absence_based_status"
        not in reasons
    )


def test_relation_backed_minority_still_fails_closed_under_low_absence_coverage() -> None:
    status, reasons, _ = assessor()._status(
        [
            review("COMPONENTS_ONLY"),
            review("PARTIAL_PRIOR_ART"),
            review("COMPONENTS_ONLY"),
        ],
        coverage(False),
    )

    assert status == "INSUFFICIENT_SEARCH_EVIDENCE"
    assert (
        "insufficient_coverage_for_absence_based_status"
        in reasons
    )


def test_no_direct_gap_still_requires_absence_coverage_even_with_relation_majority() -> None:
    status, reasons, _ = assessor()._status(
        [
            review("PARTIAL_PRIOR_ART"),
            review("DIRECT_PRIOR_ART"),
            review("NO_DIRECT_MATCH_FOUND"),
        ],
        coverage(False),
    )

    assert status == "INSUFFICIENT_SEARCH_EVIDENCE"
    assert (
        "insufficient_coverage_for_absence_based_status"
        in reasons
    )
