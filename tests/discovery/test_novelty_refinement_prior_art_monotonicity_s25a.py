
from __future__ import annotations

import pytest

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
    PriorArtMatch,
)
from pipeline_core.discovery.novelty_refinement_runtime import (
    _s25a_assert_unchanged_hypothesis_no_prior_art_upgrade,
    _s25a_merge_claim_review,
)


def _match(
    *,
    work_id: str,
    relationship: str,
    confidence: float = 0.95,
    relevance: float = 0.8,
) -> PriorArtMatch:
    return PriorArtMatch(
        work_id=work_id,
        relationship=relationship,
        confidence=confidence,
        rationale=f"{relationship} evidence",
        relevance_score=relevance,
        semantic_similarity=0.8,
        lexical_coverage=0.5,
        reaction_domain_relevance=1.0,
        catalyst_scope_relevance=1.0,
        scope_compatible_for_conflict=True,
        scope_reason_codes=[],
        title=f"work {work_id}",
        year=2025,
        doi=None,
        url=None,
        abstract_available=True,
    )


def _review(
    *,
    status: str,
    matches: list[PriorArtMatch],
    query_count: int = 2,
    work_count: int = 8,
) -> ClaimPriorArtReview:
    return ClaimPriorArtReview(
        hypothesis_id="h1",
        claim_id="c1",
        claim_text="A conditions B under C.",
        importance="core",
        status=status,
        matches=matches,
        coverage=ClaimSearchCoverage(
            claim_id="c1",
            query_count=query_count,
            successful_query_count=query_count,
            unique_work_count=work_count,
            abstract_work_count=max(1, work_count // 2),
            reviewed_work_count=len(matches),
        ),
        reason_codes=[],
        reviewer_unknown_work_ids=[],
        interpretation="bounded review",
    )


def test_s25a_direct_prior_art_cannot_disappear_in_targeted_search() -> None:
    source = _review(
        status="DIRECT_PRIOR_ART",
        matches=[
            _match(
                work_id="w_direct",
                relationship="DIRECT_PRIOR_ART",
                confidence=0.99,
            )
        ],
    )
    targeted = _review(
        status="COMPONENTS_ONLY",
        matches=[
            _match(
                work_id="w_component",
                relationship="COMPONENT_ONLY",
            )
        ],
        query_count=4,
        work_count=12,
    )

    merged = _s25a_merge_claim_review(source, targeted)

    assert merged.status == "DIRECT_PRIOR_ART"
    assert {
        row.work_id: row.relationship
        for row in merged.matches
    } == {
        "w_direct": "DIRECT_PRIOR_ART",
        "w_component": "COMPONENT_ONLY",
    }
    assert (
        "s25a_prior_art_positive_evidence_monotonic_floor"
        in merged.reason_codes
    )
    assert merged.coverage.query_count == 4
    assert merged.coverage.unique_work_count == 12


def test_s25a_targeted_search_may_find_stronger_direct_prior_art() -> None:
    source = _review(
        status="PARTIAL_PRIOR_ART",
        matches=[
            _match(
                work_id="w_partial",
                relationship="PARTIAL_PRIOR_ART",
            )
        ],
    )
    targeted = _review(
        status="DIRECT_PRIOR_ART",
        matches=[
            _match(
                work_id="w_direct",
                relationship="DIRECT_PRIOR_ART",
            )
        ],
    )

    merged = _s25a_merge_claim_review(source, targeted)
    assert merged.status == "DIRECT_PRIOR_ART"


def test_s25a_new_conflicting_prior_art_is_not_hidden_by_direct_floor() -> None:
    source = _review(
        status="DIRECT_PRIOR_ART",
        matches=[
            _match(
                work_id="w_direct",
                relationship="DIRECT_PRIOR_ART",
            )
        ],
    )
    targeted = _review(
        status="CONFLICTING_PRIOR_ART",
        matches=[
            _match(
                work_id="w_conflict",
                relationship="CONFLICTING_PRIOR_ART",
            )
        ],
    )

    merged = _s25a_merge_claim_review(source, targeted)
    assert merged.status == "CONFLICTING_PRIOR_ART"


def test_s25a_same_work_keeps_stronger_compiled_relationship() -> None:
    source = _review(
        status="DIRECT_PRIOR_ART",
        matches=[
            _match(
                work_id="w_same",
                relationship="DIRECT_PRIOR_ART",
                confidence=0.99,
            )
        ],
    )
    targeted = _review(
        status="PARTIAL_PRIOR_ART",
        matches=[
            _match(
                work_id="w_same",
                relationship="PARTIAL_PRIOR_ART",
                confidence=0.99,
            )
        ],
    )

    merged = _s25a_merge_claim_review(source, targeted)

    assert len(merged.matches) == 1
    assert merged.matches[0].relationship == "DIRECT_PRIOR_ART"


def test_s25a_unchanged_prior_art_backed_hypothesis_cannot_upgrade() -> None:
    with pytest.raises(
        RuntimeError,
        match="made an unchanged prior-art-backed hypothesis more novel",
    ):
        _s25a_assert_unchanged_hypothesis_no_prior_art_upgrade(
            source_status="LITERATURE_SUPPORTED_EXTENSION",
            targeted_status="NEW_COMBINATION_OF_KNOWN_EFFECTS",
        )


def test_s25a_more_prior_art_is_allowed() -> None:
    _s25a_assert_unchanged_hypothesis_no_prior_art_upgrade(
        source_status="LITERATURE_SUPPORTED_EXTENSION",
        targeted_status="WELL_ESTABLISHED",
    )
