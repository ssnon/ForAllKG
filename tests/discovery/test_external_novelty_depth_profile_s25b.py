
from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ClaimSearchCoverage,
)
from pipeline_core.discovery.external_novelty import (
    _novelty_depth_profile,
)


def _review(claim_id: str, status: str) -> ClaimPriorArtReview:
    return ClaimPriorArtReview(
        hypothesis_id="h1",
        claim_id=claim_id,
        claim_text=f"claim {claim_id}",
        importance="core",
        status=status,
        matches=[],
        coverage=ClaimSearchCoverage(
            claim_id=claim_id,
            query_count=2,
            successful_query_count=2,
            unique_work_count=10,
            abstract_work_count=5,
            reviewed_work_count=0,
        ),
        reason_codes=[],
        reviewer_unknown_work_ids=[],
        interpretation="bounded review",
    )


def _claim(claim_id: str, role: str | None):
    return SimpleNamespace(
        claim_id=claim_id,
        hypothesis_id="h1",
        novelty_selection_role=role,
    )


def test_s25b_single_novelty_bearing_gap_is_central() -> None:
    reviews = [
        _review("c1", "DIRECT_PRIOR_ART"),
        _review("c2", "COMPONENTS_ONLY"),
    ]
    claims = {
        "c1": _claim("c1", "REQUIRED_ENABLING_RELATION"),
        "c2": _claim("c2", "NOVELTY_BEARING"),
    }

    profile = _novelty_depth_profile(
        hypothesis_id="h1",
        reviews=reviews,
        claims_by_id=claims,
    )

    assert profile.role_binding_complete is True
    assert profile.core_claim_count == 2
    assert profile.relation_backed_core_claim_count == 1
    assert profile.known_core_relation_fraction == 0.5
    assert profile.novelty_bearing_claim_ids == ["c2"]
    assert profile.gap_centrality == "CENTRAL"
    assert (
        profile.novelty_bearing_prior_art_state
        == "ALL_GAP_LIKE"
    )
    assert profile.production_selection_authority is False


def test_s25b_relation_backed_novelty_bearing_is_visible() -> None:
    reviews = [
        _review("c1", "DIRECT_PRIOR_ART"),
        _review("c2", "PARTIAL_PRIOR_ART"),
        _review("c3", "COMPONENTS_ONLY"),
    ]
    claims = {
        "c1": _claim("c1", "REQUIRED_ENABLING_RELATION"),
        "c2": _claim("c2", "NOVELTY_BEARING"),
        "c3": _claim("c3", "AUXILIARY"),
    }

    profile = _novelty_depth_profile(
        hypothesis_id="h1",
        reviews=reviews,
        claims_by_id=claims,
    )

    assert (
        profile.novelty_bearing_relation_backed_claim_ids
        == ["c2"]
    )
    assert (
        profile.novelty_bearing_relation_backed_fraction
        == 1.0
    )
    assert (
        profile.novelty_bearing_prior_art_state
        == "ALL_RELATION_BACKED"
    )


def test_s25b_multiple_novelty_bearing_claims_are_distributed() -> None:
    reviews = [
        _review("c1", "NO_DIRECT_MATCH_FOUND"),
        _review("c2", "COMPONENTS_ONLY"),
    ]
    claims = {
        "c1": _claim("c1", "NOVELTY_BEARING"),
        "c2": _claim("c2", "NOVELTY_BEARING"),
    }

    profile = _novelty_depth_profile(
        hypothesis_id="h1",
        reviews=reviews,
        claims_by_id=claims,
    )

    assert profile.gap_centrality == "DISTRIBUTED"
    assert (
        profile.novelty_bearing_prior_art_state
        == "ALL_GAP_LIKE"
    )


def test_s25b_missing_role_binding_is_unresolved() -> None:
    reviews = [
        _review("c1", "DIRECT_PRIOR_ART"),
        _review("c2", "COMPONENTS_ONLY"),
    ]
    claims = {
        "c1": _claim("c1", "REQUIRED_ENABLING_RELATION"),
        "c2": _claim("c2", None),
    }

    profile = _novelty_depth_profile(
        hypothesis_id="h1",
        reviews=reviews,
        claims_by_id=claims,
    )

    assert profile.role_binding_complete is False
    assert profile.gap_centrality == "UNRESOLVED"
    assert profile.novelty_bearing_claim_count == 0
    assert (
        "s25b_incomplete_core_role_binding"
        in profile.reason_codes
    )


def test_s25b_conflicting_novelty_bearing_claim_is_explicit() -> None:
    reviews = [
        _review("c1", "CONFLICTING_PRIOR_ART"),
    ]
    claims = {
        "c1": _claim("c1", "NOVELTY_BEARING"),
    }

    profile = _novelty_depth_profile(
        hypothesis_id="h1",
        reviews=reviews,
        claims_by_id=claims,
    )

    assert (
        profile.novelty_bearing_conflicting_claim_ids
        == ["c1"]
    )
    assert (
        profile.novelty_bearing_prior_art_state
        == "CONFLICTING"
    )
