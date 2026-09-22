from __future__ import annotations

from pipeline_core.discovery.reframing.scientific_synthesis_n10_authority import (
    ROLE_AWARE_AUTHORITY_SOURCE,
    ROLE_AWARE_POSITIVE_REQUIREMENT,
    SYNTHESIS_AUTHORITY_SCOPE,
    _stable_id,
)


def test_scientific_synthesis_authority_scope_is_distinct_from_alpha6():
    assert SYNTHESIS_AUTHORITY_SCOPE == "scientific_cross_lane_synthesis_candidate"
    assert SYNTHESIS_AUTHORITY_SCOPE != "alpha6_post_generation_candidate"


def test_role_aware_authority_source_is_preserved():
    assert ROLE_AWARE_AUTHORITY_SOURCE == "n10_role_aware_nonobviousness_v2"


def test_positive_requirement_is_role_aware_eligible():
    assert ROLE_AWARE_POSITIVE_REQUIREMENT == (
        "ELIGIBLE_AND_ROLE_AWARE_POSITIVE_NONOBVIOUSNESS"
    )


def test_stable_ids_are_deterministic():
    assert _stable_id("x", "a", "b") == _stable_id("x", "a", "b")


def test_stable_ids_change_with_lineage():
    assert _stable_id("x", "a") != _stable_id("x", "b")
