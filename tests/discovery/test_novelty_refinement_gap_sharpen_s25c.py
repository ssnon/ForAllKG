from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.novelty_refinement_runtime import (
    _s25c_gap_sharpen_final_rejection_reason,
    _s25c_should_attempt_fresh_reaxis,
    _s25c_should_bypass_resolved_candidate_exit,
)


def _profile(*, complete=True, gap_like=None):
    return SimpleNamespace(
        role_binding_complete=complete,
        novelty_bearing_gap_like_claim_ids=list(gap_like or []),
    )


def _card(status, profile=None):
    return SimpleNamespace(
        status=status,
        novelty_depth_profile=profile,
    )


def test_gap_sharpen_bypasses_resolved_candidate_short_circuit():
    assert _s25c_should_bypass_resolved_candidate_exit("gap_sharpen")
    assert not _s25c_should_bypass_resolved_candidate_exit(
        "targeted_search_then_refine"
    )


def test_gap_sharpen_never_uses_fresh_reaxis_escape():
    assert not _s25c_should_attempt_fresh_reaxis(
        action="gap_sharpen",
        ordinary_should_attempt=True,
    )
    assert _s25c_should_attempt_fresh_reaxis(
        action="targeted_search_then_refine",
        ordinary_should_attempt=True,
    )


def test_extension_like_final_candidate_is_rejected():
    assert (
        _s25c_gap_sharpen_final_rejection_reason(
            _card(
                "LITERATURE_SUPPORTED_EXTENSION",
                _profile(gap_like=["c1"]),
            )
        )
        == "s25c_gap_sharpen_remained_literature_supported_extension"
    )


def test_new_combination_without_gap_like_novelty_claim_is_rejected():
    assert (
        _s25c_gap_sharpen_final_rejection_reason(
            _card(
                "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                _profile(gap_like=[]),
            )
        )
        == "s25c_gap_sharpen_no_novelty_bearing_gap_after_refinement"
    )


def test_new_combination_with_gap_like_novelty_claim_is_allowed():
    assert (
        _s25c_gap_sharpen_final_rejection_reason(
            _card(
                "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                _profile(gap_like=["c2"]),
            )
        )
        is None
    )


def test_relational_gap_candidate_is_allowed():
    assert (
        _s25c_gap_sharpen_final_rejection_reason(
            _card(
                "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                _profile(gap_like=["c2"]),
            )
        )
        is None
    )


def test_missing_depth_profile_fails_closed_for_new_combination():
    assert (
        _s25c_gap_sharpen_final_rejection_reason(
            _card("NEW_COMBINATION_OF_KNOWN_EFFECTS", None)
        )
        == "s25c_gap_sharpen_missing_or_incomplete_depth_profile"
    )
