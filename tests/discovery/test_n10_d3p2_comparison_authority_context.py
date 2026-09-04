from __future__ import annotations

import pytest

from pipeline_core.discovery.nonobviousness_dual_run_comparison import (
    _authoritative_gate_schema_for_authority,
    _comparison_schema_for_authority,
    _selected_production_fallback_allowed,
)


def test_v1_authority_context_preserves_frozen_schema():
    assert (
        _comparison_schema_for_authority(
            "v1_only"
        )
        == "nonobviousness-dual-run-comparison-v1"
    )

    assert (
        _authoritative_gate_schema_for_authority(
            "v1_only"
        )
        == "scientific-novelty-fallback-gate-v1"
    )


def test_v2_authority_context_uses_promoted_schema():
    assert (
        _comparison_schema_for_authority(
            "v2_production"
        )
        == "nonobviousness-dual-run-comparison-v2"
    )

    assert (
        _authoritative_gate_schema_for_authority(
            "v2_production"
        )
        == "scientific-novelty-fallback-gate-v2"
    )


def test_v1_selected_fallback_remains_v1_boolean():
    assert (
        _selected_production_fallback_allowed(
            runtime_authority_policy=
                "v1_only",
            v1_fallback_allowed=True,
            v2_selection_class=
                "INELIGIBLE",
            v2_positive_authority=False,
        )
        is True
    )


def test_v2_conditional_is_selected_fallback_negative():
    assert (
        _selected_production_fallback_allowed(
            runtime_authority_policy=
                "v2_production",
            v1_fallback_allowed=True,
            v2_selection_class=
                "CONDITIONAL",
            v2_positive_authority=False,
        )
        is False
    )


def test_v2_ineligible_is_selected_fallback_negative():
    assert (
        _selected_production_fallback_allowed(
            runtime_authority_policy=
                "v2_production",
            v1_fallback_allowed=True,
            v2_selection_class=
                "INELIGIBLE",
            v2_positive_authority=False,
        )
        is False
    )


def test_v2_eligible_positive_is_selected_fallback_positive():
    assert (
        _selected_production_fallback_allowed(
            runtime_authority_policy=
                "v2_production",
            v1_fallback_allowed=False,
            v2_selection_class=
                "ELIGIBLE",
            v2_positive_authority=True,
        )
        is True
    )


def test_v2_eligible_without_positive_authority_is_negative():
    assert (
        _selected_production_fallback_allowed(
            runtime_authority_policy=
                "v2_production",
            v1_fallback_allowed=True,
            v2_selection_class=
                "ELIGIBLE",
            v2_positive_authority=False,
        )
        is False
    )


def test_unknown_runtime_authority_policy_fails_closed():
    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        _selected_production_fallback_allowed(
            runtime_authority_policy=
                "unknown",
            v1_fallback_allowed=True,
            v2_selection_class=
                "ELIGIBLE",
            v2_positive_authority=True,
        )
