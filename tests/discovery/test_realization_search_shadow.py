from __future__ import annotations

import pytest

from pipeline_core.discovery.realization_search_shadow import (
    RealizationSearchPolicy,
    RealizationSemanticObservation,
    select_realization_shadow_winner,
)


AGG = (
    "semantic-distinctiveness-aggregation-v2.1"
)

MODEL = "openai/gpt-5.6-luna"


def obs(
    slot: int,
    tier1: str,
    tier2: str,
    *,
    hypothesis_id: str | None = None,
    model1: str = MODEL,
    model2: str = MODEL,
    diagnostic_only: tuple[
        bool,
        bool,
    ] = (
        True,
        True,
    ),
):
    return RealizationSemanticObservation(
        slot_index=slot,
        hypothesis_id=(
            hypothesis_id
            or f"hypothesis:test-{slot}"
        ),
        pass_tiers=(
            tier1,
            tier2,
        ),
        pass_aggregation_versions=(
            AGG,
            AGG,
        ),
        pass_served_models=(
            model1,
            model2,
        ),
        pass_diagnostic_only=(
            diagnostic_only
        ),
        pass_action_policy_applied=(
            False,
            False,
        ),
        pass_scientific_selection_changed=(
            False,
            False,
        ),
    )


def test_default_policy_uses_width_three_and_retains_one():
    policy = RealizationSearchPolicy()

    assert policy.search_width == 3
    assert policy.retained_hypotheses_per_axis == 1
    assert policy.shadow_only is True


def test_width_one_remains_available_as_control():
    policy = RealizationSearchPolicy(
        search_width=1
    )

    assert policy.search_width == 1
    assert policy.shadow_only is True


def test_width_three_selects_stable_high():
    report = select_realization_shadow_winner(
        [
            obs(
                0,
                "MODERATE",
                "MODERATE",
            ),
            obs(
                1,
                "HIGH",
                "HIGH",
            ),
            obs(
                2,
                "MODERATE",
                "MODERATE",
            ),
        ],
        policy=RealizationSearchPolicy(
            search_width=3
        ),
    )

    assert report.status == (
        "WINNER_SELECTED"
    )

    assert report.winner_slot_index == 1
    assert report.winner_tier == "HIGH"

    assert (
        report.production_selection_changed
        is False
    )

    assert report.shadow_only is True


def test_unstable_high_moderate_is_not_eligible():
    report = select_realization_shadow_winner(
        [
            obs(
                0,
                "HIGH",
                "MODERATE",
            ),
            obs(
                1,
                "MODERATE",
                "MODERATE",
            ),
            obs(
                2,
                "INDETERMINATE",
                "INDETERMINATE",
            ),
        ],
        policy=RealizationSearchPolicy(
            search_width=3
        ),
    )

    assert report.winner_slot_index == 1
    assert report.winner_tier == "MODERATE"

    first = report.candidates[0]

    assert first.eligible is False

    assert (
        "SEMANTIC_TIER_UNSTABLE"
        in first.reason_codes
    )


def test_indeterminate_is_not_ranked_as_low():
    report = select_realization_shadow_winner(
        [
            obs(
                0,
                "INDETERMINATE",
                "INDETERMINATE",
            ),
            obs(
                1,
                "LOW",
                "LOW",
            ),
            obs(
                2,
                "INDETERMINATE",
                "INDETERMINATE",
            ),
        ],
        policy=RealizationSearchPolicy(
            search_width=3
        ),
    )

    assert report.winner_slot_index == 1
    assert report.winner_tier == "LOW"


def test_no_stable_determinate_candidate_fails_closed():
    report = select_realization_shadow_winner(
        [
            obs(
                0,
                "HIGH",
                "MODERATE",
            ),
            obs(
                1,
                "INDETERMINATE",
                "INDETERMINATE",
            ),
            obs(
                2,
                "MODERATE",
                "HIGH",
            ),
        ],
        policy=RealizationSearchPolicy(
            search_width=3
        ),
    )

    assert report.status == (
        "NO_STABLE_DETERMINATE_CANDIDATE"
    )

    assert report.winner_hypothesis_id is None
    assert report.winner_tier is None


def test_tier_tie_breaks_to_earliest_slot():
    report = select_realization_shadow_winner(
        [
            obs(
                0,
                "HIGH",
                "HIGH",
            ),
            obs(
                1,
                "HIGH",
                "HIGH",
            ),
            obs(
                2,
                "MODERATE",
                "MODERATE",
            ),
        ],
        policy=RealizationSearchPolicy(
            search_width=3
        ),
    )

    assert report.winner_slot_index == 0


def test_requires_exact_slot_set():
    with pytest.raises(
        ValueError,
        match="slots",
    ):
        select_realization_shadow_winner(
            [
                obs(
                    0,
                    "HIGH",
                    "HIGH",
                ),
                obs(
                    2,
                    "HIGH",
                    "HIGH",
                ),
                obs(
                    3,
                    "HIGH",
                    "HIGH",
                ),
            ],
            policy=RealizationSearchPolicy(
                search_width=3
            ),
        )


def test_rejects_duplicate_hypotheses():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        select_realization_shadow_winner(
            [
                obs(
                    0,
                    "HIGH",
                    "HIGH",
                    hypothesis_id="hypothesis:same",
                ),
                obs(
                    1,
                    "MODERATE",
                    "MODERATE",
                    hypothesis_id="hypothesis:same",
                ),
            ],
            policy=RealizationSearchPolicy(
                search_width=2
            ),
        )


def test_rejects_semantic_contract_mutation():
    with pytest.raises(
        ValueError,
        match="diagnostic-only",
    ):
        select_realization_shadow_winner(
            [
                obs(
                    0,
                    "HIGH",
                    "HIGH",
                    diagnostic_only=(
                        False,
                        False,
                    ),
                ),
            ],
            policy=RealizationSearchPolicy(
                search_width=1
            ),
        )


def test_rejects_model_drift_across_passes():
    with pytest.raises(
        ValueError,
        match="served model changed",
    ):
        select_realization_shadow_winner(
            [
                obs(
                    0,
                    "HIGH",
                    "HIGH",
                    model2="different-model",
                ),
            ],
            policy=RealizationSearchPolicy(
                search_width=1
            ),
        )


def test_width_bounds_are_explicit():
    with pytest.raises(
        ValueError
    ):
        RealizationSearchPolicy(
            search_width=0
        )

    with pytest.raises(
        ValueError
    ):
        RealizationSearchPolicy(
            search_width=5
        )


def test_retained_width_may_not_exceed_search_width():
    with pytest.raises(
        ValueError,
        match="may not exceed",
    ):
        RealizationSearchPolicy(
            search_width=2,
            retained_hypotheses_per_axis=3,
        )


def test_multi_retention_policy_is_representable():
    policy = RealizationSearchPolicy(
        search_width=3,
        retained_hypotheses_per_axis=2,
    )

    assert policy.search_width == 3
    assert policy.retained_hypotheses_per_axis == 2


def test_single_winner_selector_fails_closed_for_multi_retention():
    policy = RealizationSearchPolicy(
        search_width=3,
        retained_hypotheses_per_axis=2,
    )

    with pytest.raises(
        ValueError,
        match="diversity-aware retention selector",
    ):
        select_realization_shadow_winner(
            [
                obs(
                    0,
                    "HIGH",
                    "HIGH",
                ),
                obs(
                    1,
                    "HIGH",
                    "HIGH",
                ),
                obs(
                    2,
                    "MODERATE",
                    "MODERATE",
                ),
            ],
            policy=policy,
        )


def test_retained_width_lower_bound():
    with pytest.raises(
        ValueError,
    ):
        RealizationSearchPolicy(
            search_width=3,
            retained_hypotheses_per_axis=0,
        )

