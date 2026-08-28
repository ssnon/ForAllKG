from __future__ import annotations

from pipeline_core.discovery.realization_search_production import (
    select_realization_production_winner,
)
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
) -> RealizationSemanticObservation:
    return RealizationSemanticObservation(
        slot_index=slot,
        hypothesis_id=(
            f"hypothesis:production-{slot}"
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
            MODEL,
            MODEL,
        ),
        pass_diagnostic_only=(
            True,
            True,
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


def test_production_selector_retains_stable_high():
    observations = [
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
            "INDETERMINATE",
            "INDETERMINATE",
        ),
    ]

    report = (
        select_realization_production_winner(
            observations,
            policy=RealizationSearchPolicy(),
        )
    )

    assert report.status == (
        "WINNER_SELECTED"
    )

    assert report.winner_slot_index == 1

    assert report.winner_hypothesis_id == (
        "hypothesis:production-1"
    )

    assert report.winner_tier == "HIGH"

    assert report.production_authority is True

    assert (
        report.production_selection_changed
        is True
    )

    assert (
        report.semantic_diagnostic_contract_preserved
        is True
    )


def test_production_selector_fail_closes_without_stable_determinate():
    observations = [
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
    ]

    report = (
        select_realization_production_winner(
            observations,
            policy=RealizationSearchPolicy(),
        )
    )

    assert report.status == (
        "NO_STABLE_DETERMINATE_CANDIDATE"
    )

    assert report.winner_slot_index is None
    assert report.winner_hypothesis_id is None
    assert report.winner_tier is None

    assert report.production_authority is True


def test_production_ranking_is_identical_to_frozen_shadow_ranking():
    observations = [
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
    ]

    policy = RealizationSearchPolicy()

    shadow = (
        select_realization_shadow_winner(
            observations,
            policy=policy,
        )
    )

    production = (
        select_realization_production_winner(
            observations,
            policy=policy,
        )
    )

    assert (
        production.status
        == shadow.status
    )

    assert (
        production.winner_slot_index
        == shadow.winner_slot_index
    )

    assert (
        production.winner_hypothesis_id
        == shadow.winner_hypothesis_id
    )

    assert (
        production.winner_tier
        == shadow.winner_tier
    )

    # Frozen shadow semantics remain observational.
    assert (
        shadow.production_selection_changed
        is False
    )

    # Only the new wrapper carries production authority.
    assert (
        production.production_selection_changed
        is True
    )
