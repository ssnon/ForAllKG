from __future__ import annotations

import inspect

from scripts.discovery import (
    run_dac_discovery_e2e as e2e,
)


def test_realization_candidate_chain_exists():
    assert callable(
        e2e._run_realization_candidate_chain
    )


def test_realization_candidate_chain_reuses_frozen_axis_plan():
    source = inspect.getsource(
        e2e._run_realization_candidate_chain
    )

    assert (
        '"--axis-plan-input"'
        in source
    )

    assert (
        "frozen_axis_plan"
        in source
    )


def test_realization_candidate_chain_never_rebuilds_axis_plan():
    source = inspect.getsource(
        e2e._run_realization_candidate_chain
    )

    forbidden = (
        "DiscoveryAxisPlanner(",
        "planner.build(",
        "_resolve_axis_plan(",
    )

    for token in forbidden:
        assert token not in source


def test_realization_candidate_chain_treats_empty_alpha4_as_slot_failure():
    source = inspect.getsource(
        e2e._run_realization_candidate_chain
    )

    assert (
        '"ALPHA4_EMPTY"'
        in source
    )

    assert (
        "if hypothesis_count == 0:"
        in source
    )

    assert (
        "return result"
        in source
    )


def test_realization_candidate_chain_reaches_existing_two_pass_scientific_chain():
    source = inspect.getsource(
        e2e._run_realization_candidate_chain
    )

    assert (
        "_run_scientific_novelty_action_shadow_chain("
        in source
    )

    assert (
        '"TWO_PASS_SEMANTIC_EVALUATED"'
        in source
    )


def test_realization_helper_does_not_apply_production_selection():
    source = inspect.getsource(
        e2e._run_realization_candidate_chain
    )

    assert (
        "select_realization_production_winner"
        not in source
    )

    assert (
        "production_selection_changed"
        not in source
    )
