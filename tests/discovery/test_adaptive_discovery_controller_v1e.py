import pytest

from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _verification_results_per_query,
)


def test_normal_verification_uses_base_budget():
    assert (
        _verification_results_per_query(
            base=12,
            maximum=30,
            multiplier=1,
        )
        == 12
    )


def test_retrieve_more_doubles_verification_budget():
    assert (
        _verification_results_per_query(
            base=12,
            maximum=30,
            multiplier=2,
        )
        == 24
    )


def test_retrieval_budget_is_capped():
    assert (
        _verification_results_per_query(
            base=20,
            maximum=30,
            multiplier=2,
        )
        == 30
    )


def test_invalid_retrieval_budget_fails_closed():
    with pytest.raises(ValueError):
        _verification_results_per_query(
            base=12,
            maximum=10,
            multiplier=2,
        )
