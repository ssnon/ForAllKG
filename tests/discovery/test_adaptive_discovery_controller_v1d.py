import pytest

from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _assert_reused_query_plan_identity,
    _retrieval_only_actions,
    _validate_retrieval_batch,
)


def test_retrieval_only_batch_is_recognized():
    assert (
        _retrieval_only_actions(
            ["RETRIEVE_MORE"]
        )
        is True
    )

    assert (
        _validate_retrieval_batch(
            ["RETRIEVE_MORE"]
        )
        is True
    )


def test_non_retrieval_round_is_not_reuse_mode():
    assert (
        _validate_retrieval_batch(
            ["AXIS_MUTATION"]
        )
        is False
    )


def test_mixed_retrieval_batch_fails_closed():
    with pytest.raises(
        RuntimeError,
        match="mixed verification batch",
    ):
        _validate_retrieval_batch(
            [
                "RETRIEVE_MORE",
                "AXIS_MUTATION",
            ]
        )


def test_reused_query_plan_requires_exact_identity():
    plan = {
        "source_portfolio_id": "p",
        "plan_id": "plan",
        "plan_sha256": "sha",
        "claims": [
            {
                "hypothesis_id": "h",
                "claims": [
                    {
                        "claim_id": "c",
                        "kind": "composite",
                        "text": "A with B affects C",
                    }
                ],
            }
        ],
        "queries": [
            {
                "query_id": "q",
                "claim_id": "c",
                "query_text": "A B C",
            }
        ],
    }

    _assert_reused_query_plan_identity(
        previous=plan,
        current=dict(plan),
    )


def test_retrieval_query_plan_drift_fails_closed():
    previous = {
        "source_portfolio_id": "p",
        "plan_id": "plan",
        "plan_sha256": "sha",
        "claims": [{"claim_id": "c1"}],
        "queries": [{"query_id": "q1"}],
    }
    current = {
        **previous,
        "claims": [{"claim_id": "c2"}],
    }

    with pytest.raises(
        RuntimeError,
        match="changed frozen query-plan field",
    ):
        _assert_reused_query_plan_identity(
            previous=previous,
            current=current,
        )
