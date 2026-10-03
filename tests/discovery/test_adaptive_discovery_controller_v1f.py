from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _mixed_retrieval_batch,
    _validate_retrieval_batch,
)


def test_mixed_retrieval_batch_enters_frozen_barrier():
    actions = [
        "RETRIEVE_MORE",
        "SAME_PREMISE_SHARPEN",
    ]
    assert _mixed_retrieval_batch(actions) is True
    assert _validate_retrieval_batch(actions) is True


def test_mixed_retrieval_with_terminal_action_enters_barrier():
    actions = [
        "RETRIEVE_MORE",
        "STOP",
    ]
    assert _mixed_retrieval_batch(actions) is True
    assert _validate_retrieval_batch(actions) is True


def test_pure_retrieval_is_not_mixed_but_reuses_plan():
    actions = [
        "RETRIEVE_MORE",
        "RETRIEVE_MORE",
    ]
    assert _mixed_retrieval_batch(actions) is False
    assert _validate_retrieval_batch(actions) is True


def test_non_retrieval_batch_does_not_reuse_plan():
    actions = [
        "SAME_PREMISE_SHARPEN",
        "AXIS_MUTATION",
    ]
    assert _mixed_retrieval_batch(actions) is False
    assert _validate_retrieval_batch(actions) is False
