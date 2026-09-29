from scripts.discovery.run_s242_research_value_blind_pairwise_validity import (
    profile_key,
)


def test_profile_key_is_order_independent():
    assert profile_key({"b": "2", "a": "1"}) == profile_key(
        {"a": "1", "b": "2"}
    )


def test_profile_key_changes_when_profile_changes():
    assert profile_key({"a": "1"}) != profile_key({"a": "2"})
