from __future__ import annotations

from dac_her.standard2_provider_health_probe import sha256_json


def test_hash_is_deterministic():
    left = sha256_json({"b": 2, "a": 1})
    right = sha256_json({"a": 1, "b": 2})
    assert left == right


def test_hash_changes_with_value():
    assert sha256_json({"x": 1}) != sha256_json({"x": 2})
