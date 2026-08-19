from campaigns.sers_standard2.devwide_coverage_audit import sha256_json


def test_hash_canonical():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_hash_changes():
    assert sha256_json({"x": 1}) != sha256_json({"x": 2})
