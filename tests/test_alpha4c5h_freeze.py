from __future__ import annotations

import json

from dac_her.alpha4c5f2_reserve import (
    make_blind_split,
)
from dac_her.alpha4c5h_freeze import (
    EXPECTED_SPLIT_SEMANTIC_SHA256,
    find_reserve_b_paper_ids,
    make_confirmation_protocol_id,
    make_freeze_id,
    semantic_sha256,
)


PAPERS = [f"P{i:02d}" for i in range(25)]


def test_find_reserve_b_direct_list():
    rows, path = find_reserve_b_paper_ids(
        {
            "partitions": {
                "reserve_b": PAPERS,
            }
        }
    )
    assert rows == sorted(PAPERS)
    assert "reserve_b" in path


def test_find_reserve_b_nested_paper_ids():
    rows, path = find_reserve_b_paper_ids(
        {
            "reserve_b": {
                "paper_ids": PAPERS,
                "sealed": True,
            }
        }
    )
    assert rows == sorted(PAPERS)
    assert "paper_ids" in path


def test_freeze_ids_are_deterministic():
    payload = {
        "a": 1,
        "b": ["x", "y"],
    }
    assert make_freeze_id(payload) == make_freeze_id(payload)
    assert (
        make_confirmation_protocol_id(payload)
        == make_confirmation_protocol_id(payload)
    )


def test_different_payload_changes_freeze_id():
    assert make_freeze_id({"a": 1}) != make_freeze_id({"a": 2})


def test_semantic_sha_is_not_raw_pretty_json_sha_contract():
    # This guards the exact bug fixed in alpha4c.5h.0a: a canonical semantic
    # SHA and a pretty-printed file-byte SHA are different hash domains.
    payload = {"b": 2, "a": 1}
    canonical = semantic_sha256(payload)

    import hashlib

    pretty = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    raw = hashlib.sha256(
        pretty.encode("utf-8")
    ).hexdigest()

    assert canonical != raw


def test_expected_historical_split_sha_is_semantic_sha():
    assert len(EXPECTED_SPLIT_SEMANTIC_SHA256) == 64
