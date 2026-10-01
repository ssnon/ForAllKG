from __future__ import annotations

from pathlib import Path

from scripts.discovery.recover_atomic_source_binding_bundle import (
    choose_candidate,
)


def test_choose_candidate_prefers_complete_provenance():
    rows = [
        {
            "claim_rank": 1,
            "claim_local_id": "a",
            "proposition_basis": "",
            "relation_endpoint_anchors": [],
            "source_path": "/a.json",
        },
        {
            "claim_rank": 1,
            "claim_local_id": "a",
            "proposition_basis": "A affects B.",
            "relation_endpoint_anchors": ["A", "B"],
            "source_path": "/b.json",
        },
    ]
    chosen = choose_candidate(rows, expected_rank=1)
    assert chosen is not None
    assert chosen["source_path"] == "/b.json"


def test_choose_candidate_fails_closed_on_equal_ambiguity():
    rows = [
        {
            "claim_rank": 1,
            "claim_local_id": "a",
            "proposition_basis": "A affects B.",
            "relation_endpoint_anchors": ["A", "B"],
            "source_path": "/a.json",
        },
        {
            "claim_rank": 1,
            "claim_local_id": "a",
            "proposition_basis": "A changes B.",
            "relation_endpoint_anchors": ["A", "B"],
            "source_path": "/b.json",
        },
    ]
    assert choose_candidate(rows, expected_rank=1) is None
