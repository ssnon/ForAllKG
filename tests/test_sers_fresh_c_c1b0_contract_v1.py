from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_c1b0_contract_v1 import (
    H1,
    H2,
    H3,
    EXPECTED_R2_DECISIONS,
    canonical_json_sha256,
    validate_protocol,
)


def _protocol():
    return validate_protocol(
        Path("dac_her/sers_fresh_c_c1b0_contract_v1_protocol.json")
    )


def test_protocol_is_metadata_only_and_stops():
    p = _protocol()
    assert p["fresh_c_text_semantic_read_allowed"] is False
    assert p["fresh_c_text_hash_verification_allowed"] is True
    assert p["network_calls_allowed"] is False
    assert p["llm_calls"] == 0
    assert p["automatic_c1b1_transition_allowed"] is False
    assert p["stop_after_audit"] is True


def test_r2_target_partition_is_frozen():
    p = _protocol()
    assert p["scientific_target_hypothesis_ids"] == [H1, H3]
    assert p["terminal_rejected_hypothesis_ids"] == [H2]
    assert p["primary_remaining_candidate_hypothesis_id"] == H3


def test_expected_r2_decisions_are_exact():
    assert EXPECTED_R2_DECISIONS == {
        H1: "KEEP_BOUNDED_EXTENSION",
        H2: "REJECT_AS_FORMULATED",
        H3: "KEEP_RELATIONAL_GAP_CANDIDATE",
    }


def test_protocol_binds_exact_25_source_corpus():
    p = _protocol()
    assert p["source_identity_count"] == 25
    assert (
        p["c1ar1_corpus_sha256"]
        == "cffb7eab1465258b61ea28d64b1a703cb5a2b0cb940da0342bc7c1929db89e19"
    )


def test_canonical_hash_is_deterministic():
    left = {"b": 2, "a": [3, 1]}
    right = {"a": [3, 1], "b": 2}
    assert canonical_json_sha256(left) == canonical_json_sha256(right)
