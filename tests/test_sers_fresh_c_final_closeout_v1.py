from campaigns.sers_alpha4_epoch.fresh_c.sers_fresh_c_final_closeout_v1 import (
    DEFAULT_PROTOCOL_PATH,
    FINAL_H1,
    FINAL_H2,
    FINAL_H3,
    UPSTREAM_LINEAGE,
    validate_protocol,
)

def test_final_states_are_exact():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["final_h1_state"] == FINAL_H1
    assert p["final_h2_state"] == FINAL_H2
    assert p["final_h3_state"] == FINAL_H3

def test_scientific_accounting_is_explicit():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["accepted_scientific_outputs"] == 26
    assert p["original_failed_c1b2_scientific_call_attempts"] == 1
    assert p["recovery_scientific_call_attempts"] == 26
    assert p["total_c1b2_scientific_call_attempts"] == 27

def test_epistemic_guards_are_closed():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["external_literature_used_during_c1b2"] is False
    assert p["count_threshold_used"] is False
    assert p["hypothesis_rewrite_performed"] is False
    assert p["hypothesis_upgrade_performed"] is False
    assert p["h2_resurrected"] is False

def test_recovery_does_not_claim_new_reserve():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["new_fresh_reserve_claimed_in_recovery"] is False
    assert p["failed_parent_response_reused"] is False
    assert p["verbatim_quote_evidence_enabled_in_recovery"] is False

def test_closeout_is_deterministic_and_terminal():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["network_calls_during_closeout"] == 0
    assert p["llm_calls_during_closeout"] == 0
    assert p["scientific_text_read_during_closeout"] is False
    assert p["scientific_adjudication_during_closeout"] is False
    assert p["automatic_next_stage_authorized"] is False
    assert p["stop_after_closeout_freeze"] is True

def test_upstream_lineage_is_exact():
    p = validate_protocol(DEFAULT_PROTOCOL_PATH)
    assert p["upstream_lineage"] == UPSTREAM_LINEAGE
