from pathlib import Path
import os
import pytest

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_3 import (
    EXPECTED_V22_FAILED_ATTEMPT_ID,
    load_and_validate_protocol,
    require_api_key_presence,
)


def test_v23_pins_exact_v22_failed_attempt():
    assert EXPECTED_V22_FAILED_ATTEMPT_ID == (
        "sers_fresh_c_live_discovery_recovery_attempt_v2_2:ce325ae6c64aac05be94"
    )


def test_v23_requires_api_key_but_does_not_persist_value(monkeypatch):
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        require_api_key_presence()
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-test-value")
    require_api_key_presence()


def test_v23_keeps_search_and_selection_semantics_unchanged():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_3_protocol.json")
    )
    assert p.providers == ["semantic_scholar", "crossref"]
    assert len(p.broad_queries) == 4
    assert p.results_per_query == 100
    assert p.expected_provider_query_executions == 8
    assert p.max_raw_metadata_rows == 800
    assert p.historical_identity_count == 560
    assert p.target_acquired_papers == 25
    assert p.search_queries_changed_from_v22 is False
    assert p.provider_set_changed_from_v22 is False
    assert p.search_depth_changed_from_v22 is False
    assert p.historical_ledger_changed_from_v22 is False
    assert p.target_count_changed_from_v22 is False
    assert p.blind_ordering_changed_from_v22 is False
    assert p.scientific_selection_semantics_changed_from_v22 is False


def test_v23_changes_authentication_only_not_pacing_or_retry():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_3_protocol.json")
    )
    assert p.semantic_scholar_api_key_required is True
    assert p.authenticated_transport_is_only_material_change is True
    assert p.semantic_scholar_minimum_interval_seconds == 1.1
    assert p.semantic_scholar_max_attempts == 4
    assert p.retryable_http_status == [429, 500, 502, 503, 504]
    assert p.transport_pacing_changed_from_v22 is False
    assert p.transport_retry_policy_changed_from_v22 is False


def test_v23_still_does_not_consume_fresh_c():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_3_protocol.json")
    )
    assert p.fresh_reserve_c_consumption_occurs_here is False
    assert p.semantic_read_allowed is False
    assert p.automatic_c0_1d_transition_allowed is False
    assert p.llm_calls == 0
