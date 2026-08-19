from pathlib import Path

import pytest

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_4 import (
    DiagnosticOpenAlexCatalogProvider,
    load_and_validate_protocol,
    require_openalex_api_key,
)


def test_v24_records_provider_universe_change_explicitly():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_4_protocol.json")
    )
    assert p.old_providers == ["semantic_scholar", "crossref"]
    assert p.providers == ["openalex", "crossref"]
    assert p.provider_substitution_performed is True
    assert p.provider_universe_changed is True
    assert p.provider_substitution_reason == (
        "transport_availability_only_after_repeated_http_429"
    )


def test_v24_keeps_frozen_selection_contract():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_4_protocol.json")
    )
    assert len(p.broad_queries) == 4
    assert p.results_per_query == 100
    assert p.expected_provider_query_executions == 8
    assert p.max_raw_metadata_rows == 800
    assert p.historical_identity_count == 560
    assert p.target_acquired_papers == 25
    assert p.queries_changed_from_v22 is False
    assert p.search_depth_changed_from_v22 is False
    assert p.historical_ledger_changed_from_v22 is False
    assert p.target_count_changed_from_v22 is False
    assert p.blind_ordering_changed_from_v22 is False
    assert p.hypothesis_aware_selection_added is False
    assert p.title_abstract_scoring_added is False
    assert p.scientific_selection_semantics_changed is False


def test_v24_requires_openalex_key_without_printing_value(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        require_openalex_api_key()
    monkeypatch.setenv("OPENALEX_API_KEY", "test-secret")
    require_openalex_api_key()


def test_openalex_transport_is_bounded_and_conservative(monkeypatch):
    provider = DiagnosticOpenAlexCatalogProvider(api_key="test-key")
    assert provider.minimum_interval_seconds == 1.10
    assert provider.max_attempts == 4
    assert provider.base_backoff_seconds == 2.0


def test_v24_still_does_not_consume_fresh_c():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_4_protocol.json")
    )
    assert p.fresh_reserve_c_consumption_occurs_here is False
    assert p.semantic_read_allowed is False
    assert p.automatic_c0_1d_transition_allowed is False
    assert p.llm_calls == 0
