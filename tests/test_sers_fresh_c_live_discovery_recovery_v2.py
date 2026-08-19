from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import Message

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2 import (
    EXPECTED_V1_FAILED_ATTEMPT_ID,
    RETRYABLE_HTTP_STATUS,
    SEMANTIC_SCHOLAR_BASE_BACKOFF_SECONDS,
    SEMANTIC_SCHOLAR_MAX_ATTEMPTS,
    SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS,
    DiagnosticSemanticScholarCatalogProvider,
    _retry_after_seconds,
    _sanitize_error_summary,
    load_and_validate_protocol,
)


def test_recovery_transport_policy_is_conservative():
    provider = DiagnosticSemanticScholarCatalogProvider(api_key=None)
    assert provider.minimum_interval_seconds == (
        SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS
    ) == 1.10
    assert provider.max_attempts == SEMANTIC_SCHOLAR_MAX_ATTEMPTS == 4
    assert provider.base_backoff_seconds == (
        SEMANTIC_SCHOLAR_BASE_BACKOFF_SECONDS
    ) == 2.0
    assert RETRYABLE_HTTP_STATUS == {429, 500, 502, 503, 504}


def test_error_summary_redacts_frozen_query_text():
    query = "surface enhanced Raman spectroscopy gold silver"
    summary = _sanitize_error_summary(
        "HTTP 429 for " + query,
        forbidden_queries=[query],
    )
    assert query not in summary
    assert "<query-redacted>" in summary


def test_retry_after_numeric_is_clamped():
    headers = Message()
    headers["Retry-After"] = "999"
    assert _retry_after_seconds(
        headers,
        max_seconds=60.0,
    ) == 60.0


def test_retry_after_http_date_is_supported():
    headers = Message()
    future = datetime.now(timezone.utc) + timedelta(seconds=10)
    headers["Retry-After"] = future.strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    value = _retry_after_seconds(headers, max_seconds=60.0)
    assert value is not None
    assert 0.0 <= value <= 60.0


def test_protocol_binds_exact_failed_parent():
    protocol = load_and_validate_protocol(
        __import__("pathlib").Path(
            "dac_her/sers_fresh_c_live_discovery_recovery_v2_protocol.json"
        )
    )
    assert protocol.recovery_parent_attempt_id == (
        EXPECTED_V1_FAILED_ATTEMPT_ID
    )
    assert protocol.recovery_parent_failed_epoch_must_be_preserved is True
    assert protocol.recovery_parent_same_epoch_rerun_allowed is False


def test_protocol_keeps_scientific_search_semantics_unchanged():
    protocol = load_and_validate_protocol(
        __import__("pathlib").Path(
            "dac_her/sers_fresh_c_live_discovery_recovery_v2_protocol.json"
        )
    )
    assert protocol.providers == ["semantic_scholar", "crossref"]
    assert len(protocol.broad_queries) == 4
    assert protocol.results_per_query == 100
    assert protocol.expected_provider_query_executions == 8
    assert protocol.max_raw_metadata_rows == 800
    assert protocol.historical_identity_count == 560
    assert protocol.target_acquired_papers == 25
    assert protocol.search_queries_changed_from_v1 is False
    assert protocol.provider_set_changed_from_v1 is False
    assert protocol.search_depth_changed_from_v1 is False
    assert protocol.historical_ledger_changed_from_v1 is False
    assert protocol.target_count_changed_from_v1 is False
    assert protocol.blind_ordering_changed_from_v1 is False
    assert (
        protocol.scientific_selection_semantics_changed_from_v1
        is False
    )


def test_protocol_keeps_fresh_c_unconsumed():
    protocol = load_and_validate_protocol(
        __import__("pathlib").Path(
            "dac_her/sers_fresh_c_live_discovery_recovery_v2_protocol.json"
        )
    )
    assert (
        protocol.fresh_reserve_c_consumption_occurs_in_recovery
        is False
    )
    assert protocol.semantic_read_allowed_in_recovery is False
    assert protocol.automatic_c0_1d_transition_allowed is False
    assert protocol.llm_calls == 0
