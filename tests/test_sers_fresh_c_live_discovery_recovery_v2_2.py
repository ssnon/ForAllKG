from pathlib import Path

from dac_her.fresh_c_live_discovery_recovery_v2_2 import (
    EXPECTED_V21_FAILED_ATTEMPT_ID,
    make_transport_diagnostics_payload_v2_2,
    load_and_validate_protocol,
)
from dac_her.literature_catalog_contracts import CatalogQueryExecution


def test_v22_pins_exact_v21_failed_attempt():
    assert EXPECTED_V21_FAILED_ATTEMPT_ID == (
        "sers_fresh_c_live_discovery_recovery_attempt_v2_1:12b9dbc6d57618c9ed15"
    )


def test_v22_is_compatibility_only():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_2_protocol.json")
    )
    assert p.parent_v21_network_epoch_started is True
    assert p.parent_v21_same_epoch_rerun_allowed is False
    assert p.parent_v21_success_artifacts_absent_required is True
    assert p.compatibility_change_only is True
    assert p.diagnostics_builder_protocol_version_independent is True


def test_v22_keeps_all_frozen_search_transport_semantics():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_2_protocol.json")
    )
    assert p.providers == ["semantic_scholar", "crossref"]
    assert len(p.broad_queries) == 4
    assert p.results_per_query == 100
    assert p.expected_provider_query_executions == 8
    assert p.max_raw_metadata_rows == 800
    assert p.historical_identity_count == 560
    assert p.target_acquired_papers == 25
    assert p.transport_policy_changed_from_v21 is False
    assert p.search_queries_changed_from_v21 is False
    assert p.provider_set_changed_from_v21 is False
    assert p.search_depth_changed_from_v21 is False
    assert p.historical_ledger_changed_from_v21 is False
    assert p.target_count_changed_from_v21 is False
    assert p.blind_ordering_changed_from_v21 is False
    assert p.scientific_selection_semantics_changed_from_v21 is False


def test_diagnostics_builder_does_not_require_protocol_shape():
    execution = CatalogQueryExecution(
        query_id="q1",
        axis_id="a",
        provider="semantic_scholar",
        success=False,
        result_count=0,
        elapsed_seconds=1.0,
        error="RecoveryProviderHTTPError: HTTP status 429",
    )
    payload = make_transport_diagnostics_payload_v2_2(
        protocol_id="protocol:test",
        parent_attempt_id="attempt:test",
        broad_queries=["secret query"],
        executions=[execution],
        semantic_scholar_attempts=[],
    )
    assert payload["protocol_id"] == "protocol:test"
    assert payload["recovery_parent_attempt_id"] == "attempt:test"
    assert payload["provider_executions"][0]["http_status"] == 429
    assert payload["fresh_reserve_c_consumed"] is False
    assert payload["semantic_read_performed"] is False


def test_v22_does_not_consume_fresh_c():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_2_protocol.json")
    )
    assert p.fresh_reserve_c_consumption_occurs_here is False
    assert p.semantic_read_allowed is False
    assert p.automatic_c0_1d_transition_allowed is False
    assert p.llm_calls == 0
