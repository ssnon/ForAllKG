from pathlib import Path

from dac_her.fresh_c_live_discovery_recovery_v2_1 import (
    EXPECTED_V2_FREEZE_COMMIT,
    load_and_validate_protocol,
)


def test_v21_is_harness_only_and_parent_v2_never_started():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_1_protocol.json")
    )
    assert p.parent_v2_network_epoch_started is False
    assert p.parent_v2_failure_kind == "pre_network_argparse_harness_mismatch"
    assert p.harness_change_only is True


def test_v21_keeps_all_scientific_search_transport_semantics():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_1_protocol.json")
    )
    assert p.providers == ["semantic_scholar", "crossref"]
    assert len(p.broad_queries) == 4
    assert p.results_per_query == 100
    assert p.expected_provider_query_executions == 8
    assert p.max_raw_metadata_rows == 800
    assert p.historical_identity_count == 560
    assert p.target_acquired_papers == 25
    assert p.transport_policy_changed_from_v2 is False
    assert p.search_queries_changed_from_v2 is False
    assert p.provider_set_changed_from_v2 is False
    assert p.search_depth_changed_from_v2 is False
    assert p.historical_ledger_changed_from_v2 is False
    assert p.target_count_changed_from_v2 is False
    assert p.blind_ordering_changed_from_v2 is False
    assert p.scientific_selection_semantics_changed_from_v2 is False


def test_v21_pins_parent_freeze_commit():
    assert EXPECTED_V2_FREEZE_COMMIT == (
        "4edb50343a733857a10c3d599a88c878f1e04958"
    )


def test_v21_still_does_not_consume_fresh_c():
    p = load_and_validate_protocol(
        Path("dac_her/sers_fresh_c_live_discovery_recovery_v2_1_protocol.json")
    )
    assert p.fresh_reserve_c_consumption_occurs_here is False
    assert p.semantic_read_allowed is False
    assert p.automatic_c0_1d_transition_allowed is False
    assert p.llm_calls == 0


def test_runner_source_exposes_exact_preflight_and_confirmation_flags():
    source = Path(
        "scripts/run_sers_fresh_c_live_discovery_recovery_v2_1.py"
    ).read_text(encoding="utf-8")
    assert 'group.add_argument("--preflight", action="store_true")' in source
    assert '"--confirm-live-discovery-recovery-v2-1"' in source
    assert 'args.confirm_live_discovery_recovery_v2_1' in source
    assert '"--confirm-live-discovery-recovery-v2.1-1"' not in source
