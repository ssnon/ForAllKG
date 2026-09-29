from __future__ import annotations

from pathlib import Path


def test_candidate_replay_audit_separates_raw_sha_from_authority():
    helper = Path(
        "scripts/discovery/run_higher_order_shadow_lane.py"
    ).read_text(encoding="utf-8")

    assert '"replay_sha_matches_canonical"' in helper
    assert '"replay_authoritative_payload_matches_canonical": True' in helper
    assert '"all_discovery_bundle_fields_except_"' in helper
    assert '"bundle_sha256_and_warnings_exact"' in helper


def test_direct_ho_lane_uses_authoritative_replay_guard_not_raw_sha():
    lane = Path(
        "scripts/discovery/run_direct_higher_order_shadow_lane.py"
    ).read_text(encoding="utf-8")

    assert '"replay_authoritative_payload_matches_canonical"' in lane
    assert "candidate modifier replay changed the authoritative " in lane
    assert "DiscoveryBundle payload" in lane
    assert (
        "candidate modifier replay SHA does not match canonical bundle"
        not in lane
    )
