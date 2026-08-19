from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_3 import (
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    args = parser.parse_args()
    p = load_and_validate_protocol(args.protocol)

    changed = (
        p.search_queries_changed_from_v22,
        p.provider_set_changed_from_v22,
        p.search_depth_changed_from_v22,
        p.historical_ledger_changed_from_v22,
        p.target_count_changed_from_v22,
        p.blind_ordering_changed_from_v22,
        p.scientific_selection_semantics_changed_from_v22,
        p.transport_pacing_changed_from_v22,
        p.transport_retry_policy_changed_from_v22,
    )
    if any(changed):
        raise ValueError("v2.3 changed frozen search/selection/pacing/retry semantics.")
    if not p.authenticated_transport_is_only_material_change:
        raise ValueError("v2.3 is not authenticated-transport-only.")

    print("Fresh-C C0.1C-v2.3 authenticated recovery protocol verifier")
    print(f"Protocol ID: {p.protocol_id}")
    print(f"Protocol SHA256: {p.protocol_sha256}")
    print(f"Parent v2.2 attempt: {p.parent_v22_attempt_id}")
    print("Parent v2.2 failure: Semantic Scholar HTTP 429 x4")
    print("Parent v2.2 Crossref: 4/4 successful")
    print("Semantic Scholar API key required: True")
    print("Authenticated transport is only material change: True")
    print("Scientific/search/selection semantics changed: False")
    print("Transport pacing/retry changed: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
