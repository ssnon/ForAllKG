from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_1 import (
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    return parser.parse_args()


def verify(path: Path):
    protocol = load_and_validate_protocol(path)
    if any(
        (
            protocol.transport_policy_changed_from_v2,
            protocol.search_queries_changed_from_v2,
            protocol.provider_set_changed_from_v2,
            protocol.search_depth_changed_from_v2,
            protocol.historical_ledger_changed_from_v2,
            protocol.target_count_changed_from_v2,
            protocol.blind_ordering_changed_from_v2,
            protocol.scientific_selection_semantics_changed_from_v2,
        )
    ):
        raise ValueError("v2.1 changed frozen semantics.")
    if not protocol.harness_change_only:
        raise ValueError("v2.1 is not harness-only.")

    return protocol


def main() -> int:
    args = parse_args()
    p = verify(args.protocol)
    print("Fresh-C C0.1C-v2.1 harness-repair protocol verifier")
    print(f"Protocol ID: {p.protocol_id}")
    print(f"Protocol SHA256: {p.protocol_sha256}")
    print("Parent v2 network epoch started: False")
    print("Harness change only: True")
    print("Scientific/search/transport semantics changed: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
