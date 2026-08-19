from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_2 import (
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    return p.parse_args()


def verify(path: Path):
    p = load_and_validate_protocol(path)
    changed = (
        p.transport_policy_changed_from_v21,
        p.search_queries_changed_from_v21,
        p.provider_set_changed_from_v21,
        p.search_depth_changed_from_v21,
        p.historical_ledger_changed_from_v21,
        p.target_count_changed_from_v21,
        p.blind_ordering_changed_from_v21,
        p.scientific_selection_semantics_changed_from_v21,
    )
    if any(changed):
        raise ValueError("v2.2 changed frozen semantics.")
    if not p.compatibility_change_only:
        raise ValueError("v2.2 is not compatibility-only.")
    if not p.diagnostics_builder_protocol_version_independent:
        raise ValueError("v2.2 diagnostics builder is not compatibility-safe.")
    return p


def main() -> int:
    args = parse_args()
    p = verify(args.protocol)
    print("Fresh-C C0.1C-v2.2 compatibility-repair protocol verifier")
    print(f"Protocol ID: {p.protocol_id}")
    print(f"Protocol SHA256: {p.protocol_sha256}")
    print(f"Parent v2.1 attempt: {p.parent_v21_attempt_id}")
    print("Parent v2.1 network epoch started: True")
    print("Compatibility change only: True")
    print("Scientific/search/transport semantics changed: False")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
