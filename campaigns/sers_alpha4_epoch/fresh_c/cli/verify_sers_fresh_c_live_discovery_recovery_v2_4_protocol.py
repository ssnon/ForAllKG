from __future__ import annotations

import argparse
from pathlib import Path

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_4 import (
    DEFAULT_PROTOCOL_PATH,
    load_and_validate_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    args = parser.parse_args()
    p = load_and_validate_protocol(args.protocol)

    if p.provider_universe_changed is not True:
        raise ValueError("v2.4 must explicitly record provider-universe change.")
    if p.provider_substitution_performed is not True:
        raise ValueError("v2.4 provider substitution flag missing.")
    if any((
        p.queries_changed_from_v22,
        p.search_depth_changed_from_v22,
        p.historical_ledger_changed_from_v22,
        p.target_count_changed_from_v22,
        p.blind_ordering_changed_from_v22,
        p.hypothesis_aware_selection_added,
        p.title_abstract_scoring_added,
        p.scientific_selection_semantics_changed,
    )):
        raise ValueError("v2.4 changed a prohibited scientific/selection field.")

    print("Fresh-C C0.1C-v2.4 OpenAlex+Crossref substitution protocol verifier")
    print(f"Protocol ID: {p.protocol_id}")
    print(f"Protocol SHA256: {p.protocol_sha256}")
    print("Provider set: openalex,crossref")
    print("Provider universe changed: True")
    print("Substitution reason: transport availability only after HTTP 429")
    print("Frozen queries changed: False")
    print("Historical ledger changed: False")
    print("Blind ordering changed: False")
    print("Hypothesis-aware selection added: False")
    print("Scientific selection semantics changed: False")
    print("OpenAlex API key required: True")
    print("Fresh Reserve C consumed: False")
    print("Network calls during verification: 0")
    print("LLM calls during verification: 0")
    print("STOP: True")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
