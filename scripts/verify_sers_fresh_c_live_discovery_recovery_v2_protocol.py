from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.fresh_c_live_discovery_recovery_v2 import (
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_V1_FAILED_ATTEMPT_ID,
    EXPECTED_V1_FREEZE_COMMIT,
    load_and_validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify C0.1C-v2 transport-recovery protocol. "
            "No network, LLM, semantic read, or Fresh-C consumption."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
    )
    return parser.parse_args()


def verify(path: Path) -> dict[str, object]:
    protocol = load_and_validate_protocol(path)
    if protocol.recovery_parent_attempt_id != EXPECTED_V1_FAILED_ATTEMPT_ID:
        raise ValueError("Recovery parent attempt drifted.")
    if protocol.recovery_parent_freeze_commit != EXPECTED_V1_FREEZE_COMMIT:
        raise ValueError("Recovery parent freeze commit drifted.")
    if any(
        (
            protocol.search_queries_changed_from_v1,
            protocol.provider_set_changed_from_v1,
            protocol.search_depth_changed_from_v1,
            protocol.historical_ledger_changed_from_v1,
            protocol.target_count_changed_from_v1,
            protocol.blind_ordering_changed_from_v1,
            protocol.scientific_selection_semantics_changed_from_v1,
        )
    ):
        raise ValueError("Recovery-v2 scientific/search semantics changed.")

    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "stage": protocol.stage,
        "recovery_parent_attempt_id": protocol.recovery_parent_attempt_id,
        "providers": ",".join(protocol.providers),
        "broad_queries": len(protocol.broad_queries),
        "results_per_query_provider": protocol.results_per_query,
        "historical_identity_count": protocol.historical_identity_count,
        "target_acquired_papers": protocol.target_acquired_papers,
        "blind_order_namespace": protocol.blind_order_namespace,
        "semantic_scholar_min_interval_seconds": (
            protocol.transport_policy.semantic_scholar_minimum_interval_seconds
        ),
        "semantic_scholar_max_attempts": (
            protocol.transport_policy.semantic_scholar_max_attempts
        ),
        "scientific_selection_semantics_changed": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "network_calls_during_verification": 0,
        "llm_calls": 0,
        "automatic_c0_1d_transition_authorized": False,
        "stop": True,
    }


def main() -> int:
    args = parse_args()
    result = verify(args.protocol)
    print("Fresh-C C0.1C-v2 recovery protocol verifier")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
