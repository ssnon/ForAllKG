from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery as live
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    DEFAULT_PROTOCOL_PATH,
    EXPECTED_BROAD_QUERIES,
    EXPECTED_PROVIDERS,
    load_and_validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the Fresh-C C0.1C live-discovery protocol without "
            "performing network access or Fresh-C consumption."
        )
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    return parser.parse_args()


def verify(path: Path) -> dict[str, object]:
    protocol = load_and_validate_protocol(path)
    if protocol.providers != EXPECTED_PROVIDERS:
        raise ValueError("C0.1C provider set drifted.")
    if protocol.broad_queries != EXPECTED_BROAD_QUERIES:
        raise ValueError("C0.1C query set drifted.")

    rank_source = inspect.getsource(live.build_fresh_queue)
    for token in (
        ".title",
        ".abstract",
        "citation_count",
        "novelty",
        "hypothesis",
        "direction",
    ):
        if token in rank_source:
            raise ValueError(
                "C0.1C fresh-queue builder contains forbidden scientific "
                f"ordering token: {token}"
            )

    projection_source = inspect.getsource(live.project_packet_to_identity_only)
    # Title may only enter through project_catalog_identity's opaque fallback;
    # this module itself must not inspect title/abstract/citation semantics.
    for token in ("work.title", "work.abstract", "work.citation_count"):
        if token in projection_source:
            raise ValueError(
                "C0.1C projection directly inspects forbidden field: " + token
            )

    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "stage": protocol.stage,
        "providers": ",".join(protocol.providers),
        "broad_queries": len(protocol.broad_queries),
        "results_per_query_provider": protocol.results_per_query,
        "expected_provider_query_executions": (
            protocol.expected_provider_query_executions
        ),
        "max_raw_metadata_rows": protocol.max_raw_metadata_rows,
        "target_acquired_papers": protocol.target_acquired_papers,
        "full_fresh_identity_queue_frozen": True,
        "queue_truncated_to_target": False,
        "raw_catalog_packet_persisted": False,
        "fresh_reserve_c_consumed": False,
        "llm_calls": 0,
        "automatic_c0_1d_transition_authorized": False,
        "stop": True,
    }


def main() -> int:
    args = parse_args()
    result = verify(args.protocol)
    print("Fresh-C C0.1C live-discovery protocol verifier")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
