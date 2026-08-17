from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import dac_her.fresh_c_acquisition as fresh_c
from dac_her.fresh_c_acquisition import (
    FRESH_C_BLIND_ORDER_NAMESPACE,
    load_and_validate_protocol,
)


DEFAULT_PROTOCOL = Path(
    "dac_her/sers_fresh_c_acquisition_protocol_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the Fresh-C C0.1A preregistration protocol. "
            "This performs no discovery, selection, download, semantic "
            "read, network call, LLM call, or Reserve-C consumption."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    return parser.parse_args()


def verify(protocol_path: Path) -> dict[str, object]:
    protocol = load_and_validate_protocol(protocol_path)

    for raw in protocol.reuse_policy.reused_components:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(path)

    module_source = inspect.getsource(fresh_c)
    banned_import_fragments = (
        "corpus_acquisition.candidate_selection",
        "from requests",
        "import requests",
        "from urllib.request",
        "urlopen(",
        "import httpx",
        "import aiohttp",
    )
    for fragment in banned_import_fragments:
        if fragment in module_source:
            raise ValueError(
                "Fresh-C identity module contains forbidden dependency: "
                f"{fragment}"
            )

    rank_source = inspect.getsource(
        fresh_c.rank_fresh_identities
    )
    forbidden_rank_tokens = (
        ".title",
        ".abstract",
        "citation_count",
        "matched_axes",
        "hypothesis",
        "novelty",
        "direction",
        "open_access",
    )
    for token in forbidden_rank_tokens:
        if token in rank_source:
            raise ValueError(
                "Blind ranker source contains scientific/access token: "
                f"{token}"
            )

    if (
        protocol.blind_ordering_policy.namespace
        != FRESH_C_BLIND_ORDER_NAMESPACE
    ):
        raise ValueError("Blind-order namespace mismatch.")

    safety = protocol.safety
    if any(
        (
            safety.fresh_c_stage_activated,
            safety.activation_preconditions_satisfied_at_preregistration,
            safety.live_discovery_started,
            safety.live_selection_started,
            safety.live_acquisition_started,
            safety.content_sealed,
            safety.fresh_reserve_c_consumed,
            safety.semantic_read_performed,
            safety.automatic_next_stage_authorized,
        )
    ):
        raise ValueError(
            "C0.1A preregistration safety boundary was activated."
        )
    if safety.network_calls != 0 or safety.llm_calls != 0:
        raise ValueError(
            "C0.1A preregistration must record zero network/LLM calls."
        )
    if not safety.stop_after_preregistration_freeze:
        raise ValueError("C0.1A preregistration STOP guard drifted.")

    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "stage": protocol.stage,
        "status": protocol.status,
        "fresh_c_stage_activated": False,
        "live_discovery_started": False,
        "live_selection_started": False,
        "live_acquisition_started": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "network_calls": 0,
        "llm_calls": 0,
        "automatic_next_stage_authorized": False,
        "stop": True,
    }


def main() -> int:
    args = parse_args()
    result = verify(args.protocol)
    print("Fresh-C C0.1A protocol preregistration verifier")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
