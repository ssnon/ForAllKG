from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    EXPECTED_BROAD_QUERIES,
    EXPECTED_HISTORICAL_IDENTITY_COUNT,
    EXPECTED_PROVIDERS,
    TARGET_ACQUIRED_PAPERS,
)

V21_SEMANTICS_ID = "sers_fresh_c_live_discovery_recovery_harness_v2_1"
V21_PROTOCOL_PREFIX = (
    "sers_fresh_c_live_discovery_recovery_harness_protocol_v2_1"
)

EXPECTED_V2_PROTOCOL_ID = (
    "sers_fresh_c_live_discovery_recovery_protocol_v2:"
    "299b9616ba279e0e3519"
)
EXPECTED_V2_PROTOCOL_SHA256 = (
    "e3e25c816005d6bd5d2b2a5ea3e3a619a645c84c52270a4da9c0a306dfdbe5dd"
)
EXPECTED_V2_FREEZE_ID = (
    "sers_fresh_c_live_discovery_recovery_v2_protocol_freeze_v1:"
    "f0d39fa65a19718594c0"
)
EXPECTED_V2_FREEZE_MANIFEST_SHA256 = (
    "f1c24db2eb24704e3fdda6e5b148b7eccce8f16d923a5c803f6290055cc37e01"
)
EXPECTED_V2_SOURCE_COMMIT = (
    "9978d331aaa239883eaf235c61cf78e48c07e2f4"
)
EXPECTED_V2_FREEZE_COMMIT = (
    "4edb50343a733857a10c3d599a88c878f1e04958"
)
EXPECTED_V1_FAILED_ATTEMPT_ID = (
    "sers_fresh_c_live_discovery_attempt_v1:0912aca95ffe39b9f8a3"
)

V2_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_recovery_run_v1"
)
DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_1_protocol.json"
)
DEFAULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/"
    "c0_1c_v2_1_harness_repair_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_1_recovery_run_v1"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HarnessRepairProtocol(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-live-discovery-recovery-harness-protocol-v2-1"
    ]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal[
        "sers_fresh_c_live_discovery_recovery_harness_v2_1"
    ]
    stage: Literal["C0.1C-v2.1"]

    parent_v2_protocol_id: Literal[
        "sers_fresh_c_live_discovery_recovery_protocol_v2:"
        "299b9616ba279e0e3519"
    ]
    parent_v2_protocol_sha256: Literal[
        "e3e25c816005d6bd5d2b2a5ea3e3a619a645c84c52270a4da9c0a306dfdbe5dd"
    ]
    parent_v2_freeze_id: Literal[
        "sers_fresh_c_live_discovery_recovery_v2_protocol_freeze_v1:"
        "f0d39fa65a19718594c0"
    ]
    parent_v2_freeze_manifest_sha256: Literal[
        "f1c24db2eb24704e3fdda6e5b148b7eccce8f16d923a5c803f6290055cc37e01"
    ]
    parent_v2_source_commit: Literal[
        "9978d331aaa239883eaf235c61cf78e48c07e2f4"
    ]
    parent_v2_freeze_commit: Literal[
        "4edb50343a733857a10c3d599a88c878f1e04958"
    ]
    parent_v2_network_epoch_started: Literal[False]
    parent_v2_failure_kind: Literal[
        "pre_network_argparse_harness_mismatch"
    ]
    parent_v2_frozen_artifacts_preserved: Literal[True]

    providers: list[str]
    broad_queries: list[str]
    results_per_query: Literal[100]
    expected_provider_query_executions: Literal[8]
    max_raw_metadata_rows: Literal[800]
    historical_identity_count: Literal[560]
    target_acquired_papers: Literal[25]
    blind_order_namespace: Literal[
        "sers_fresh_c_blind_identity_order_v1"
    ]

    transport_policy_changed_from_v2: Literal[False]
    search_queries_changed_from_v2: Literal[False]
    provider_set_changed_from_v2: Literal[False]
    search_depth_changed_from_v2: Literal[False]
    historical_ledger_changed_from_v2: Literal[False]
    target_count_changed_from_v2: Literal[False]
    blind_ordering_changed_from_v2: Literal[False]
    scientific_selection_semantics_changed_from_v2: Literal[False]

    harness_change_only: Literal[True]
    preflight_flag_required: Literal[True]
    explicit_live_confirmation_required: Literal[True]
    recovery_started_marker_before_first_network_call: Literal[True]
    same_epoch_rerun_after_start_allowed: Literal[False]
    fresh_reserve_c_consumption_occurs_here: Literal[False]
    semantic_read_allowed: Literal[False]
    automatic_c0_1d_transition_allowed: Literal[False]
    stop_after_success: Literal[True]
    llm_calls: Literal[0]

    @model_validator(mode="after")
    def _exact_contract(self) -> "HarnessRepairProtocol":
        if self.providers != EXPECTED_PROVIDERS:
            raise ValueError("v2.1 provider set drifted.")
        if self.broad_queries != EXPECTED_BROAD_QUERIES:
            raise ValueError("v2.1 query set drifted.")
        if self.historical_identity_count != EXPECTED_HISTORICAL_IDENTITY_COUNT:
            raise ValueError("v2.1 historical identity count drifted.")
        if self.target_acquired_papers != TARGET_ACQUIRED_PAPERS:
            raise ValueError("v2.1 target count drifted.")
        return self


def _payload_sha(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return sha256_json(value)


def _protocol_identity_sha(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("protocol_id", None)
    value.pop("protocol_sha256", None)
    return sha256_json(value)


def expected_protocol_id(payload: Mapping[str, Any]) -> str:
    return V21_PROTOCOL_PREFIX + ":" + _protocol_identity_sha(payload)[:20]


def load_and_validate_protocol(path: Path) -> HarnessRepairProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v2.1 harness protocol must be a JSON object.")
    protocol = HarnessRepairProtocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("v2.1 protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("v2.1 protocol SHA mismatch.")
    return protocol


def assert_parent_v2_never_started_network(root: Path) -> None:
    run_dir = root / V2_RUN_DIR
    if not run_dir.exists():
        return
    files = [p for p in run_dir.iterdir() if p.is_file()]
    if files:
        raise RuntimeError(
            "Parent v2 recovery run directory contains artifacts; "
            "cannot classify v2 as pre-network harness-only failure."
        )
