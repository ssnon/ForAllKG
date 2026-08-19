from __future__ import annotations

import json
import os
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

V23_SEMANTICS_ID = "sers_fresh_c_authenticated_transport_recovery_v2_3"
V23_PROTOCOL_PREFIX = (
    "sers_fresh_c_authenticated_transport_recovery_protocol_v2_3"
)

EXPECTED_V22_PROTOCOL_ID = (
    "sers_fresh_c_live_discovery_recovery_compat_protocol_v2_2:"
    "374c0c4a6eada05717b7"
)
EXPECTED_V22_PROTOCOL_SHA256 = (
    "ea944cc62da38138d61902273a4293bfc21bf25d05eb1adb57cb1b0fad2cd0b6"
)
EXPECTED_V22_FREEZE_ID = (
    "sers_fresh_c_live_discovery_recovery_v2_2_freeze_v1:"
    "b08723445dadf2fa329e"
)
EXPECTED_V22_FREEZE_MANIFEST_SHA256 = (
    "45bd8c9a5347869611905af377ab6390d66aefcf8dfc44475f57d8875f07e498"
)
EXPECTED_V22_SOURCE_COMMIT = (
    "59d9a320f7840d6de5d45172c94957ed479fe583"
)
EXPECTED_V22_FREEZE_COMMIT = (
    "54a09740deb099dd3f68a551aba3e617d9088a67"
)
EXPECTED_V22_FAILED_ATTEMPT_ID = (
    "sers_fresh_c_live_discovery_recovery_attempt_v2_2:ce325ae6c64aac05be94"
)

V22_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_2_recovery_run_v1"
)
V22_STARTED_PATH = V22_RUN_DIR / "DISCOVERY_RECOVERY_STARTED.json"
V22_FAILED_PATH = V22_RUN_DIR / "DISCOVERY_RECOVERY_FAILED.json"
V22_DIAGNOSTICS_PATH = V22_RUN_DIR / "TRANSPORT_DIAGNOSTICS.json"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_3_protocol.json"
)
DEFAULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_3_authenticated_recovery_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_3_recovery_run_v1"
)

FORBIDDEN_V22_SUCCESS_ARTIFACTS = (
    "run_manifest.json",
    "blind_selection_queue.json",
    "access_locator_manifest.json",
    "DISCOVERY_RECOVERY_COMPLETE.json",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthenticatedRecoveryProtocol(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-authenticated-transport-recovery-protocol-v2-3"
    ]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal[
        "sers_fresh_c_authenticated_transport_recovery_v2_3"
    ]
    stage: Literal["C0.1C-v2.3"]

    parent_v22_protocol_id: Literal[
        "sers_fresh_c_live_discovery_recovery_compat_protocol_v2_2:"
        "374c0c4a6eada05717b7"
    ]
    parent_v22_protocol_sha256: Literal[
        "ea944cc62da38138d61902273a4293bfc21bf25d05eb1adb57cb1b0fad2cd0b6"
    ]
    parent_v22_freeze_id: Literal[
        "sers_fresh_c_live_discovery_recovery_v2_2_freeze_v1:"
        "b08723445dadf2fa329e"
    ]
    parent_v22_freeze_manifest_sha256: Literal[
        "45bd8c9a5347869611905af377ab6390d66aefcf8dfc44475f57d8875f07e498"
    ]
    parent_v22_source_commit: Literal[
        "59d9a320f7840d6de5d45172c94957ed479fe583"
    ]
    parent_v22_freeze_commit: Literal[
        "54a09740deb099dd3f68a551aba3e617d9088a67"
    ]
    parent_v22_attempt_id: Literal[
        "sers_fresh_c_live_discovery_recovery_attempt_v2_2:ce325ae6c64aac05be94"
    ]
    parent_v22_network_epoch_started: Literal[True]
    parent_v22_failure_kind: Literal[
        "semantic_scholar_http_429_unauthenticated"
    ]
    parent_v22_semantic_scholar_http_status: Literal[429]
    parent_v22_crossref_success_count: Literal[4]
    parent_v22_semantic_scholar_success_count: Literal[0]
    parent_v22_api_key_present: Literal[False]
    parent_v22_same_epoch_rerun_allowed: Literal[False]
    parent_v22_failed_epoch_preserved: Literal[True]
    parent_v22_success_artifacts_absent_required: Literal[True]

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

    search_queries_changed_from_v22: Literal[False]
    provider_set_changed_from_v22: Literal[False]
    search_depth_changed_from_v22: Literal[False]
    historical_ledger_changed_from_v22: Literal[False]
    target_count_changed_from_v22: Literal[False]
    blind_ordering_changed_from_v22: Literal[False]
    scientific_selection_semantics_changed_from_v22: Literal[False]

    semantic_scholar_api_key_required: Literal[True]
    semantic_scholar_api_key_env: Literal["SEMANTIC_SCHOLAR_API_KEY"]
    credential_value_may_be_persisted: Literal[False]
    authenticated_transport_is_only_material_change: Literal[True]
    semantic_scholar_minimum_interval_seconds: Literal[1.1]
    semantic_scholar_max_attempts: Literal[4]
    retryable_http_status: list[int]
    transport_pacing_changed_from_v22: Literal[False]
    transport_retry_policy_changed_from_v22: Literal[False]

    explicit_live_confirmation_required: Literal[True]
    started_marker_before_first_network_call: Literal[True]
    same_epoch_rerun_after_start_allowed: Literal[False]
    failure_authorizes_query_or_selection_tuning: Literal[False]
    fresh_reserve_c_consumption_occurs_here: Literal[False]
    semantic_read_allowed: Literal[False]
    automatic_c0_1d_transition_allowed: Literal[False]
    stop_after_success: Literal[True]
    llm_calls: Literal[0]

    @model_validator(mode="after")
    def _exact_contract(self) -> "AuthenticatedRecoveryProtocol":
        if self.providers != EXPECTED_PROVIDERS:
            raise ValueError("v2.3 provider set drifted.")
        if self.broad_queries != EXPECTED_BROAD_QUERIES:
            raise ValueError("v2.3 query set drifted.")
        if self.historical_identity_count != EXPECTED_HISTORICAL_IDENTITY_COUNT:
            raise ValueError("v2.3 historical identity count drifted.")
        if self.target_acquired_papers != TARGET_ACQUIRED_PAPERS:
            raise ValueError("v2.3 target count drifted.")
        if self.retryable_http_status != [429, 500, 502, 503, 504]:
            raise ValueError("v2.3 retryable HTTP set drifted.")
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
    return V23_PROTOCOL_PREFIX + ":" + _protocol_identity_sha(payload)[:20]


def load_and_validate_protocol(path: Path) -> AuthenticatedRecoveryProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v2.3 protocol must be a JSON object.")
    protocol = AuthenticatedRecoveryProtocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("v2.3 protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("v2.3 protocol SHA mismatch.")
    return protocol


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def require_api_key_presence() -> None:
    if not os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        raise RuntimeError(
            "SEMANTIC_SCHOLAR_API_KEY is required for C0.1C-v2.3; "
            "credential value must not be printed or persisted."
        )


def validate_v22_failed_epoch(root: Path) -> dict[str, Any]:
    required = (
        root / V22_STARTED_PATH,
        root / V22_FAILED_PATH,
        root / V22_DIAGNOSTICS_PATH,
    )
    if any(not p.exists() for p in required):
        raise FileNotFoundError(
            "v2.2 STARTED/FAILED/TRANSPORT_DIAGNOSTICS are required."
        )

    started = _read_json(root / V22_STARTED_PATH)
    failed = _read_json(root / V22_FAILED_PATH)
    diagnostics = _read_json(root / V22_DIAGNOSTICS_PATH)

    if started.get("attempt_id") != EXPECTED_V22_FAILED_ATTEMPT_ID:
        raise ValueError("v2.2 STARTED attempt ID drifted.")
    if failed.get("attempt_id") != EXPECTED_V22_FAILED_ATTEMPT_ID:
        raise ValueError("v2.2 FAILED attempt ID drifted.")
    if started.get("network_boundary_opened") is not True:
        raise ValueError("v2.2 network boundary flag drifted.")
    if failed.get("network_boundary_opened") is not True:
        raise ValueError("v2.2 FAILED network boundary flag drifted.")
    if started.get("same_recovery_epoch_rerun_allowed") is not False:
        raise ValueError("v2.2 STARTED rerun guard drifted.")
    if failed.get("same_recovery_epoch_rerun_allowed") is not False:
        raise ValueError("v2.2 FAILED rerun guard drifted.")
    if failed.get("new_protocol_epoch_required") is not True:
        raise ValueError("v2.2 new-epoch requirement drifted.")

    for row in (started, failed, diagnostics):
        if row.get("fresh_reserve_c_consumed") is not False:
            raise ValueError("v2.2 unexpectedly consumed Fresh C.")
        if row.get("semantic_read_performed") is not False:
            raise ValueError("v2.2 unexpectedly performed semantic read.")

    creds = diagnostics.get("credential_presence") or {}
    if creds.get("semantic_scholar_api_key_present") is not False:
        raise ValueError("v2.2 diagnostics do not record unauthenticated state.")
    executions = diagnostics.get("provider_executions") or []
    s2 = [r for r in executions if r.get("provider") == "semantic_scholar"]
    cr = [r for r in executions if r.get("provider") == "crossref"]
    if len(s2) != 4 or any(r.get("http_status") != 429 for r in s2):
        raise ValueError("v2.2 did not fail as frozen 4x Semantic Scholar 429.")
    if any(r.get("success") is not False for r in s2):
        raise ValueError("v2.2 Semantic Scholar execution unexpectedly succeeded.")
    if len(cr) != 4 or any(r.get("success") is not True for r in cr):
        raise ValueError("v2.2 Crossref was not 4/4 successful.")

    for name in FORBIDDEN_V22_SUCCESS_ARTIFACTS:
        if (root / V22_RUN_DIR / name).exists():
            raise RuntimeError(
                f"v2.2 success-stage artifact unexpectedly exists: {name}"
            )

    return {
        "started": started,
        "failed": failed,
        "diagnostics": diagnostics,
    }
