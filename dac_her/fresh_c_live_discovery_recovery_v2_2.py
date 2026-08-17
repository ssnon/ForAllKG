from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dac_her.fresh_c_acquisition import sha256_json
from dac_her.fresh_c_live_discovery import (
    EXPECTED_BROAD_QUERIES,
    EXPECTED_HISTORICAL_IDENTITY_COUNT,
    EXPECTED_PROVIDERS,
    TARGET_ACQUIRED_PAPERS,
)
from dac_her.fresh_c_live_discovery_recovery_v2 import (
    TransportAttemptDiagnostic,
)
from dac_her.literature_catalog_contracts import CatalogQueryExecution

V22_SEMANTICS_ID = "sers_fresh_c_live_discovery_recovery_compat_v2_2"
V22_PROTOCOL_PREFIX = (
    "sers_fresh_c_live_discovery_recovery_compat_protocol_v2_2"
)

EXPECTED_V21_PROTOCOL_ID = (
    "sers_fresh_c_live_discovery_recovery_harness_protocol_v2_1:"
    "b1961d5be8b475afd730"
)
EXPECTED_V21_PROTOCOL_SHA256 = (
    "f560eaf0c7ef5f877dd312646d09cacce71390191194d61a7eea636e5d1f1af9"
)
EXPECTED_V21_FREEZE_ID = (
    "sers_fresh_c_live_discovery_recovery_v2_1_freeze_v1:"
    "086de32c435669ff582a"
)
EXPECTED_V21_FREEZE_MANIFEST_SHA256 = (
    "0c9a259d16385c05ab04fb5bb4f5d24c1fe98cb7adc6fbc3fcee348b9686a6f1"
)
EXPECTED_V21_SOURCE_COMMIT = (
    "ab801359eb5255c3be6fc6a1cef12a4e757ef8dd"
)
EXPECTED_V21_FREEZE_COMMIT = (
    "3ef4c5f2e2f37a92cedb2171ce07b912515c26f1"
)
EXPECTED_V21_FAILED_ATTEMPT_ID = (
    "sers_fresh_c_live_discovery_recovery_attempt_v2_1:12b9dbc6d57618c9ed15"
)

V21_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_1_recovery_run_v1"
)
V21_STARTED_PATH = V21_RUN_DIR / "DISCOVERY_RECOVERY_STARTED.json"
V21_FAILED_PATH = V21_RUN_DIR / "DISCOVERY_RECOVERY_FAILED.json"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_2_protocol.json"
)
DEFAULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_2_compat_repair_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_2_recovery_run_v1"
)

FORBIDDEN_V21_SUCCESS_ARTIFACTS = (
    "run_manifest.json",
    "blind_selection_queue.json",
    "access_locator_manifest.json",
    "DISCOVERY_RECOVERY_COMPLETE.json",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompatibilityRepairProtocol(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-live-discovery-recovery-compat-protocol-v2-2"
    ]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal[
        "sers_fresh_c_live_discovery_recovery_compat_v2_2"
    ]
    stage: Literal["C0.1C-v2.2"]

    parent_v21_protocol_id: Literal[
        "sers_fresh_c_live_discovery_recovery_harness_protocol_v2_1:"
        "b1961d5be8b475afd730"
    ]
    parent_v21_protocol_sha256: Literal[
        "f560eaf0c7ef5f877dd312646d09cacce71390191194d61a7eea636e5d1f1af9"
    ]
    parent_v21_freeze_id: Literal[
        "sers_fresh_c_live_discovery_recovery_v2_1_freeze_v1:"
        "086de32c435669ff582a"
    ]
    parent_v21_freeze_manifest_sha256: Literal[
        "0c9a259d16385c05ab04fb5bb4f5d24c1fe98cb7adc6fbc3fcee348b9686a6f1"
    ]
    parent_v21_source_commit: Literal[
        "ab801359eb5255c3be6fc6a1cef12a4e757ef8dd"
    ]
    parent_v21_freeze_commit: Literal[
        "3ef4c5f2e2f37a92cedb2171ce07b912515c26f1"
    ]
    parent_v21_attempt_id: Literal[
        "sers_fresh_c_live_discovery_recovery_attempt_v2_1:12b9dbc6d57618c9ed15"
    ]
    parent_v21_network_epoch_started: Literal[True]
    parent_v21_failure_kind: Literal[
        "post_retrieval_diagnostics_protocol_shape_mismatch"
    ]
    parent_v21_same_epoch_rerun_allowed: Literal[False]
    parent_v21_success_artifacts_absent_required: Literal[True]
    parent_v21_failed_epoch_preserved: Literal[True]

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

    transport_policy_changed_from_v21: Literal[False]
    search_queries_changed_from_v21: Literal[False]
    provider_set_changed_from_v21: Literal[False]
    search_depth_changed_from_v21: Literal[False]
    historical_ledger_changed_from_v21: Literal[False]
    target_count_changed_from_v21: Literal[False]
    blind_ordering_changed_from_v21: Literal[False]
    scientific_selection_semantics_changed_from_v21: Literal[False]

    compatibility_change_only: Literal[True]
    diagnostics_builder_protocol_version_independent: Literal[True]
    explicit_live_confirmation_required: Literal[True]
    recovery_started_marker_before_first_network_call: Literal[True]
    same_epoch_rerun_after_start_allowed: Literal[False]
    failure_authorizes_query_or_selection_tuning: Literal[False]
    fresh_reserve_c_consumption_occurs_here: Literal[False]
    semantic_read_allowed: Literal[False]
    automatic_c0_1d_transition_allowed: Literal[False]
    stop_after_success: Literal[True]
    llm_calls: Literal[0]

    @model_validator(mode="after")
    def _exact_contract(self) -> "CompatibilityRepairProtocol":
        if self.providers != EXPECTED_PROVIDERS:
            raise ValueError("v2.2 provider set drifted.")
        if self.broad_queries != EXPECTED_BROAD_QUERIES:
            raise ValueError("v2.2 query set drifted.")
        if self.historical_identity_count != EXPECTED_HISTORICAL_IDENTITY_COUNT:
            raise ValueError("v2.2 historical identity count drifted.")
        if self.target_acquired_papers != TARGET_ACQUIRED_PAPERS:
            raise ValueError("v2.2 target count drifted.")
        return self


class ProviderExecutionDiagnostic(StrictModel):
    provider: str
    query_id: str
    success: bool
    result_count: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    error_type: str | None = None
    error_summary: str | None = None
    http_status: int | None = None


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
    return V22_PROTOCOL_PREFIX + ":" + _protocol_identity_sha(payload)[:20]


def load_and_validate_protocol(path: Path) -> CompatibilityRepairProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v2.2 protocol must be a JSON object.")
    protocol = CompatibilityRepairProtocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("v2.2 protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("v2.2 protocol SHA mismatch.")
    return protocol


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def validate_v21_failed_epoch(root: Path) -> dict[str, Any]:
    started_path = root / V21_STARTED_PATH
    failed_path = root / V21_FAILED_PATH
    if not started_path.exists() or not failed_path.exists():
        raise FileNotFoundError(
            "v2.1 STARTED/FAILED markers are required for v2.2."
        )

    started = _read_json(started_path)
    failed = _read_json(failed_path)
    if started.get("attempt_id") != EXPECTED_V21_FAILED_ATTEMPT_ID:
        raise ValueError("v2.1 STARTED attempt ID drifted.")
    if failed.get("attempt_id") != EXPECTED_V21_FAILED_ATTEMPT_ID:
        raise ValueError("v2.1 FAILED attempt ID drifted.")
    if started.get("network_boundary_opened") is not True:
        raise ValueError("v2.1 network boundary was not recorded open.")
    if failed.get("network_boundary_opened") is not True:
        raise ValueError("v2.1 FAILED network boundary flag drifted.")
    if started.get("same_recovery_epoch_rerun_allowed") is not False:
        raise ValueError("v2.1 STARTED rerun guard drifted.")
    if failed.get("same_recovery_epoch_rerun_allowed") is not False:
        raise ValueError("v2.1 FAILED rerun guard drifted.")
    if failed.get("new_protocol_epoch_required") is not True:
        raise ValueError("v2.1 FAILED new-epoch requirement drifted.")

    for row in (started, failed):
        if row.get("fresh_reserve_c_consumed") is not False:
            raise ValueError("v2.1 unexpectedly consumed Fresh C.")
        if row.get("semantic_read_performed") is not False:
            raise ValueError("v2.1 unexpectedly performed semantic read.")

    for name in FORBIDDEN_V21_SUCCESS_ARTIFACTS:
        if (root / V21_RUN_DIR / name).exists():
            raise RuntimeError(
                f"v2.1 success-stage artifact unexpectedly exists: {name}"
            )

    return {"started": started, "failed": failed}


def _sanitize_error_summary(
    text: str | None,
    *,
    forbidden_queries: Sequence[str],
) -> str | None:
    if text is None:
        return None
    value = " ".join(str(text).split())
    for query in forbidden_queries:
        if query:
            value = value.replace(query, "<query-redacted>")
    return value[:400] or None


_HTTP_ERROR_RE = re.compile(r"(?:HTTP status|HTTP Error)\s+(\d{3})")


def _execution_http_status(error: str | None) -> int | None:
    if not error:
        return None
    match = _HTTP_ERROR_RE.search(error)
    return int(match.group(1)) if match else None


def make_transport_diagnostics_payload_v2_2(
    *,
    protocol_id: str,
    parent_attempt_id: str,
    broad_queries: Sequence[str],
    executions: Sequence[CatalogQueryExecution],
    semantic_scholar_attempts: Sequence[TransportAttemptDiagnostic],
) -> dict[str, Any]:
    rows: list[ProviderExecutionDiagnostic] = []
    for execution in executions:
        error_type = None
        if execution.error:
            error_type = str(execution.error).split(":", 1)[0].strip()
        rows.append(
            ProviderExecutionDiagnostic(
                provider=execution.provider,
                query_id=execution.query_id,
                success=execution.success,
                result_count=execution.result_count,
                elapsed_seconds=execution.elapsed_seconds,
                error_type=error_type or None,
                error_summary=_sanitize_error_summary(
                    execution.error,
                    forbidden_queries=broad_queries,
                ),
                http_status=_execution_http_status(execution.error),
            )
        )

    payload: dict[str, Any] = {
        "schema_version": (
            "sers-fresh-c-live-discovery-transport-diagnostics-v2-2"
        ),
        "protocol_id": protocol_id,
        "recovery_parent_attempt_id": parent_attempt_id,
        "credential_presence": {
            "semantic_scholar_api_key_present": bool(
                os.getenv("SEMANTIC_SCHOLAR_API_KEY")
            ),
            "crossref_mailto_present": bool(
                os.getenv("CROSSREF_MAILTO")
            ),
        },
        "provider_executions": [
            row.model_dump(mode="json") for row in rows
        ],
        "semantic_scholar_transport_attempts": [
            row.model_dump(mode="json")
            for row in semantic_scholar_attempts
        ],
        "scientific_response_body_persisted": False,
        "query_text_persisted": False,
        "title_persisted": False,
        "abstract_persisted": False,
        "citation_count_persisted": False,
        "credential_values_persisted": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "llm_calls": 0,
    }
    payload["diagnostics_sha256"] = _payload_sha(
        payload,
        "diagnostics_sha256",
    )
    return payload
