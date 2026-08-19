from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import sha256_json
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    EXPECTED_BROAD_QUERIES,
    EXPECTED_HISTORICAL_IDENTITY_COUNT,
    TARGET_ACQUIRED_PAPERS,
)
from dac_her.literature_catalog import normalize_doi
from dac_her.literature_catalog_contracts import (
    CatalogQuery,
    CatalogQueryExecution,
    CatalogWork,
)

V24_SEMANTICS_ID = "sers_fresh_c_openalex_crossref_substitution_v2_4"
V24_PROTOCOL_PREFIX = (
    "sers_fresh_c_openalex_crossref_substitution_protocol_v2_4"
)

EXPECTED_V22_FAILED_ATTEMPT_ID = (
    "sers_fresh_c_live_discovery_recovery_attempt_v2_2:ce325ae6c64aac05be94"
)
EXPECTED_V23_PROTOCOL_ID = (
    "sers_fresh_c_authenticated_transport_recovery_protocol_v2_3:796d965ed7d78f4224c5"
)
EXPECTED_V23_PROTOCOL_SHA256 = (
    "4b719343c11b6c55fab1cf05d4b8fcd6310756fa1fd855230d6b2fa4056435c8"
)

V22_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_2_recovery_run_v1"
)
V22_STARTED_PATH = V22_RUN_DIR / "DISCOVERY_RECOVERY_STARTED.json"
V22_FAILED_PATH = V22_RUN_DIR / "DISCOVERY_RECOVERY_FAILED.json"
V22_DIAGNOSTICS_PATH = V22_RUN_DIR / "TRANSPORT_DIAGNOSTICS.json"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_4_protocol.json"
)
DEFAULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/"
    "c0_1c_v2_4_openalex_crossref_substitution_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_4_recovery_run_v1"
)

OPENALEX_PROVIDER_NAME = "openalex"
CROSSREF_PROVIDER_NAME = "crossref"
EXPECTED_V24_PROVIDERS = ["openalex", "crossref"]

OPENALEX_MINIMUM_INTERVAL_SECONDS = 1.10
OPENALEX_MAX_ATTEMPTS = 4
OPENALEX_BASE_BACKOFF_SECONDS = 2.0
OPENALEX_MAX_RETRY_AFTER_SECONDS = 60.0
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

FORBIDDEN_V22_SUCCESS_ARTIFACTS = (
    "run_manifest.json",
    "blind_selection_queue.json",
    "access_locator_manifest.json",
    "DISCOVERY_RECOVERY_COMPLETE.json",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderSubstitutionProtocol(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-openalex-crossref-substitution-protocol-v2-4"
    ]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal[
        "sers_fresh_c_openalex_crossref_substitution_v2_4"
    ]
    stage: Literal["C0.1C-v2.4"]

    v22_failed_attempt_id: Literal[
        "sers_fresh_c_live_discovery_recovery_attempt_v2_2:ce325ae6c64aac05be94"
    ]
    v22_network_epoch_started: Literal[True]
    v22_failure_kind: Literal[
        "semantic_scholar_http_429_unauthenticated"
    ]
    v22_semantic_scholar_http_status: Literal[429]
    v22_semantic_scholar_success_count: Literal[0]
    v22_crossref_success_count: Literal[4]
    v22_failed_epoch_preserved: Literal[True]

    v23_protocol_id: Literal[
        "sers_fresh_c_authenticated_transport_recovery_protocol_v2_3:796d965ed7d78f4224c5"
    ]
    v23_protocol_sha256: Literal[
        "4b719343c11b6c55fab1cf05d4b8fcd6310756fa1fd855230d6b2fa4056435c8"
    ]
    v23_frozen_but_unexecuted_required: Literal[True]
    v23_nonexecution_reason: Literal[
        "semantic_scholar_credential_not_used_by_operator"
    ]

    old_providers: list[str]
    providers: list[str]
    provider_substitution_performed: Literal[True]
    provider_substitution_from: Literal["semantic_scholar"]
    provider_substitution_to: Literal["openalex"]
    provider_substitution_reason: Literal[
        "transport_availability_only_after_repeated_http_429"
    ]
    provider_universe_changed: Literal[True]

    broad_queries: list[str]
    results_per_query: Literal[100]
    expected_provider_query_executions: Literal[8]
    max_raw_metadata_rows: Literal[800]
    historical_identity_count: Literal[560]
    target_acquired_papers: Literal[25]
    blind_order_namespace: Literal[
        "sers_fresh_c_blind_identity_order_v1"
    ]

    queries_changed_from_v22: Literal[False]
    search_depth_changed_from_v22: Literal[False]
    historical_ledger_changed_from_v22: Literal[False]
    target_count_changed_from_v22: Literal[False]
    blind_ordering_changed_from_v22: Literal[False]
    hypothesis_aware_selection_added: Literal[False]
    title_abstract_scoring_added: Literal[False]
    scientific_selection_semantics_changed: Literal[False]

    openalex_api_key_required: Literal[True]
    openalex_api_key_env: Literal["OPENALEX_API_KEY"]
    openalex_api_key_value_persisted: Literal[False]
    openalex_search_parameter: Literal["search"]
    openalex_results_parameter: Literal["per_page"]
    openalex_minimum_interval_seconds: Literal[1.1]
    openalex_max_attempts: Literal[4]
    retryable_http_status: list[int]
    raw_provider_packet_persisted: Literal[False]

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
    def _exact_contract(self) -> "ProviderSubstitutionProtocol":
        if self.old_providers != ["semantic_scholar", "crossref"]:
            raise ValueError("v2.4 old provider set drifted.")
        if self.providers != EXPECTED_V24_PROVIDERS:
            raise ValueError("v2.4 provider set drifted.")
        if self.broad_queries != EXPECTED_BROAD_QUERIES:
            raise ValueError("v2.4 frozen query set drifted.")
        if self.historical_identity_count != EXPECTED_HISTORICAL_IDENTITY_COUNT:
            raise ValueError("v2.4 historical identity count drifted.")
        if self.target_acquired_papers != TARGET_ACQUIRED_PAPERS:
            raise ValueError("v2.4 target count drifted.")
        if self.retryable_http_status != sorted(RETRYABLE_HTTP_STATUS):
            raise ValueError("v2.4 retryable HTTP set drifted.")
        return self


class OpenAlexTransportAttemptDiagnostic(StrictModel):
    provider: Literal["openalex"] = "openalex"
    query_id: str
    attempt: int = Field(ge=1)
    outcome: Literal["success", "retry", "failure"]
    http_status: int | None = None
    exception_type: str | None = None
    error_class: str | None = None
    elapsed_seconds: float = Field(ge=0.0)


class OpenAlexProviderHTTPError(RuntimeError):
    def __init__(self, status: int) -> None:
        self.status = int(status)
        super().__init__(f"HTTP status {self.status}")


class OpenAlexProviderNetworkError(RuntimeError):
    pass


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
    return V24_PROTOCOL_PREFIX + ":" + _protocol_identity_sha(payload)[:20]


def load_and_validate_protocol(path: Path) -> ProviderSubstitutionProtocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v2.4 protocol must be a JSON object.")
    protocol = ProviderSubstitutionProtocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("v2.4 protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("v2.4 protocol SHA mismatch.")
    return protocol


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_openalex_api_key() -> None:
    if not os.getenv("OPENALEX_API_KEY"):
        raise RuntimeError(
            "OPENALEX_API_KEY is required for C0.1C-v2.4; "
            "the credential value must not be printed or persisted."
        )


def validate_v22_failure(root: Path) -> dict[str, Any]:
    required = (
        root / V22_STARTED_PATH,
        root / V22_FAILED_PATH,
        root / V22_DIAGNOSTICS_PATH,
    )
    if any(not path.exists() for path in required):
        raise FileNotFoundError(
            "v2.2 STARTED/FAILED/TRANSPORT_DIAGNOSTICS are required."
        )

    started = _read_json(root / V22_STARTED_PATH)
    failed = _read_json(root / V22_FAILED_PATH)
    diag = _read_json(root / V22_DIAGNOSTICS_PATH)

    if started.get("attempt_id") != EXPECTED_V22_FAILED_ATTEMPT_ID:
        raise ValueError("v2.2 STARTED attempt ID drifted.")
    if failed.get("attempt_id") != EXPECTED_V22_FAILED_ATTEMPT_ID:
        raise ValueError("v2.2 FAILED attempt ID drifted.")
    if started.get("network_boundary_opened") is not True:
        raise ValueError("v2.2 network boundary was not opened.")
    if failed.get("same_recovery_epoch_rerun_allowed") is not False:
        raise ValueError("v2.2 rerun guard drifted.")
    if failed.get("new_protocol_epoch_required") is not True:
        raise ValueError("v2.2 did not require a new epoch.")

    for row in (started, failed, diag):
        if row.get("fresh_reserve_c_consumed") is not False:
            raise ValueError("v2.2 unexpectedly consumed Fresh C.")
        if row.get("semantic_read_performed") is not False:
            raise ValueError("v2.2 unexpectedly performed semantic read.")

    creds = diag.get("credential_presence") or {}
    if creds.get("semantic_scholar_api_key_present") is not False:
        raise ValueError("v2.2 was not the frozen unauthenticated epoch.")

    executions = diag.get("provider_executions") or []
    s2 = [row for row in executions if row.get("provider") == "semantic_scholar"]
    cr = [row for row in executions if row.get("provider") == "crossref"]
    if len(s2) != 4:
        raise ValueError("v2.2 Semantic Scholar execution count drifted.")
    if any(
        row.get("success") is not False or row.get("http_status") != 429
        for row in s2
    ):
        raise ValueError("v2.2 is not exact 4x Semantic Scholar HTTP 429.")
    if len(cr) != 4 or any(row.get("success") is not True for row in cr):
        raise ValueError("v2.2 Crossref is not exact 4/4 success.")

    for name in FORBIDDEN_V22_SUCCESS_ARTIFACTS:
        if (root / V22_RUN_DIR / name).exists():
            raise RuntimeError(
                f"v2.2 success-stage artifact unexpectedly exists: {name}"
            )

    return {"started": started, "failed": failed, "diagnostics": diag}


def validate_v23_frozen_unexecuted(root: Path) -> dict[str, Any]:
    from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery_recovery_v2_3 import (
        DEFAULT_FREEZE_DIR as V23_FREEZE_DIR,
        DEFAULT_PROTOCOL_PATH as V23_PROTOCOL_PATH,
        DEFAULT_RUN_DIR as V23_RUN_DIR,
        load_and_validate_protocol as load_v23_protocol,
    )

    protocol = load_v23_protocol(root / V23_PROTOCOL_PATH)
    if protocol.protocol_id != EXPECTED_V23_PROTOCOL_ID:
        raise ValueError("v2.3 protocol ID drifted.")
    if protocol.protocol_sha256 != EXPECTED_V23_PROTOCOL_SHA256:
        raise ValueError("v2.3 protocol SHA drifted.")

    freeze_manifest = root / V23_FREEZE_DIR / "freeze_manifest.json"
    ready_path = root / V23_FREEZE_DIR / "FREEZE_READY.json"
    if not freeze_manifest.exists() or not ready_path.exists():
        raise FileNotFoundError("v2.3 frozen artifacts are required.")
    manifest = _read_json(freeze_manifest)
    ready = _read_json(ready_path)
    if manifest.get("protocol_id") != EXPECTED_V23_PROTOCOL_ID:
        raise ValueError("v2.3 freeze protocol ID drifted.")
    if manifest.get("protocol_sha256") != EXPECTED_V23_PROTOCOL_SHA256:
        raise ValueError("v2.3 freeze protocol SHA drifted.")
    if manifest.get("recovery_live_discovery_started") is not False:
        raise ValueError("v2.3 freeze incorrectly records started.")
    if manifest.get("fresh_reserve_c_consumed") is not False:
        raise ValueError("v2.3 freeze consumed Fresh C.")
    if ready.get("recovery_live_discovery_authorized") is not False:
        raise ValueError("v2.3 freeze unexpectedly authorized live execution.")

    run_dir = root / V23_RUN_DIR
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            "v2.3 is not unexecuted: run directory contains artifacts."
        )

    return {"manifest": manifest, "ready": ready}


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _openalex_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.rsplit("/", 1)[-1]


class DiagnosticOpenAlexCatalogProvider:
    provider_name = "openalex"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENALEX_API_KEY",
        timeout: float = 30.0,
        minimum_interval_seconds: float = OPENALEX_MINIMUM_INTERVAL_SECONDS,
        max_attempts: int = OPENALEX_MAX_ATTEMPTS,
        base_backoff_seconds: float = OPENALEX_BASE_BACKOFF_SECONDS,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env)
        self.timeout = float(timeout)
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self.max_attempts = int(max_attempts)
        self.base_backoff_seconds = float(base_backoff_seconds)
        self.attempt_diagnostics: list[OpenAlexTransportAttemptDiagnostic] = []

    def _request_json(self, *, url: str, query_id: str) -> Any:
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": "GraphAgentsDAC-FreshC-OpenAlex-v2.4"
                    },
                )
                with urlopen(request, timeout=self.timeout) as response:
                    status = int(
                        getattr(response, "status", None)
                        or response.getcode()
                        or 200
                    )
                    payload = json.loads(response.read().decode("utf-8"))
                self.attempt_diagnostics.append(
                    OpenAlexTransportAttemptDiagnostic(
                        query_id=query_id,
                        attempt=attempt,
                        outcome="success",
                        http_status=status,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                return payload
            except HTTPError as exc:
                retryable = int(exc.code) in RETRYABLE_HTTP_STATUS
                can_retry = retryable and attempt < self.max_attempts
                self.attempt_diagnostics.append(
                    OpenAlexTransportAttemptDiagnostic(
                        query_id=query_id,
                        attempt=attempt,
                        outcome="retry" if can_retry else "failure",
                        http_status=int(exc.code),
                        exception_type=type(exc).__name__,
                        error_class=(
                            "http_retryable" if retryable else "http_terminal"
                        ),
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                if not can_retry:
                    raise OpenAlexProviderHTTPError(int(exc.code)) from exc
                time.sleep(
                    self.base_backoff_seconds * (2 ** (attempt - 1))
                )
            except URLError as exc:
                can_retry = attempt < self.max_attempts
                self.attempt_diagnostics.append(
                    OpenAlexTransportAttemptDiagnostic(
                        query_id=query_id,
                        attempt=attempt,
                        outcome="retry" if can_retry else "failure",
                        exception_type=type(exc).__name__,
                        error_class="url_error",
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                if not can_retry:
                    raise OpenAlexProviderNetworkError(
                        "OpenAlex network error after bounded attempts."
                    ) from exc
                time.sleep(
                    self.base_backoff_seconds * (2 ** (attempt - 1))
                )
        raise RuntimeError("OpenAlex request exhausted without a result.")

    def search(self, query: CatalogQuery, *, limit: int) -> list[CatalogWork]:
        if not self.api_key:
            raise RuntimeError("OPENALEX_API_KEY missing.")

        params = urlencode(
            {
                "search": query.query_text,
                "per_page": max(1, min(100, int(limit))),
                "api_key": self.api_key,
                "select": (
                    "id,doi,display_name,publication_year,"
                    "publication_date,type,primary_location,open_access"
                ),
            }
        )
        started = time.perf_counter()
        try:
            payload = self._request_json(
                url="https://api.openalex.org/works?" + params,
                query_id=query.query_id,
            )
        finally:
            elapsed = time.perf_counter() - started
            if self.minimum_interval_seconds > elapsed:
                time.sleep(self.minimum_interval_seconds - elapsed)

        items = payload.get("results", []) if isinstance(payload, dict) else []
        rows: list[CatalogWork] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("display_name") or "").strip()
            if not title:
                continue
            doi = normalize_doi(item.get("doi"))
            openalex_id = _openalex_id(item.get("id"))
            primary = (
                item.get("primary_location")
                if isinstance(item.get("primary_location"), dict)
                else {}
            )
            oa = (
                item.get("open_access")
                if isinstance(item.get("open_access"), dict)
                else {}
            )
            source = (
                primary.get("source")
                if isinstance(primary.get("source"), dict)
                else {}
            )
            landing = str(primary.get("landing_page_url") or "").strip()
            pdf = str(primary.get("pdf_url") or "").strip()
            oa_url = str(oa.get("oa_url") or "").strip()
            venue = str(source.get("display_name") or "").strip()
            publication_type = str(item.get("type") or "").strip()

            rows.append(
                CatalogWork(
                    work_id=_stable_id(
                        "catalog_work",
                        doi or openalex_id or title,
                    ),
                    title=title,
                    year=(
                        int(item["publication_year"])
                        if item.get("publication_year") is not None
                        else None
                    ),
                    publication_date=(
                        str(item.get("publication_date"))
                        if item.get("publication_date")
                        else None
                    ),
                    doi=doi,
                    url=landing or (
                        str(item.get("id"))
                        if item.get("id")
                        else None
                    ),
                    open_access_url=pdf or oa_url or None,
                    abstract=None,
                    authors=[],
                    venue=venue or None,
                    citation_count=None,
                    publication_types=(
                        [publication_type] if publication_type else []
                    ),
                    providers=[self.provider_name],
                    provider_ids=(
                        {self.provider_name: openalex_id}
                        if openalex_id
                        else {}
                    ),
                    retrieval_query_ids=[query.query_id],
                    retrieval_axis_ids=[query.axis_id],
                )
            )
        return rows


def _sanitize_error(text: str | None, queries: Sequence[str]) -> str | None:
    if text is None:
        return None
    value = " ".join(str(text).split())
    for query in queries:
        value = value.replace(query, "<query-redacted>")
    return value[:400] or None


def make_transport_diagnostics_payload_v2_4(
    *,
    protocol: ProviderSubstitutionProtocol,
    executions: Sequence[CatalogQueryExecution],
    openalex_attempts: Sequence[OpenAlexTransportAttemptDiagnostic],
) -> dict[str, Any]:
    rows = []
    for execution in executions:
        error_type = None
        if execution.error:
            error_type = str(execution.error).split(":", 1)[0].strip()
        rows.append(
            {
                "provider": execution.provider,
                "query_id": execution.query_id,
                "success": execution.success,
                "result_count": execution.result_count,
                "elapsed_seconds": execution.elapsed_seconds,
                "error_type": error_type or None,
                "error_summary": _sanitize_error(
                    execution.error,
                    protocol.broad_queries,
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": (
            "sers-fresh-c-openalex-crossref-transport-diagnostics-v2-4"
        ),
        "protocol_id": protocol.protocol_id,
        "parent_v22_attempt_id": protocol.v22_failed_attempt_id,
        "credential_presence": {
            "openalex_api_key_present": bool(os.getenv("OPENALEX_API_KEY")),
            "crossref_mailto_present": bool(os.getenv("CROSSREF_MAILTO")),
        },
        "provider_executions": rows,
        "openalex_transport_attempts": [
            row.model_dump(mode="json") for row in openalex_attempts
        ],
        "provider_substitution": {
            "from": "semantic_scholar",
            "to": "openalex",
            "reason": (
                "transport_availability_only_after_repeated_http_429"
            ),
        },
        "query_text_persisted": False,
        "title_persisted": False,
        "abstract_persisted": False,
        "citation_count_persisted": False,
        "raw_provider_packet_persisted": False,
        "credential_values_persisted": False,
        "fresh_reserve_c_consumed": False,
        "semantic_read_performed": False,
        "llm_calls": 0,
    }
    payload["diagnostics_sha256"] = _payload_sha(
        payload, "diagnostics_sha256"
    )
    return payload
