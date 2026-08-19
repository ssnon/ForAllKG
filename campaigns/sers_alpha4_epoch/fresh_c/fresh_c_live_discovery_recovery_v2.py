from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_acquisition import (
    FRESH_C_BLIND_ORDER_NAMESPACE,
    sha256_json,
)
from campaigns.sers_alpha4_epoch.fresh_c.fresh_c_live_discovery import (
    EXPECTED_BROAD_QUERIES,
    EXPECTED_HISTORICAL_IDENTITY_COUNT,
    EXPECTED_PROVIDERS,
    TARGET_ACQUIRED_PAPERS,
)
from dac_her.literature_catalog import normalize_doi
from dac_her.literature_catalog_contracts import (
    CatalogQuery,
    CatalogQueryExecution,
    CatalogWork,
)

RECOVERY_V2_SEMANTICS_ID = (
    "sers_fresh_c_live_metadata_discovery_recovery_v2"
)
RECOVERY_V2_PROTOCOL_PREFIX = (
    "sers_fresh_c_live_discovery_recovery_protocol_v2"
)

EXPECTED_V1_PROTOCOL_ID = (
    "sers_fresh_c_live_discovery_protocol_v1:ab8a9fe230a95b047b82"
)
EXPECTED_V1_PROTOCOL_SHA256 = (
    "b8270261fa4371c1245923910f75ddac61a10a52682f120f46a5b4e97e988a96"
)
EXPECTED_V1_FREEZE_ID = (
    "sers_fresh_c_live_discovery_protocol_freeze_v1:"
    "da61ad7bd0e49e140713"
)
EXPECTED_V1_FREEZE_MANIFEST_SHA256 = (
    "e851d06725fd144ce7dc644638d766cc0b5ced2d1bc819715f274f35b5b6ab3c"
)
EXPECTED_V1_FREEZE_COMMIT = (
    "0947ef7ed7dd85be4ade3367b92d8ee49793a266"
)
EXPECTED_V1_FAILED_ATTEMPT_ID = (
    "sers_fresh_c_live_discovery_attempt_v1:0912aca95ffe39b9f8a3"
)

V1_FAILED_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_live_discovery_run_v1"
)
V1_STARTED_PATH = V1_FAILED_RUN_DIR / "DISCOVERY_STARTED.json"
V1_FAILED_PATH = V1_FAILED_RUN_DIR / "DISCOVERY_FAILED.json"

DEFAULT_PROTOCOL_PATH = Path(
    "dac_her/sers_fresh_c_live_discovery_recovery_v2_protocol.json"
)
DEFAULT_FREEZE_DIR = Path(
    "evaluation/sers_fresh_c/"
    "c0_1c_v2_recovery_protocol_freeze_v1"
)
DEFAULT_RUN_DIR = Path(
    "evaluation/sers_fresh_c/c0_1c_v2_recovery_run_v1"
)

SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS = 1.10
SEMANTIC_SCHOLAR_MAX_ATTEMPTS = 4
SEMANTIC_SCHOLAR_BASE_BACKOFF_SECONDS = 2.0
SEMANTIC_SCHOLAR_MAX_RETRY_AFTER_SECONDS = 60.0
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TransportPolicy(StrictModel):
    semantic_scholar_minimum_interval_seconds: Literal[1.1]
    semantic_scholar_max_attempts: Literal[4]
    semantic_scholar_base_backoff_seconds: Literal[2.0]
    semantic_scholar_max_retry_after_seconds: Literal[60.0]
    retryable_http_status: list[int]
    retry_after_respected: Literal[True]
    semantic_scholar_api_key_optional: Literal[True]
    credential_values_persisted: Literal[False]
    crossref_transport_semantics_changed: Literal[False]
    scientific_response_body_persisted_in_diagnostics: Literal[False]
    query_text_persisted_in_diagnostics: Literal[False]
    title_or_abstract_persisted_in_diagnostics: Literal[False]


class RecoveryV2Protocol(StrictModel):
    schema_version: Literal[
        "sers-fresh-c-live-discovery-recovery-protocol-v2"
    ]
    protocol_id: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantics_id: Literal[
        "sers_fresh_c_live_metadata_discovery_recovery_v2"
    ]
    stage: Literal["C0.1C-v2"]

    recovery_parent_protocol_id: Literal[
        "sers_fresh_c_live_discovery_protocol_v1:ab8a9fe230a95b047b82"
    ]
    recovery_parent_protocol_sha256: Literal[
        "b8270261fa4371c1245923910f75ddac61a10a52682f120f46a5b4e97e988a96"
    ]
    recovery_parent_freeze_id: Literal[
        "sers_fresh_c_live_discovery_protocol_freeze_v1:"
        "da61ad7bd0e49e140713"
    ]
    recovery_parent_freeze_manifest_sha256: Literal[
        "e851d06725fd144ce7dc644638d766cc0b5ced2d1bc819715f274f35b5b6ab3c"
    ]
    recovery_parent_freeze_commit: Literal[
        "0947ef7ed7dd85be4ade3367b92d8ee49793a266"
    ]
    recovery_parent_attempt_id: Literal[
        "sers_fresh_c_live_discovery_attempt_v1:0912aca95ffe39b9f8a3"
    ]
    recovery_parent_failed_epoch_must_be_preserved: Literal[True]
    recovery_parent_same_epoch_rerun_allowed: Literal[False]

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

    search_queries_changed_from_v1: Literal[False]
    provider_set_changed_from_v1: Literal[False]
    search_depth_changed_from_v1: Literal[False]
    historical_ledger_changed_from_v1: Literal[False]
    target_count_changed_from_v1: Literal[False]
    blind_ordering_changed_from_v1: Literal[False]
    scientific_selection_semantics_changed_from_v1: Literal[False]

    transport_policy: TransportPolicy

    every_provider_query_execution_must_succeed: Literal[True]
    explicit_recovery_confirmation_required: Literal[True]
    recovery_started_marker_before_first_network_call: Literal[True]
    same_recovery_epoch_rerun_after_start_allowed: Literal[False]
    failure_authorizes_query_or_selection_tuning: Literal[False]
    fresh_reserve_c_consumption_occurs_in_recovery: Literal[False]
    semantic_read_allowed_in_recovery: Literal[False]
    automatic_c0_1d_transition_allowed: Literal[False]
    stop_after_success: Literal[True]
    llm_calls: Literal[0]

    @model_validator(mode="after")
    def _exact_contract(self) -> "RecoveryV2Protocol":
        if self.providers != EXPECTED_PROVIDERS:
            raise ValueError("Recovery-v2 provider set drifted.")
        if self.broad_queries != EXPECTED_BROAD_QUERIES:
            raise ValueError("Recovery-v2 query set drifted.")
        if self.historical_identity_count != EXPECTED_HISTORICAL_IDENTITY_COUNT:
            raise ValueError("Recovery-v2 historical identity count drifted.")
        if self.target_acquired_papers != TARGET_ACQUIRED_PAPERS:
            raise ValueError("Recovery-v2 target count drifted.")
        if self.transport_policy.retryable_http_status != sorted(
            RETRYABLE_HTTP_STATUS
        ):
            raise ValueError("Recovery-v2 retryable HTTP set drifted.")
        return self


class TransportAttemptDiagnostic(StrictModel):
    provider: str
    query_id: str
    attempt: int = Field(ge=1)
    outcome: Literal["success", "retry", "failure"]
    http_status: int | None = None
    retry_after_seconds: float | None = None
    exception_type: str | None = None
    error_class: str | None = None
    elapsed_seconds: float = Field(ge=0.0)


class ProviderExecutionDiagnostic(StrictModel):
    provider: str
    query_id: str
    success: bool
    result_count: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0)
    error_type: str | None = None
    error_summary: str | None = None
    http_status: int | None = None


class RecoveryProviderHTTPError(RuntimeError):
    def __init__(
        self,
        *,
        status: int,
        retry_after_seconds: float | None,
        original_type: str,
    ) -> None:
        self.status = int(status)
        self.retry_after_seconds = retry_after_seconds
        self.original_type = original_type
        super().__init__(
            f"HTTP status {self.status}; retry_after_seconds="
            f"{self.retry_after_seconds}"
        )


class RecoveryProviderNetworkError(RuntimeError):
    pass


def _payload_sha(
    payload: Mapping[str, Any],
    sha_field: str,
) -> str:
    value = dict(payload)
    value.pop(sha_field, None)
    return sha256_json(value)


def _protocol_identity_sha(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("protocol_id", None)
    value.pop("protocol_sha256", None)
    return sha256_json(value)


def expected_protocol_id(payload: Mapping[str, Any]) -> str:
    return (
        RECOVERY_V2_PROTOCOL_PREFIX
        + ":"
        + _protocol_identity_sha(payload)[:20]
    )


def load_and_validate_protocol(path: Path) -> RecoveryV2Protocol:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Recovery-v2 protocol must be a JSON object.")
    protocol = RecoveryV2Protocol.model_validate(raw)
    if protocol.protocol_id != expected_protocol_id(raw):
        raise ValueError("Recovery-v2 protocol ID mismatch.")
    if protocol.protocol_sha256 != _payload_sha(raw, "protocol_sha256"):
        raise ValueError("Recovery-v2 protocol SHA mismatch.")
    return protocol


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def validate_v1_failed_epoch(root: Path) -> dict[str, Any]:
    started_path = root / V1_STARTED_PATH
    failed_path = root / V1_FAILED_PATH
    if not started_path.exists() or not failed_path.exists():
        raise FileNotFoundError(
            "Recovery parent failed-epoch markers are missing."
        )
    started = _read_json(started_path)
    failed = _read_json(failed_path)

    if started.get("attempt_id") != EXPECTED_V1_FAILED_ATTEMPT_ID:
        raise ValueError("v1 STARTED attempt ID drifted.")
    if failed.get("attempt_id") != EXPECTED_V1_FAILED_ATTEMPT_ID:
        raise ValueError("v1 FAILED attempt ID drifted.")
    if started.get("network_boundary_opened") is not True:
        raise ValueError("v1 STARTED network boundary flag drifted.")
    if started.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("v1 STARTED rerun guard drifted.")
    if failed.get("network_boundary_opened") is not True:
        raise ValueError("v1 FAILED network boundary flag drifted.")
    if failed.get("same_epoch_rerun_allowed") is not False:
        raise ValueError("v1 FAILED rerun guard drifted.")
    if failed.get("new_protocol_epoch_required") is not True:
        raise ValueError("v1 FAILED recovery requirement drifted.")
    for row in (started, failed):
        if row.get("fresh_reserve_c_consumed") is not False:
            raise ValueError("v1 failed epoch unexpectedly consumed Fresh C.")
        if row.get("semantic_read_performed") is not False:
            raise ValueError("v1 failed epoch unexpectedly read science.")
    return {
        "started": started,
        "failed": failed,
    }


def _retry_after_seconds(
    headers: Any,
    *,
    max_seconds: float,
) -> float | None:
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            value = (
                when.astimezone(timezone.utc)
                - datetime.now(timezone.utc)
            ).total_seconds()
        except Exception:
            return None
    return max(0.0, min(float(max_seconds), float(value)))


def _strip_markup(value: Any) -> str | None:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    return text or None


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class DiagnosticSemanticScholarCatalogProvider:
    provider_name = "semantic_scholar"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY",
        timeout: float = 30.0,
        minimum_interval_seconds: float = (
            SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS
        ),
        max_attempts: int = SEMANTIC_SCHOLAR_MAX_ATTEMPTS,
        base_backoff_seconds: float = (
            SEMANTIC_SCHOLAR_BASE_BACKOFF_SECONDS
        ),
        max_retry_after_seconds: float = (
            SEMANTIC_SCHOLAR_MAX_RETRY_AFTER_SECONDS
        ),
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv(api_key_env)
        )
        self.timeout = float(timeout)
        self.minimum_interval_seconds = float(
            minimum_interval_seconds
        )
        self.max_attempts = int(max_attempts)
        self.base_backoff_seconds = float(base_backoff_seconds)
        self.max_retry_after_seconds = float(
            max_retry_after_seconds
        )
        self.attempt_diagnostics: list[
            TransportAttemptDiagnostic
        ] = []

    def _request_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        query_id: str,
    ) -> Any:
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                request = Request(url, headers=headers)
                with urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:
                    status = int(
                        getattr(response, "status", None)
                        or response.getcode()
                        or 200
                    )
                    payload = json.loads(
                        response.read().decode("utf-8")
                    )
                self.attempt_diagnostics.append(
                    TransportAttemptDiagnostic(
                        provider=self.provider_name,
                        query_id=query_id,
                        attempt=attempt,
                        outcome="success",
                        http_status=status,
                        elapsed_seconds=(
                            time.perf_counter() - started
                        ),
                    )
                )
                return payload
            except HTTPError as exc:
                retry_after = _retry_after_seconds(
                    exc.headers,
                    max_seconds=self.max_retry_after_seconds,
                )
                retryable = exc.code in RETRYABLE_HTTP_STATUS
                can_retry = retryable and attempt < self.max_attempts
                self.attempt_diagnostics.append(
                    TransportAttemptDiagnostic(
                        provider=self.provider_name,
                        query_id=query_id,
                        attempt=attempt,
                        outcome=("retry" if can_retry else "failure"),
                        http_status=int(exc.code),
                        retry_after_seconds=retry_after,
                        exception_type=type(exc).__name__,
                        error_class="http_retryable" if retryable else "http_terminal",
                        elapsed_seconds=(
                            time.perf_counter() - started
                        ),
                    )
                )
                if not can_retry:
                    raise RecoveryProviderHTTPError(
                        status=int(exc.code),
                        retry_after_seconds=retry_after,
                        original_type=type(exc).__name__,
                    ) from exc
                wait = self.base_backoff_seconds * (
                    2 ** (attempt - 1)
                )
                if retry_after is not None:
                    wait = max(wait, retry_after)
                time.sleep(
                    min(
                        self.max_retry_after_seconds,
                        wait,
                    )
                )
            except URLError as exc:
                can_retry = attempt < self.max_attempts
                self.attempt_diagnostics.append(
                    TransportAttemptDiagnostic(
                        provider=self.provider_name,
                        query_id=query_id,
                        attempt=attempt,
                        outcome=("retry" if can_retry else "failure"),
                        exception_type=type(exc).__name__,
                        error_class="url_error",
                        elapsed_seconds=(
                            time.perf_counter() - started
                        ),
                    )
                )
                if not can_retry:
                    raise RecoveryProviderNetworkError(
                        "Semantic Scholar network error after "
                        f"{self.max_attempts} attempts."
                    ) from exc
                time.sleep(
                    self.base_backoff_seconds
                    * (2 ** (attempt - 1))
                )
        raise RuntimeError(
            "Semantic Scholar request exhausted without result."
        )

    def search(
        self,
        query: CatalogQuery,
        *,
        limit: int,
    ) -> list[CatalogWork]:
        fields = (
            "title,abstract,year,authors,venue,url,externalIds,"
            "citationCount,openAccessPdf,publicationDate,"
            "publicationTypes"
        )
        params = urlencode(
            {
                "query": query.query_text,
                "limit": max(1, min(100, int(limit))),
                "fields": fields,
            }
        )
        headers = {
            "User-Agent": "GraphAgentsDAC-FreshC-RecoveryV2"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        started = time.perf_counter()
        try:
            payload = self._request_json(
                url=(
                    "https://api.semanticscholar.org/"
                    "graph/v1/paper/search?"
                    + params
                ),
                headers=headers,
                query_id=query.query_id,
            )
        finally:
            elapsed = time.perf_counter() - started
            if self.minimum_interval_seconds > elapsed:
                time.sleep(
                    self.minimum_interval_seconds - elapsed
                )

        rows: list[CatalogWork] = []
        items = (
            payload.get("data", [])
            if isinstance(payload, dict)
            else []
        )
        for item in items:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            external = item.get("externalIds") or {}
            doi = (
                normalize_doi(external.get("DOI"))
                if isinstance(external, dict)
                else None
            )
            authors = [
                str(row.get("name") or "").strip()
                for row in item.get("authors") or []
                if isinstance(row, dict)
                and str(row.get("name") or "").strip()
            ]
            oa = item.get("openAccessPdf") or {}
            oa_url = (
                str(oa.get("url") or "").strip()
                if isinstance(oa, dict)
                else ""
            )
            provider_id = str(item.get("paperId") or "")
            publication_types = [
                str(value).strip()
                for value in item.get("publicationTypes") or []
                if str(value).strip()
            ]
            rows.append(
                CatalogWork(
                    work_id=_stable_id(
                        "catalog_work",
                        doi or provider_id or title,
                    ),
                    title=title,
                    year=(
                        int(item["year"])
                        if item.get("year") is not None
                        else None
                    ),
                    publication_date=(
                        str(item.get("publicationDate"))
                        if item.get("publicationDate")
                        else None
                    ),
                    doi=doi,
                    url=(
                        str(item.get("url"))
                        if item.get("url")
                        else None
                    ),
                    open_access_url=oa_url or None,
                    abstract=_strip_markup(item.get("abstract")),
                    authors=authors,
                    venue=(
                        str(item.get("venue"))
                        if item.get("venue")
                        else None
                    ),
                    citation_count=(
                        int(item["citationCount"])
                        if item.get("citationCount") is not None
                        else None
                    ),
                    publication_types=publication_types,
                    providers=[self.provider_name],
                    provider_ids=(
                        {self.provider_name: provider_id}
                        if provider_id
                        else {}
                    ),
                    retrieval_query_ids=[query.query_id],
                    retrieval_axis_ids=[query.axis_id],
                )
            )
        return rows


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


def make_transport_diagnostics_payload(
    *,
    protocol: RecoveryV2Protocol,
    executions: Sequence[CatalogQueryExecution],
    semantic_scholar_attempts: Sequence[
        TransportAttemptDiagnostic
    ],
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
                    forbidden_queries=protocol.broad_queries,
                ),
                http_status=_execution_http_status(
                    execution.error
                ),
            )
        )

    payload: dict[str, Any] = {
        "schema_version": (
            "sers-fresh-c-live-discovery-transport-diagnostics-v2"
        ),
        "protocol_id": protocol.protocol_id,
        "recovery_parent_attempt_id": (
            protocol.recovery_parent_attempt_id
        ),
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
