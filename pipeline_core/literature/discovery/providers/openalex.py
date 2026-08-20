from __future__ import annotations

import json
import time
from dataclasses import dataclass
from email.message import Message
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ..contracts import LiteratureRecord
from .base import LiteratureSearchRequest


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class OpenAlexTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str],
        timeout: float,
        headers: Mapping[str, str],
    ) -> TransportResponse: ...


class UrllibOpenAlexTransport:
    """Small stdlib transport so discovery adds no new HTTP dependency."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str],
        timeout: float,
        headers: Mapping[str, str],
    ) -> TransportResponse:
        query = urlencode(params)
        target = f"{url}?{query}" if query else url
        request = Request(target, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS API URL
                response_headers = _headers_to_dict(response.headers)
                return TransportResponse(
                    status=int(response.status),
                    headers=response_headers,
                    body=response.read(),
                )
        except HTTPError as exc:
            return TransportResponse(
                status=int(exc.code),
                headers=_headers_to_dict(exc.headers),
                body=exc.read(),
            )
        except URLError as exc:
            raise OpenAlexTransportError(str(exc.reason)) from exc


class OpenAlexError(RuntimeError):
    pass


class OpenAlexTransportError(OpenAlexError):
    pass


class OpenAlexHTTPError(OpenAlexError):
    def __init__(self, status: int, message: str):
        super().__init__(f"OpenAlex HTTP {status}: {message}")
        self.status = status


def _headers_to_dict(headers: Message | Mapping[str, str] | None) -> dict[str, str]:
    if headers is None:
        return {}
    return {str(key): str(value) for key, value in headers.items()}


def reconstruct_abstract(inverted_index: Any) -> str | None:
    """Reconstruct OpenAlex abstract_inverted_index into plain text."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None

    positioned: list[tuple[int, str]] = []
    for token, raw_positions in inverted_index.items():
        if not isinstance(raw_positions, list):
            continue
        for raw_position in raw_positions:
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                continue
            if position < 0:
                continue
            positioned.append((position, str(token)))

    if not positioned:
        return None
    positioned.sort(key=lambda item: (item[0], item[1]))
    return " ".join(token for _, token in positioned).strip() or None


def _openalex_provider_id(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        raise ValueError("OpenAlex work is missing id")
    return text.rsplit("/", 1)[-1]


def _nested(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    value = next(
        (raw for key, raw in headers.items() if key.lower() == "retry-after"),
        None,
    )
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, seconds)


def _openalex_work_lookup_url(identifier: str) -> str:
    text = str(identifier or "").strip().rstrip("/")
    if not text:
        raise ValueError("OpenAlex work identifier must not be empty")

    lowered = text.casefold()
    if lowered.startswith("https://openalex.org/"):
        provider_id = _openalex_provider_id(text)
        return f"{OPENALEX_WORKS_URL}/{provider_id}"
    if text[:1].upper() == "W" and text[1:].isdigit():
        return f"{OPENALEX_WORKS_URL}/{text.upper()}"

    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if not text.startswith("10.") or "/" not in text:
        raise ValueError(f"Unsupported OpenAlex work identifier: {identifier!r}")

    doi_url = f"https://doi.org/{text}"
    return f"{OPENALEX_WORKS_URL}/{quote(doi_url, safe=':/')}"


class OpenAlexProvider:
    provider_name = "openalex"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        mailto: str | None = None,
        timeout: float = 30.0,
        page_size: int = 100,
        max_retries: int = 4,
        backoff_base_seconds: float = 1.0,
        transport: OpenAlexTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds must be non-negative")

        self.api_key = (api_key or "").strip() or None
        self.mailto = (mailto or "").strip() or None
        self.timeout = float(timeout)
        self.page_size = int(page_size)
        self.max_retries = int(max_retries)
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.transport = transport or UrllibOpenAlexTransport()
        self.sleep = sleep

    def search(self, request: LiteratureSearchRequest) -> list[LiteratureRecord]:
        records: list[LiteratureRecord] = []
        cursor = "*"

        while len(records) < request.limit:
            remaining = request.limit - len(records)
            per_page = min(self.page_size, remaining)
            payload = self._request_page(
                query=request.query,
                cursor=cursor,
                per_page=per_page,
            )
            raw_results = payload.get("results")
            if not isinstance(raw_results, list) or not raw_results:
                break

            for work in raw_results:
                if not isinstance(work, dict):
                    continue
                record = self._normalize_work(
                    work,
                    discovery_query=request.query,
                    mechanism_bucket=request.mechanism_bucket,
                )
                if record is not None:
                    records.append(record)
                if len(records) >= request.limit:
                    break

            meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)

        return records[: request.limit]

    def _request_json(
        self,
        url: str,
        *,
        params: Mapping[str, str],
    ) -> dict[str, Any]:
        request_params = dict(params)
        if self.api_key:
            request_params["api_key"] = self.api_key
        if self.mailto:
            request_params["mailto"] = self.mailto

        headers = {
            "Accept": "application/json",
            "User-Agent": "GraphAgentsDAC-literature-discovery/0.1",
        }

        attempt = 0
        while True:
            try:
                response = self.transport.get(
                    url,
                    params=request_params,
                    timeout=self.timeout,
                    headers=headers,
                )
            except OpenAlexTransportError:
                if attempt >= self.max_retries:
                    raise
                self.sleep(self.backoff_base_seconds * (2**attempt))
                attempt += 1
                continue

            if response.status == 200:
                try:
                    payload = json.loads(response.body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise OpenAlexError("OpenAlex returned invalid JSON") from exc
                if not isinstance(payload, dict):
                    raise OpenAlexError("OpenAlex response must be a JSON object")
                return payload

            retryable = response.status == 429 or 500 <= response.status < 600
            if retryable and attempt < self.max_retries:
                retry_after = _retry_after_seconds(response.headers)
                backoff = self.backoff_base_seconds * (2**attempt)
                self.sleep(max(backoff, retry_after or 0.0))
                attempt += 1
                continue

            message = _response_error_message(response.body)
            raise OpenAlexHTTPError(response.status, message)

    def _request_page(
        self,
        *,
        query: str,
        cursor: str,
        per_page: int,
    ) -> dict[str, Any]:
        return self._request_json(
            OPENALEX_WORKS_URL,
            params={
                "search": query,
                "per_page": str(per_page),
                "cursor": cursor,
            },
        )

    def get_work(self, identifier: str) -> dict[str, Any]:
        """Fetch one work by OpenAlex ID or DOI using shared retry machinery."""
        return self._request_json(
            _openalex_work_lookup_url(identifier),
            params={},
        )

    def _normalize_work(
        self,
        work: dict[str, Any],
        *,
        discovery_query: str,
        mechanism_bucket: str,
    ) -> LiteratureRecord | None:
        title = str(work.get("title") or work.get("display_name") or "").strip()
        if not title:
            return None

        try:
            provider_id = _openalex_provider_id(work.get("id"))
        except ValueError:
            return None

        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        source = _nested(work, "primary_location", "source")
        venue = source.get("display_name") if isinstance(source, dict) else None

        metadata: dict[str, Any] = {
            "openalex_id": work.get("id"),
            "publication_date": work.get("publication_date"),
            "work_type": work.get("type"),
            "language": work.get("language"),
            "cited_by_count": work.get("cited_by_count"),
            "relevance_score": work.get("relevance_score"),
            "is_retracted": work.get("is_retracted"),
            "is_paratext": work.get("is_paratext"),
            "primary_location": _location_metadata(work.get("primary_location")),
            "best_oa_location": _location_metadata(work.get("best_oa_location")),
            "open_access": dict(work.get("open_access") or {}),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        year = work.get("publication_year")
        try:
            normalized_year = int(year) if year is not None else None
        except (TypeError, ValueError):
            normalized_year = None

        return LiteratureRecord.from_provider_result(
            provider=self.provider_name,
            provider_id=provider_id,
            title=title,
            abstract=abstract,
            doi=work.get("doi"),
            year=normalized_year,
            venue=(str(venue) if venue else None),
            discovery_query=discovery_query,
            mechanism_bucket=mechanism_bucket,
            metadata=metadata,
        )


def openalex_location_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    source = value.get("source") if isinstance(value.get("source"), dict) else {}
    result = {
        "is_oa": value.get("is_oa"),
        "landing_page_url": value.get("landing_page_url"),
        "pdf_url": value.get("pdf_url"),
        "license": value.get("license"),
        "version": value.get("version"),
        "source_id": source.get("id"),
        "source_name": source.get("display_name"),
        "source_type": source.get("type"),
    }
    cleaned = {key: item for key, item in result.items() if item is not None}
    return cleaned or None


def _location_metadata(value: Any) -> dict[str, Any] | None:
    # Compatibility wrapper for existing discovery normalization.
    return openalex_location_metadata(value)


def _response_error_message(body: bytes) -> str:
    if not body:
        return "empty response"
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return body.decode("utf-8", errors="replace")[:500]
    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if value:
                return str(value)[:500]
    return str(payload)[:500]
