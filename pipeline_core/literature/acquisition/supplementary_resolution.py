from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from pipeline_core.literature.acquisition.access_contracts import AccessResolution
from pipeline_core.literature.acquisition.supplementary_contracts import SupplementaryCandidate, SupplementaryDiscovery, SupplementaryDiscoveryPolicy, SupplementaryResolverAttempt
from pipeline_core.literature.catalog_contracts import CatalogWork


_STRONG_ANCHOR_PATTERNS = (
    "supporting information",
    "supporting info",
    "supplementary information",
    "supplementary material",
    "supplemental material",
    "electronic supplementary information",
    "supplementary data",
    "supplemental data",
    "supporting data",
    "supplementary file",
    "supplemental file",
    "additional file",
)

_MEDIUM_URL_PATTERNS = (
    "supplement",
    "supp_info",
    "supporting-information",
    "supporting_information",
    "_si_",
    "/esi/",
    "electronic-supplementary",
    "mmc",
)

_FILE_EXTENSIONS = (
    ".pdf",
    ".zip",
    ".xlsx",
    ".xls",
    ".csv",
    ".docx",
    ".doc",
    ".txt",
    ".pptx",
)

_STRONG_CROSSREF_RELATIONS = {
    "is-supplemented-by",
    "issupplementedby",
}
_MEDIUM_CROSSREF_RELATIONS = {
    "has-related-material",
    "hasrelatedmaterial",
    "has-part",
    "haspart",
}


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _norm(value: str | None) -> str:
    text = html.unescape(str(value or "")).casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _relation_norm(value: str) -> str:
    return re.sub(r"[^a-z]+", "", str(value or "").casefold())


def _public_http_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host or host in {"localhost", "localhost.localdomain"}:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _looks_file_like(url: str) -> bool:
    path = urlparse(url).path.casefold()
    return any(path.endswith(ext) for ext in _FILE_EXTENSIONS)


def _request_bytes(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    retries: int,
    retry_backoff: float,
    max_bytes: int,
    accept: str,
) -> tuple[bytes, str, str]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": accept,
                },
            )
            with urlopen(request, timeout=timeout) as response:
                resolved = response.geturl()
                content_type = str(
                    response.headers.get("Content-Type") or ""
                ).split(";", 1)[0].strip().lower()
                chunks = []
                total = 0
                while True:
                    chunk = response.read(min(1024 * 1024, max_bytes + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError(
                            f"response_exceeds_max_bytes:{max_bytes}"
                        )
                    chunks.append(chunk)
                return b"".join(chunks), resolved, content_type
        except HTTPError as exc:
            last = exc
            if (
                exc.code not in {429, 500, 502, 503, 504}
                or attempt >= retries
            ):
                raise
        except URLError as exc:
            last = exc
            if attempt >= retries:
                raise
        time.sleep(retry_backoff * (2**attempt))
    if last is not None:
        raise last
    raise RuntimeError("request failed without an exception")


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.casefold() != "a":
            return
        mapping = {
            str(key).casefold(): str(value or "")
            for key, value in attrs
            if key
        }
        href = mapping.get("href", "").strip()
        if not href:
            return
        self._current = {
            "href": href,
            "title": mapping.get("title", ""),
            "download": mapping.get("download", ""),
        }
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "a" or self._current is None:
            return
        row = dict(self._current)
        row["text"] = " ".join("".join(self._text).split())
        self.links.append(row)
        self._current = None
        self._text = []


def _anchor_candidate(
    *,
    work_id: str,
    source_page_url: str,
    resolved_source_page_url: str,
    href: str,
    anchor_text: str,
    title_hint: str,
) -> SupplementaryCandidate | None:
    absolute = urljoin(resolved_source_page_url, href)
    if not _public_http_url(absolute):
        return None

    anchor_norm = _norm(anchor_text + " " + title_hint)
    url_norm = absolute.casefold()

    strong_text = any(
        pattern in anchor_norm for pattern in _STRONG_ANCHOR_PATTERNS
    ) or anchor_norm == "esi"
    medium_url = any(pattern in url_norm for pattern in _MEDIUM_URL_PATTERNS)
    file_like = _looks_file_like(absolute)

    if not strong_text and not medium_url:
        return None

    if file_like and strong_text:
        confidence = "high"
        kind = "direct_file"
        auto = True
        reasons = [
            "public_landing_anchor",
            "strong_supplementary_anchor_text",
            "file_like_url",
        ]
    elif file_like and medium_url:
        confidence = "medium"
        kind = "direct_file"
        auto = False
        reasons = [
            "public_landing_anchor",
            "supplementary_url_pattern",
            "file_like_url",
        ]
    else:
        confidence = "medium" if strong_text else "low"
        kind = "supplementary_landing"
        auto = False
        reasons = [
            "public_landing_anchor",
            (
                "strong_supplementary_anchor_text"
                if strong_text
                else "supplementary_url_pattern"
            ),
            "not_direct_file",
        ]

    return SupplementaryCandidate(
        candidate_id=_stable_id(
            "supp_candidate",
            work_id,
            "public_landing_html",
            absolute,
        ),
        work_id=work_id,
        kind=kind,
        resolver="public_landing_html",
        source_page_url=source_page_url,
        resolved_source_page_url=resolved_source_page_url,
        url=absolute,
        anchor_text=anchor_text or None,
        title_hint=title_hint or None,
        confidence=confidence,
        automatic_download_eligible=auto,
        reason_codes=reasons,
    )


def _crossref_candidates(
    *,
    work: CatalogWork,
    relation: dict[str, Any],
) -> list[SupplementaryCandidate]:
    rows: list[SupplementaryCandidate] = []
    for relation_type, values in relation.items():
        relation_key = str(relation_type)
        normalized = _relation_norm(relation_key)
        if normalized in {
            _relation_norm(value) for value in _STRONG_CROSSREF_RELATIONS
        }:
            confidence = "high"
            reason = "crossref_explicit_supplement_relation"
        elif normalized in {
            _relation_norm(value) for value in _MEDIUM_CROSSREF_RELATIONS
        }:
            confidence = "medium"
            reason = "crossref_related_material_relation"
        else:
            continue

        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            continue

        for row in values:
            if not isinstance(row, dict):
                continue
            identifier = str(
                row.get("id")
                or row.get("identifier")
                or row.get("id-value")
                or ""
            ).strip()
            identifier_type = str(
                row.get("id-type")
                or row.get("identifier-type")
                or ""
            ).strip().casefold()
            if not identifier:
                continue

            direct_url = None
            if identifier_type in {"uri", "url"} and _public_http_url(identifier):
                direct_url = identifier
            elif identifier.startswith(("http://", "https://")) and _public_http_url(identifier):
                direct_url = identifier

            is_direct_file = bool(direct_url and _looks_file_like(direct_url))
            rows.append(
                SupplementaryCandidate(
                    candidate_id=_stable_id(
                        "supp_candidate",
                        work.work_id,
                        "crossref_relation",
                        relation_key,
                        identifier_type,
                        identifier,
                    ),
                    work_id=work.work_id,
                    kind=(
                        "direct_file"
                        if is_direct_file
                        else "related_identifier"
                    ),
                    resolver="crossref_relation",
                    url=direct_url,
                    identifier=identifier,
                    identifier_type=identifier_type or None,
                    relation_type=relation_key,
                    confidence=confidence,
                    automatic_download_eligible=(
                        bool(is_direct_file and confidence == "high")
                    ),
                    reason_codes=[
                        reason,
                        (
                            "direct_file_identifier"
                            if is_direct_file
                            else "metadata_only_related_identifier"
                        ),
                    ],
                )
            )
    return rows


def _deduplicate_candidates(
    rows: list[SupplementaryCandidate],
) -> list[SupplementaryCandidate]:
    priority = {"high": 0, "medium": 1, "low": 2}
    by_key: dict[str, SupplementaryCandidate] = {}
    for row in rows:
        key = (
            ("url:" + str(row.url).strip())
            if row.url
            else (
                "id:"
                + str(row.identifier_type or "")
                + ":"
                + str(row.identifier or "")
            )
        )
        current = by_key.get(key)
        if current is None or priority[row.confidence] < priority[current.confidence]:
            by_key[key] = row
    return sorted(
        by_key.values(),
        key=lambda row: (
            priority[row.confidence],
            0 if row.kind == "direct_file" else 1,
            row.url or row.identifier or row.candidate_id,
        ),
    )


@dataclass(frozen=True)
class SupplementaryArtifactResolver:
    policy: SupplementaryDiscoveryPolicy

    def discover(
        self,
        *,
        work: CatalogWork,
        main_access: AccessResolution | None,
    ) -> SupplementaryDiscovery:
        candidates: list[SupplementaryCandidate] = []
        attempts: list[SupplementaryResolverAttempt] = []
        notes: list[str] = []
        scanned_pages: list[str] = []

        if self.policy.use_crossref_relations:
            if not work.doi:
                attempts.append(
                    SupplementaryResolverAttempt(
                        resolver="crossref_relation",
                        status="skipped",
                        message="missing_doi",
                    )
                )
            else:
                started = time.perf_counter()
                try:
                    params = {}
                    mailto = str(
                        os.getenv(self.policy.crossref_mailto_env) or ""
                    ).strip()
                    if mailto:
                        params["mailto"] = mailto
                    suffix = ("?" + urlencode(params)) if params else ""
                    payload, _, _ = _request_bytes(
                        (
                            "https://api.crossref.org/works/"
                            + quote(work.doi, safe="")
                            + suffix
                        ),
                        user_agent=self.policy.user_agent,
                        timeout=self.policy.request_timeout_seconds,
                        retries=self.policy.retries,
                        retry_backoff=self.policy.retry_backoff_seconds,
                        max_bytes=self.policy.max_html_bytes,
                        accept="application/json",
                    )
                    loaded = json.loads(payload.decode("utf-8"))
                    message = (
                        loaded.get("message", {})
                        if isinstance(loaded, dict)
                        else {}
                    )
                    relation = (
                        message.get("relation", {})
                        if isinstance(message, dict)
                        else {}
                    )
                    relation = relation if isinstance(relation, dict) else {}
                    found = _crossref_candidates(
                        work=work,
                        relation=relation,
                    )
                    candidates.extend(found)
                    attempts.append(
                        SupplementaryResolverAttempt(
                            resolver="crossref_relation",
                            status="success",
                            elapsed_seconds=time.perf_counter() - started,
                            result_count=len(found),
                        )
                    )
                except Exception as exc:
                    attempts.append(
                        SupplementaryResolverAttempt(
                            resolver="crossref_relation",
                            status="failed",
                            elapsed_seconds=time.perf_counter() - started,
                            message=f"{type(exc).__name__}: {exc}",
                        )
                    )
                finally:
                    if self.policy.resolver_delay_seconds > 0:
                        time.sleep(self.policy.resolver_delay_seconds)

        if self.policy.use_public_landing_html:
            page_candidates: list[str] = []
            if main_access is not None:
                for location in main_access.locations:
                    for value in (
                        location.url_for_landing_page,
                        (
                            location.url
                            if not location.automatic_download_eligible
                            else None
                        ),
                    ):
                        url = str(value or "").strip()
                        if url and _public_http_url(url):
                            page_candidates.append(url)
            if work.doi:
                page_candidates.append(
                    "https://doi.org/" + quote(work.doi, safe="/():._-")
                )
            work_url = str(work.url or "").strip()
            if (
                work_url
                and _public_http_url(work_url)
                and "semanticscholar.org" not in urlparse(work_url).netloc.casefold()
            ):
                page_candidates.append(work_url)

            unique_pages: list[str] = []
            for url in page_candidates:
                if url not in unique_pages:
                    unique_pages.append(url)
            unique_pages = unique_pages[: self.policy.max_landing_pages_per_work]

            if not unique_pages:
                attempts.append(
                    SupplementaryResolverAttempt(
                        resolver="public_landing_html",
                        status="skipped",
                        message="no_public_landing_candidate",
                    )
                )
            else:
                for source_page in unique_pages:
                    started = time.perf_counter()
                    try:
                        body, resolved_page, content_type = _request_bytes(
                            source_page,
                            user_agent=self.policy.user_agent,
                            timeout=self.policy.request_timeout_seconds,
                            retries=self.policy.retries,
                            retry_backoff=self.policy.retry_backoff_seconds,
                            max_bytes=self.policy.max_html_bytes,
                            accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                        )
                        scanned_pages.append(resolved_page)
                        if content_type and content_type not in {
                            "text/html",
                            "application/xhtml+xml",
                        }:
                            attempts.append(
                                SupplementaryResolverAttempt(
                                    resolver="public_landing_html",
                                    status="skipped",
                                    elapsed_seconds=time.perf_counter() - started,
                                    message=f"non_html_content_type:{content_type}",
                                )
                            )
                            continue
                        parser = _AnchorParser()
                        parser.feed(body.decode("utf-8", errors="replace"))
                        found = []
                        for link in parser.links:
                            candidate = _anchor_candidate(
                                work_id=work.work_id,
                                source_page_url=source_page,
                                resolved_source_page_url=resolved_page,
                                href=link["href"],
                                anchor_text=link.get("text", ""),
                                title_hint=link.get("title", ""),
                            )
                            if candidate is not None:
                                found.append(candidate)
                        candidates.extend(found)
                        attempts.append(
                            SupplementaryResolverAttempt(
                                resolver="public_landing_html",
                                status="success",
                                elapsed_seconds=time.perf_counter() - started,
                                result_count=len(found),
                                message=f"resolved_page={resolved_page}",
                            )
                        )
                    except Exception as exc:
                        attempts.append(
                            SupplementaryResolverAttempt(
                                resolver="public_landing_html",
                                status="failed",
                                elapsed_seconds=time.perf_counter() - started,
                                message=f"{type(exc).__name__}: {exc}",
                            )
                        )
                    finally:
                        if self.policy.resolver_delay_seconds > 0:
                            time.sleep(self.policy.resolver_delay_seconds)

        candidates = _deduplicate_candidates(candidates)
        candidates = candidates[: self.policy.max_candidates_per_work]

        if any(
            row.kind == "direct_file"
            and row.automatic_download_eligible
            for row in candidates
        ):
            status = "direct_file_candidates"
        elif candidates:
            status = "metadata_only_candidates"
        else:
            status = "unresolved"

        return SupplementaryDiscovery(
            work_id=work.work_id,
            doi=work.doi,
            status=status,
            candidates=candidates,
            resolver_attempts=attempts,
            scanned_landing_pages=sorted(set(scanned_pages)),
            discovery_notes=notes,
            publisher_specific_url_guessing_performed=False,
            paywall_bypass_attempted=False,
        )
