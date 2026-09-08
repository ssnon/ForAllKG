from __future__ import annotations

import hashlib
import ipaddress
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pipeline_core.literature.acquisition.access_contracts import (
    AccessLocation,
    ResolverAttempt,
    SourceAcquisitionPolicy,
)
from pipeline_core.literature.catalog_contracts import CatalogWork


_SUPPLEMENTARY_DOI_RE = re.compile(r"\.s\d+$", re.I)


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


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


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urljoin(req.full_url, newurl)
        if not _public_http_url(target):
            raise RuntimeError("redirect_target_not_public")
        return super().redirect_request(req, fp, code, msg, headers, target)


def _request_html(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    retries: int,
    retry_backoff: float,
    max_bytes: int,
) -> tuple[str, str, str]:
    """Fetch one public landing page with a strict size bound.

    A return of ("", resolved_url, "application/pdf") is reserved for a
    response whose bytes actually start with PDF magic. A misleading
    Content-Type alone is never sufficient.
    """
    last: Exception | None = None
    opener = build_opener(_SafeRedirectHandler())

    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": (
                        "text/html,application/xhtml+xml,"
                        "application/xml;q=0.8,*/*;q=0.2"
                    ),
                    "Accept-Language": "en-US,en;q=0.8",
                },
            )
            with opener.open(request, timeout=timeout) as response:
                resolved = response.geturl()
                if not _public_http_url(resolved):
                    raise RuntimeError("resolved_target_not_public")

                content_type = str(
                    response.headers.get("Content-Type") or ""
                ).split(";", 1)[0].strip().lower()

                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(min(256 * 1024, max_bytes + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError(
                            f"landing_html_exceeds_max_bytes:{max_bytes}"
                        )
                    chunks.append(chunk)

                raw = b"".join(chunks)
                if raw.startswith(b"%PDF-"):
                    return "", resolved, "application/pdf"

                charset = response.headers.get_content_charset() or "utf-8"
                return raw.decode(charset, errors="replace"), resolved, content_type

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
    raise RuntimeError("landing request failed without an exception")


_MAIN_META_NAMES = {
    "citation_pdf_url",
    "bepress_citation_pdf_url",
    "eprints.document_url",
    "pdf_url",
    "fulltext_pdf_url",
}

_MAIN_ANCHOR_PHRASES = (
    "download pdf",
    "download this paper",
    "open pdf",
    "open pdf in browser",
    "view pdf",
    "full text pdf",
    "full-text pdf",
    "article pdf",
)

_SUPPLEMENTARY_PATTERNS = (
    "supplement",
    "supplementary",
    "supporting information",
    "supporting-information",
    "supporting_information",
    "_si.",
    "_si_",
    "/si/",
    "/supp/",
)


def _looks_supplementary(url: str, label: str = "") -> bool:
    text = f"{url} {label}".casefold()
    return any(pattern in text for pattern in _SUPPLEMENTARY_PATTERNS)


class _MainPdfParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta_urls: list[str] = []
        self.link_pdf_urls: list[str] = []
        self.anchor_candidates: list[tuple[str, str]] = []
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        mapping = {
            str(key).casefold(): str(value or "")
            for key, value in attrs
            if key
        }
        tag_norm = tag.casefold()

        if tag_norm == "meta":
            name = (
                mapping.get("name")
                or mapping.get("property")
                or ""
            ).casefold().strip()
            content = mapping.get("content", "").strip()
            if name in _MAIN_META_NAMES and content:
                self.meta_urls.append(content)

        elif tag_norm == "link":
            href = mapping.get("href", "").strip()
            type_value = mapping.get("type", "").casefold().strip()
            rel_value = mapping.get("rel", "").casefold().strip()
            if (
                href
                and "application/pdf" in type_value
                and "alternate" in rel_value
            ):
                self.link_pdf_urls.append(href)

        elif tag_norm == "a":
            href = mapping.get("href", "").strip()
            if href:
                self._anchor_href = href
                self._anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._anchor_href is not None:
            self._anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "a" or self._anchor_href is None:
            return
        text = " ".join("".join(self._anchor_text).split())
        self.anchor_candidates.append((self._anchor_href, text))
        self._anchor_href = None
        self._anchor_text = []


def _extract_main_pdf_urls(
    html_text: str,
    *,
    resolved_page_url: str,
) -> list[tuple[str, str]]:
    parser = _MainPdfParser()
    parser.feed(html_text)

    rows: list[tuple[str, str]] = []

    def add(raw_url: str, reason: str, label: str = "") -> None:
        absolute = urljoin(resolved_page_url, raw_url)
        if not _public_http_url(absolute):
            return
        if _looks_supplementary(absolute, label):
            return
        if all(existing_url != absolute for existing_url, _ in rows):
            rows.append((absolute, reason))

    for raw_url in parser.meta_urls:
        add(raw_url, "landing_meta_pdf")

    for raw_url in parser.link_pdf_urls:
        add(raw_url, "landing_link_application_pdf")

    for href, text in parser.anchor_candidates:
        norm = " ".join(text.casefold().split())
        if any(phrase in norm for phrase in _MAIN_ANCHOR_PHRASES):
            add(href, "landing_strong_pdf_anchor", text)

    return rows


@dataclass(frozen=True)
class PublicLandingMainPdfProbe:
    attempt: ResolverAttempt
    locations: list[AccessLocation]


Fetcher = Callable[..., tuple[str, str, str]]


@dataclass(frozen=True)
class PublicLandingMainPdfResolver:
    """Recover public main-PDF candidates from public landing HTML.

    No authentication, JavaScript execution, or paywall bypass is attempted.
    The candidate is not labelled OA merely because it is publicly reachable;
    its access/license status remains unverified. Existing downstream PDF-magic
    validation is still required before a SourceArtifact can be downloaded.
    """

    policy: SourceAcquisitionPolicy
    fetcher: Fetcher = _request_html

    def resolve(
        self,
        work: CatalogWork,
        *,
        existing_locations: list[AccessLocation],
    ) -> PublicLandingMainPdfProbe:
        if not self.policy.use_public_landing_html:
            return PublicLandingMainPdfProbe(
                attempt=ResolverAttempt(
                    resolver="public_landing_html",
                    status="skipped",
                    message="disabled_by_policy",
                ),
                locations=[],
            )

        if work.doi and _SUPPLEMENTARY_DOI_RE.search(work.doi):
            return PublicLandingMainPdfProbe(
                attempt=ResolverAttempt(
                    resolver="public_landing_html",
                    status="skipped",
                    message="supplementary_doi_not_main_work",
                ),
                locations=[],
            )

        page_candidates: list[str] = []

        for location in existing_locations:
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
            and "semanticscholar.org"
            not in urlparse(work_url).netloc.casefold()
        ):
            page_candidates.append(work_url)

        unique_pages: list[str] = []
        for url in page_candidates:
            if url not in unique_pages:
                unique_pages.append(url)

        unique_pages = unique_pages[
            : self.policy.max_landing_pages_per_work
        ]

        if not unique_pages:
            return PublicLandingMainPdfProbe(
                attempt=ResolverAttempt(
                    resolver="public_landing_html",
                    status="skipped",
                    message="no_public_landing_candidate",
                ),
                locations=[],
            )

        started = time.perf_counter()
        locations: list[AccessLocation] = []
        successes = 0
        failures = 0
        scanned = 0
        direct_page_pdf = 0

        for source_page in unique_pages:
            scanned += 1
            try:
                body, resolved_page, content_type = self.fetcher(
                    source_page,
                    user_agent=self.policy.download_user_agent,
                    timeout=self.policy.request_timeout_seconds,
                    retries=self.policy.retries,
                    retry_backoff=self.policy.retry_backoff_seconds,
                    max_bytes=self.policy.landing_html_max_bytes,
                )

                if content_type == "application/pdf" and body == "":
                    if not _looks_supplementary(resolved_page):
                        locations.append(
                            AccessLocation(
                                location_id=_stable_id(
                                    "access_location",
                                    work.work_id,
                                    "public_landing_html",
                                    resolved_page,
                                ),
                                resolver="public_landing_html",
                                url=resolved_page,
                                url_for_pdf=resolved_page,
                                url_for_landing_page=source_page,
                                is_oa=False,
                                is_best=False,
                                automatic_download_eligible=True,
                                reason_codes=[
                                    "public_landing_resolution",
                                    "landing_resolved_direct_pdf",
                                    "oa_status_unverified",
                                    "download_requires_pdf_validation",
                                ],
                            )
                        )
                        direct_page_pdf += 1
                    successes += 1
                    continue

                found = _extract_main_pdf_urls(
                    body,
                    resolved_page_url=resolved_page,
                )
                for pdf_url, reason in found:
                    locations.append(
                        AccessLocation(
                            location_id=_stable_id(
                                "access_location",
                                work.work_id,
                                "public_landing_html",
                                pdf_url,
                            ),
                            resolver="public_landing_html",
                            url=pdf_url,
                            url_for_pdf=pdf_url,
                            url_for_landing_page=resolved_page,
                            is_oa=False,
                            is_best=False,
                            automatic_download_eligible=True,
                            reason_codes=[
                                "public_landing_resolution",
                                reason,
                                "oa_status_unverified",
                                "download_requires_pdf_validation",
                            ],
                        )
                    )
                successes += 1
            except Exception:
                failures += 1
            finally:
                if self.policy.resolver_delay_seconds > 0:
                    time.sleep(self.policy.resolver_delay_seconds)

        deduped: list[AccessLocation] = []
        seen_urls: set[str] = set()
        for location in locations:
            url = str(location.url_for_pdf or location.url).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(location)

        status = "success" if successes > 0 else "failed"
        return PublicLandingMainPdfProbe(
            attempt=ResolverAttempt(
                resolver="public_landing_html",
                status=status,
                elapsed_seconds=time.perf_counter() - started,
                message=(
                    f"pages={scanned}; successes={successes}; "
                    f"failures={failures}; locations={len(deduped)}; "
                    f"direct_page_pdf={direct_page_pdf}"
                ),
            ),
            locations=deduped,
        )
