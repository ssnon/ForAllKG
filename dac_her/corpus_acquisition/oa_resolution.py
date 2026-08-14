from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from dac_her.corpus_acquisition.access_contracts import (
    AccessLocation,
    AccessResolution,
    ResolverAttempt,
    SourceAcquisitionPolicy,
)
from dac_her.literature_catalog_contracts import CatalogWork


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


def _request_json(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> Any:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/json",
                },
            )
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
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


def _location_from_unpaywall(
    *,
    work_id: str,
    row: dict[str, Any],
    is_best: bool,
) -> AccessLocation | None:
    pdf_url = str(row.get("url_for_pdf") or "").strip()
    landing = str(row.get("url_for_landing_page") or "").strip()
    generic_url = str(row.get("url") or "").strip()
    url = pdf_url or generic_url or landing
    if not url or not _public_http_url(url):
        return None

    direct_pdf = bool(pdf_url and _public_http_url(pdf_url))
    reason_codes = ["unpaywall_oa_location"]
    if is_best:
        reason_codes.append("best_oa_location")
    if direct_pdf:
        reason_codes.append("direct_pdf_url")
    else:
        reason_codes.append("landing_only")

    return AccessLocation(
        location_id=_stable_id(
            "access_location",
            work_id,
            "unpaywall",
            url,
        ),
        resolver="unpaywall",
        url=url,
        url_for_pdf=pdf_url or None,
        url_for_landing_page=landing or None,
        is_oa=True,
        is_best=bool(is_best),
        host_type=(
            str(row.get("host_type"))
            if row.get("host_type")
            else None
        ),
        version=(
            str(row.get("version"))
            if row.get("version")
            else None
        ),
        license=(
            str(row.get("license"))
            if row.get("license")
            else None
        ),
        automatic_download_eligible=direct_pdf,
        reason_codes=reason_codes,
    )


def _catalog_location(
    work: CatalogWork,
) -> AccessLocation | None:
    url = str(work.open_access_url or "").strip()
    if not url or not _public_http_url(url):
        return None
    return AccessLocation(
        location_id=_stable_id(
            "access_location",
            work.work_id,
            "catalog_open_access",
            url,
        ),
        resolver="catalog_open_access",
        url=url,
        url_for_pdf=url,
        url_for_landing_page=None,
        is_oa=True,
        is_best=False,
        automatic_download_eligible=True,
        reason_codes=[
            "provider_reported_open_access_pdf_candidate",
            "download_requires_pdf_validation",
        ],
    )


@dataclass(frozen=True)
class OpenAccessResolver:
    policy: SourceAcquisitionPolicy

    def resolve(
        self,
        work: CatalogWork,
    ) -> AccessResolution:
        attempts: list[ResolverAttempt] = []
        locations: list[AccessLocation] = []
        notes: list[str] = []

        email = (
            os.getenv(self.policy.unpaywall_email_env)
            or os.getenv(self.policy.fallback_email_env)
            or ""
        ).strip()

        if self.policy.use_unpaywall:
            if not work.doi:
                attempts.append(
                    ResolverAttempt(
                        resolver="unpaywall",
                        status="skipped",
                        message="missing_doi",
                    )
                )
            elif not email:
                attempts.append(
                    ResolverAttempt(
                        resolver="unpaywall",
                        status="skipped",
                        message=(
                            "missing_email_env:"
                            f"{self.policy.unpaywall_email_env}"
                        ),
                    )
                )
                notes.append(
                    "Unpaywall requires an email parameter; "
                    "catalog OA fallback may still be used."
                )
            else:
                started = time.perf_counter()
                try:
                    endpoint = (
                        "https://api.unpaywall.org/v2/"
                        + quote(work.doi, safe="")
                        + "?"
                        + urlencode({"email": email})
                    )
                    payload = _request_json(
                        endpoint,
                        user_agent=self.policy.user_agent,
                        timeout=self.policy.request_timeout_seconds,
                        retries=self.policy.retries,
                        retry_backoff=self.policy.retry_backoff_seconds,
                    )
                    if not isinstance(payload, dict):
                        raise ValueError(
                            "Unpaywall response is not an object"
                        )

                    best = payload.get("best_oa_location")
                    if isinstance(best, dict):
                        location = _location_from_unpaywall(
                            work_id=work.work_id,
                            row=best,
                            is_best=True,
                        )
                        if location is not None:
                            locations.append(location)

                    for row in payload.get("oa_locations") or []:
                        if not isinstance(row, dict):
                            continue
                        location = _location_from_unpaywall(
                            work_id=work.work_id,
                            row=row,
                            is_best=False,
                        )
                        if location is None:
                            continue
                        if all(
                            location.location_id != existing.location_id
                            for existing in locations
                        ):
                            locations.append(location)

                    attempts.append(
                        ResolverAttempt(
                            resolver="unpaywall",
                            status="success",
                            elapsed_seconds=(
                                time.perf_counter() - started
                            ),
                            message=(
                                f"is_oa={bool(payload.get('is_oa'))};"
                                f" locations={len(locations)}"
                            ),
                        )
                    )
                except Exception as exc:
                    attempts.append(
                        ResolverAttempt(
                            resolver="unpaywall",
                            status="failed",
                            elapsed_seconds=(
                                time.perf_counter() - started
                            ),
                            message=(
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    )
                finally:
                    if self.policy.resolver_delay_seconds > 0:
                        time.sleep(
                            self.policy.resolver_delay_seconds
                        )

        if self.policy.use_catalog_open_access_url:
            fallback = _catalog_location(work)
            if fallback is not None:
                if all(
                    fallback.location_id != existing.location_id
                    for existing in locations
                ):
                    locations.append(fallback)
                attempts.append(
                    ResolverAttempt(
                        resolver="catalog_open_access",
                        status="success",
                        message="open_access_url_present",
                    )
                )
            else:
                attempts.append(
                    ResolverAttempt(
                        resolver="catalog_open_access",
                        status="skipped",
                        message="no_open_access_url",
                    )
                )

        # Stable priority: best direct Unpaywall PDF, other direct Unpaywall
        # PDFs, catalog OA PDF candidate, then landing-only locations.
        locations = sorted(
            locations,
            key=lambda row: (
                0
                if (
                    row.resolver == "unpaywall"
                    and row.is_best
                    and row.automatic_download_eligible
                )
                else 1
                if (
                    row.resolver == "unpaywall"
                    and row.automatic_download_eligible
                )
                else 2
                if (
                    row.resolver == "catalog_open_access"
                    and row.automatic_download_eligible
                )
                else 3,
                row.location_id,
            ),
        )

        selected = next(
            (
                row
                for row in locations
                if row.automatic_download_eligible
            ),
            None,
        )
        if selected is not None:
            status = "resolved_direct_pdf"
            selected_download_url = (
                selected.url_for_pdf or selected.url
            )
            selected_location_id = selected.location_id
        elif locations:
            status = "resolved_landing_only"
            selected_download_url = None
            selected_location_id = None
        else:
            status = "unresolved"
            selected_download_url = None
            selected_location_id = None

        return AccessResolution(
            work_id=work.work_id,
            doi=work.doi,
            status=status,
            locations=locations,
            resolver_attempts=attempts,
            selected_location_id=selected_location_id,
            selected_download_url=selected_download_url,
            resolution_notes=notes,
            paywall_bypass_attempted=False,
        )
