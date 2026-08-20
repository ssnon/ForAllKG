from __future__ import annotations

import hashlib
import ipaddress
import os
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from pipeline_core.literature.acquisition.access_contracts import AccessLocation, ResolverAttempt, SourceAcquisitionPolicy
from pipeline_core.literature.catalog_contracts import CatalogWork
from pipeline_core.literature.discovery.providers.openalex import (
    OpenAlexHTTPError,
    OpenAlexProvider,
    openalex_location_metadata,
)


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


@dataclass(frozen=True)
class OpenAlexAccessProbe:
    attempt: ResolverAttempt
    locations: list[AccessLocation]


ProviderFactory = Callable[..., OpenAlexProvider]


@dataclass(frozen=True)
class OpenAlexAccessResolver:
    """Adapt existing OpenAlex discovery transport into the M3 access lane.

    The adapter performs identifier-targeted lookup through OpenAlexProvider,
    then converts every explicitly OA location into the existing AccessLocation
    contract. It does not download artifacts and it never promotes metadata to
    positive scientific evidence.
    """

    policy: SourceAcquisitionPolicy
    provider_factory: ProviderFactory = OpenAlexProvider

    def resolve(self, work: CatalogWork) -> OpenAlexAccessProbe:
        openalex_id = str(work.provider_ids.get("openalex") or "").strip()
        identifier = openalex_id or str(work.doi or "").strip()
        if not identifier:
            return OpenAlexAccessProbe(
                attempt=ResolverAttempt(
                    resolver="openalex",
                    status="skipped",
                    message="missing_doi_and_openalex_id",
                ),
                locations=[],
            )

        api_key = str(
            os.getenv(self.policy.openalex_api_key_env) or ""
        ).strip()
        if self.policy.openalex_require_api_key and not api_key:
            return OpenAlexAccessProbe(
                attempt=ResolverAttempt(
                    resolver="openalex",
                    status="skipped",
                    message=(
                        "missing_api_key_env:"
                        f"{self.policy.openalex_api_key_env}"
                    ),
                ),
                locations=[],
            )

        mailto = str(
            os.getenv(self.policy.openalex_mailto_env)
            or os.getenv(self.policy.fallback_email_env)
            or ""
        ).strip()

        provider = self.provider_factory(
            api_key=api_key or None,
            mailto=mailto or None,
            timeout=self.policy.request_timeout_seconds,
            max_retries=self.policy.retries,
            backoff_base_seconds=self.policy.retry_backoff_seconds,
        )
        started = time.perf_counter()
        try:
            raw_work = provider.get_work(identifier)
            locations = _locations_from_openalex_work(
                work_id=work.work_id,
                raw_work=raw_work,
            )
            direct_count = sum(
                row.automatic_download_eligible for row in locations
            )
            attempt = ResolverAttempt(
                resolver="openalex",
                status="success",
                elapsed_seconds=time.perf_counter() - started,
                message=(
                    f"locations={len(locations)};"
                    f" direct_pdf={direct_count};"
                    f" identifier={'openalex_id' if openalex_id else 'doi'}"
                ),
            )
        except OpenAlexHTTPError as exc:
            if exc.status == 404:
                attempt = ResolverAttempt(
                    resolver="openalex",
                    status="success",
                    elapsed_seconds=time.perf_counter() - started,
                    message="work_not_found",
                )
                locations = []
            else:
                attempt = ResolverAttempt(
                    resolver="openalex",
                    status="failed",
                    elapsed_seconds=time.perf_counter() - started,
                    message=f"{type(exc).__name__}: {exc}",
                )
                locations = []
        except Exception as exc:
            attempt = ResolverAttempt(
                resolver="openalex",
                status="failed",
                elapsed_seconds=time.perf_counter() - started,
                message=f"{type(exc).__name__}: {exc}",
            )
            locations = []
        finally:
            if self.policy.resolver_delay_seconds > 0:
                time.sleep(self.policy.resolver_delay_seconds)

        return OpenAlexAccessProbe(
            attempt=attempt,
            locations=locations,
        )


def _locations_from_openalex_work(
    *,
    work_id: str,
    raw_work: dict[str, Any],
) -> list[AccessLocation]:
    """Collect best, primary, and all explicitly OA OpenAlex locations."""

    candidates: list[tuple[dict[str, Any], bool, bool]] = []
    best = raw_work.get("best_oa_location")
    if isinstance(best, dict):
        candidates.append((best, True, False))
    primary = raw_work.get("primary_location")
    if isinstance(primary, dict):
        candidates.append((primary, False, True))
    for raw in raw_work.get("locations") or []:
        if isinstance(raw, dict):
            candidates.append((raw, False, False))

    # Keep one row per concrete URL pair while preserving role annotations.
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for raw, is_best, is_primary in candidates:
        metadata = openalex_location_metadata(raw)
        if not metadata or metadata.get("is_oa") is not True:
            continue
        pdf_url = str(metadata.get("pdf_url") or "").strip()
        landing = str(metadata.get("landing_page_url") or "").strip()
        usable_pdf = pdf_url if _public_http_url(pdf_url) else ""
        usable_landing = landing if _public_http_url(landing) else ""
        if not usable_pdf and not usable_landing:
            continue

        key = (usable_pdf, usable_landing)
        row = merged.setdefault(
            key,
            {
                "metadata": metadata,
                "is_best": False,
                "is_primary": False,
            },
        )
        row["is_best"] = bool(row["is_best"] or is_best)
        row["is_primary"] = bool(row["is_primary"] or is_primary)

    locations: list[AccessLocation] = []
    for (pdf_url, landing), item in merged.items():
        metadata = item["metadata"]
        direct_pdf = bool(pdf_url)
        url = pdf_url or landing
        reason_codes = ["openalex_oa_location"]
        if item["is_best"]:
            reason_codes.append("best_oa_location")
        if item["is_primary"]:
            reason_codes.append("primary_location")
        reason_codes.append(
            "direct_pdf_url" if direct_pdf else "landing_only"
        )

        locations.append(
            AccessLocation(
                location_id=_stable_id(
                    "access_location",
                    work_id,
                    "openalex",
                    url,
                    landing,
                ),
                resolver="openalex",
                url=url,
                url_for_pdf=pdf_url or None,
                url_for_landing_page=landing or None,
                is_oa=True,
                is_best=bool(item["is_best"]),
                host_type=(
                    str(metadata.get("source_type"))
                    if metadata.get("source_type")
                    else None
                ),
                version=(
                    str(metadata.get("version"))
                    if metadata.get("version")
                    else None
                ),
                license=(
                    str(metadata.get("license"))
                    if metadata.get("license")
                    else None
                ),
                source_id=(
                    str(metadata.get("source_id"))
                    if metadata.get("source_id")
                    else None
                ),
                source_name=(
                    str(metadata.get("source_name"))
                    if metadata.get("source_name")
                    else None
                ),
                automatic_download_eligible=direct_pdf,
                reason_codes=reason_codes,
            )
        )

    return sorted(
        locations,
        key=lambda row: (
            0 if row.is_best else 1,
            0 if "primary_location" in row.reason_codes else 1,
            row.location_id,
        ),
    )
