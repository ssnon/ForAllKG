from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from pipeline_core.literature.acquisition.access_contracts import AccessLocation, ResolverAttempt, SourceAcquisitionPolicy
from pipeline_core.literature.catalog_contracts import CatalogWork


PMC_AWS_BUCKET = (
    "https://pmc-oa-opendata.s3.amazonaws.com"
)

_PMCID_RE = re.compile(
    r"(?i)\b(PMC[0-9]+)\b"
)


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(value)
        for value in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _normalize_doi(value: str | None) -> str:
    text = str(value or "").strip().lower()

    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]

    return text.strip()


def _numeric_pmcid_from_official_url(
    value: str | None,
) -> str | None:
    """Normalize numeric PMC article URLs into canonical PMCIDs.

    Only official NCBI/PMC article URL shapes are accepted. Arbitrary
    numeric paths and non-PMC hosts are deliberately ignored.
    """

    parsed = urlparse(
        str(value or "").strip()
    )

    host = (
        parsed.hostname
        or ""
    ).lower()

    path = parsed.path

    match = None

    if host in {
        "www.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
    }:
        match = re.match(
            r"^/pmc/articles/([0-9]+)(?:/|$)",
            path,
            flags=re.IGNORECASE,
        )

    elif host == "pmc.ncbi.nlm.nih.gov":
        match = re.match(
            r"^/articles/([0-9]+)(?:/|$)",
            path,
            flags=re.IGNORECASE,
        )

    if match is None:
        return None

    return "PMC" + match.group(1)


def extract_pmcids(
    locations: list[AccessLocation],
) -> list[str]:
    """Extract strong PMCID identifiers from existing OA locations."""

    rows: set[str] = set()

    for location in locations:
        values = [
            location.url,
            location.url_for_pdf,
            location.url_for_landing_page,
            location.source_id,
        ]

        for value in values:
            text = str(value or "")

            for match in _PMCID_RE.finditer(text):
                rows.add(
                    match.group(1).upper()
                )

            normalized = (
                _numeric_pmcid_from_official_url(
                    text
                )
            )

            if normalized is not None:
                rows.add(normalized)

    return sorted(rows)


def _request_bytes(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> bytes:
    last: Exception | None = None

    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": (
                        "application/json,"
                        "application/xml,"
                        "text/xml;q=0.9,"
                        "*/*;q=0.1"
                    ),
                },
            )

            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                return response.read()

        except HTTPError as exc:
            last = exc

            if (
                exc.code not in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                or attempt >= retries
            ):
                raise

        except URLError as exc:
            last = exc

            if attempt >= retries:
                raise

        time.sleep(
            retry_backoff * (2**attempt)
        )

    if last is not None:
        raise last

    raise RuntimeError(
        "PMC AWS request failed without exception"
    )


def _version_prefixes(
    payload: bytes,
    *,
    pmcid: str,
) -> list[str]:
    root = ElementTree.fromstring(payload)

    ns = (
        "{http://s3.amazonaws.com/"
        "doc/2006-03-01/}"
    )

    rows: list[tuple[int, str]] = []

    for node in root.findall(
        f"{ns}CommonPrefixes"
    ):
        prefix = node.findtext(
            f"{ns}Prefix"
        )

        prefix = str(
            prefix or ""
        ).strip().rstrip("/")

        match = re.fullmatch(
            rf"{re.escape(pmcid)}\.([0-9]+)",
            prefix,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue

        rows.append(
            (
                int(match.group(1)),
                prefix,
            )
        )

    rows.sort(
        key=lambda row: row[0],
        reverse=True,
    )

    return [
        prefix
        for _, prefix in rows
    ]


def _s3_to_https(
    value: str,
) -> str | None:
    parsed = urlparse(
        str(value or "").strip()
    )

    if parsed.scheme == "https":
        return value

    if parsed.scheme != "s3":
        return None

    bucket = parsed.netloc.strip()

    if not bucket:
        return None

    return urlunparse(
        (
            "https",
            f"{bucket}.s3.amazonaws.com",
            parsed.path,
            "",
            parsed.query,
            "",
        )
    )


@dataclass(frozen=True)
class PmcAwsAccessProbe:
    attempt: ResolverAttempt
    locations: list[AccessLocation]


RequestBytes = Callable[..., bytes]


@dataclass(frozen=True)
class PmcAwsAccessResolver:
    """Resolve known PMCIDs into current PMC AWS public PDF objects."""

    policy: SourceAcquisitionPolicy
    request_bytes: RequestBytes = _request_bytes

    def resolve(
        self,
        work: CatalogWork,
        *,
        existing_locations: list[AccessLocation],
    ) -> PmcAwsAccessProbe:
        started = time.perf_counter()

        pmcids = extract_pmcids(
            existing_locations
        )

        if not pmcids:
            return PmcAwsAccessProbe(
                attempt=ResolverAttempt(
                    resolver="pmc_aws",
                    status="skipped",
                    message=(
                        "no_pmcid_in_existing_locations"
                    ),
                ),
                locations=[],
            )

        locations: list[AccessLocation] = []

        successful_lookup_count = 0
        failure_count = 0

        for pmcid in pmcids:
            try:
                query = urlencode(
                    {
                        "list-type": "2",
                        "prefix": f"{pmcid}.",
                        "delimiter": "/",
                    }
                )

                list_payload = self.request_bytes(
                    f"{PMC_AWS_BUCKET}/?{query}",
                    user_agent=self.policy.user_agent,
                    timeout=(
                        self.policy
                        .request_timeout_seconds
                    ),
                    retries=self.policy.retries,
                    retry_backoff=(
                        self.policy
                        .retry_backoff_seconds
                    ),
                )

                versions = _version_prefixes(
                    list_payload,
                    pmcid=pmcid,
                )

                successful_lookup_count += 1

            except Exception:
                failure_count += 1
                continue

            for version_prefix in versions:
                metadata_url = (
                    f"{PMC_AWS_BUCKET}/metadata/"
                    f"{version_prefix}.json"
                )

                try:
                    metadata_payload = (
                        self.request_bytes(
                            metadata_url,
                            user_agent=(
                                self.policy.user_agent
                            ),
                            timeout=(
                                self.policy
                                .request_timeout_seconds
                            ),
                            retries=(
                                self.policy.retries
                            ),
                            retry_backoff=(
                                self.policy
                                .retry_backoff_seconds
                            ),
                        )
                    )

                    metadata = json.loads(
                        metadata_payload
                        .decode("utf-8")
                    )

                except Exception:
                    failure_count += 1
                    continue

                if not isinstance(
                    metadata,
                    dict,
                ):
                    continue

                metadata_pmcid = str(
                    metadata.get("pmcid")
                    or ""
                ).strip().upper()

                if metadata_pmcid != pmcid:
                    continue

                if (
                    metadata.get(
                        "is_retracted"
                    )
                    is True
                ):
                    continue

                if (
                    metadata.get(
                        "is_pmc_openaccess"
                    )
                    is not True
                ):
                    continue

                metadata_doi = _normalize_doi(
                    metadata.get("doi")
                )

                work_doi = _normalize_doi(
                    work.doi
                )

                if (
                    metadata_doi
                    and work_doi
                    and metadata_doi != work_doi
                ):
                    continue

                pdf_url = _s3_to_https(
                    str(
                        metadata.get(
                            "pdf_url"
                        )
                        or ""
                    )
                )

                if not pdf_url:
                    continue

                version = metadata.get(
                    "version"
                )

                location = AccessLocation(
                    location_id=_stable_id(
                        "access_location",
                        work.work_id,
                        "pmc_aws",
                        pmcid,
                        version_prefix,
                        pdf_url,
                    ),
                    resolver="pmc_aws",
                    url=pdf_url,
                    url_for_pdf=pdf_url,
                    url_for_landing_page=(
                        "https://pmc.ncbi.nlm.nih.gov/"
                        f"articles/{pmcid}/"
                    ),
                    is_oa=True,
                    is_best=True,
                    host_type="repository",
                    version=(
                        str(version)
                        if version is not None
                        else None
                    ),
                    license=(
                        str(
                            metadata.get(
                                "license_code"
                            )
                        )
                        if metadata.get(
                            "license_code"
                        )
                        else None
                    ),
                    source_id=pmcid,
                    source_name="PubMed Central",
                    automatic_download_eligible=True,
                    reason_codes=[
                        "pmc_aws_current_dataset",
                        "public_repository",
                        "pmc_open_access_subset",
                        "direct_pdf_url",
                    ],
                )

                if all(
                    location.location_id
                    != existing.location_id
                    for existing in locations
                ):
                    locations.append(location)

                # Highest current article version with a valid
                # OA PDF wins for this PMCID.
                break

        status = (
            "success"
            if successful_lookup_count > 0
            else "failed"
        )

        attempt = ResolverAttempt(
            resolver="pmc_aws",
            status=status,
            elapsed_seconds=(
                time.perf_counter()
                - started
            ),
            message=(
                f"pmcids={len(pmcids)};"
                f" lookups={successful_lookup_count};"
                f" locations={len(locations)};"
                f" failures={failure_count}"
            ),
        )

        if (
            self.policy.resolver_delay_seconds
            > 0
        ):
            time.sleep(
                self.policy
                .resolver_delay_seconds
            )

        return PmcAwsAccessProbe(
            attempt=attempt,
            locations=locations,
        )
