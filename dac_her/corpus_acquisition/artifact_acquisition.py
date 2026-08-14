from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dac_her.corpus_acquisition.access_contracts import (
    AccessLocation,
    AccessResolution,
    ArtifactDownloadAttempt,
    SourceAcquisitionPolicy,
    SourceArtifact,
)
from dac_her.corpus_acquisition.access_priority import (
    access_location_priority,
)
from dac_her.corpus_acquisition.access_recovery import (
    suppressed_download_urls,
)
from dac_her.literature_catalog_contracts import CatalogWork


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _safe_work_dir(work_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", work_id).strip("_")
    digest = hashlib.sha256(work_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:48]}__{digest}"


def _pdf_magic(prefix: bytes) -> bool:
    return prefix.startswith(b"%PDF-")


def _location_priority(row: AccessLocation) -> tuple:
    return access_location_priority(row)


def ordered_download_locations(
    resolution: AccessResolution,
) -> list[AccessLocation]:
    rows = []
    seen_urls: set[str] = set()
    for location in sorted(
        resolution.locations,
        key=_location_priority,
    ):
        if not location.automatic_download_eligible:
            continue
        url = str(
            location.url_for_pdf or location.url or ""
        ).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        rows.append(location)
    return rows


def _error_code(exc: Exception) -> str:
    message = str(exc)
    if isinstance(exc, HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, URLError):
        return "url_error"
    if "downloaded_content_is_not_pdf" in message:
        return "not_pdf"
    if "artifact_exceeds_max_bytes" in message:
        return "size_limit"
    if "empty_download" in message:
        return "empty_download"
    return type(exc).__name__.casefold()


@dataclass(frozen=True)
class MainArtifactDownloader:
    policy: SourceAcquisitionPolicy

    def acquire(
        self,
        *,
        work: CatalogWork,
        resolution: AccessResolution,
        output_root: Path,
    ) -> SourceArtifact:
        all_locations = ordered_download_locations(resolution)
        suppressed_urls = suppressed_download_urls(output_root, work.work_id)
        locations = [
            location
            for location in all_locations
            if str(location.url_for_pdf or location.url or "").strip()
            not in suppressed_urls
        ]
        suppressed_count = len(all_locations) - len(locations)

        if (
            not self.policy.auto_download_main
            or resolution.status != "resolved_direct_pdf"
            or not locations
        ):
            return SourceArtifact(
                artifact_id=_stable_id(
                    "source_artifact",
                    work.work_id,
                    "main",
                ),
                work_id=work.work_id,
                role="main",
                status="not_attempted",
                source_url=(
                    resolution.selected_download_url
                    or None
                ),
                acquisition_method=(
                    "access_recovery_suppressed_hard_failed_locations"
                    if suppressed_count and not locations
                    else None
                ),
                error=(
                    "all_direct_locations_suppressed_after_hard_failure"
                    if suppressed_count and not locations
                    else None
                ),
                attempted_location_count=0,
                positive_evidence_promotion_performed=False,
            )

        if not self.policy.try_all_direct_pdf_locations:
            locations = locations[:1]

        work_dir = (
            output_root
            / "artifacts"
            / _safe_work_dir(work.work_id)
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        final_path = work_dir / "main.pdf"

        # Existing verified PDF is resume-safe.
        if final_path.exists():
            digest = hashlib.sha256()
            byte_count = 0
            with final_path.open("rb") as handle:
                prefix = handle.read(5)
                if (
                    self.policy.require_pdf_magic
                    and not _pdf_magic(prefix)
                ):
                    raise RuntimeError(
                        f"Existing artifact is not a PDF: {final_path}"
                    )
                digest.update(prefix)
                byte_count += len(prefix)
                for chunk in iter(
                    lambda: handle.read(1024 * 1024),
                    b"",
                ):
                    digest.update(chunk)
                    byte_count += len(chunk)
            return SourceArtifact(
                artifact_id=_stable_id(
                    "source_artifact",
                    work.work_id,
                    "main",
                ),
                work_id=work.work_id,
                role="main",
                status="downloaded",
                source_url=(
                    resolution.selected_download_url
                    or None
                ),
                resolved_url=(
                    resolution.selected_download_url
                    or None
                ),
                local_path=str(final_path),
                sha256=digest.hexdigest(),
                byte_count=byte_count,
                content_type="application/pdf",
                acquired_at_utc=None,
                acquisition_method=(
                    "resume_existing_verified_pdf"
                ),
                selected_location_id=(
                    resolution.selected_location_id
                ),
                attempted_location_count=0,
                download_attempts=[],
                positive_evidence_promotion_performed=False,
            )

        attempts: list[ArtifactDownloadAttempt] = []
        last_error: Exception | None = None

        for location_index, location in enumerate(
            locations,
            start=1,
        ):
            url = str(
                location.url_for_pdf
                or location.url
                or ""
            ).strip()
            if not url:
                continue

            # Per-location retries happen before falling back to the next
            # public direct-PDF location.
            for retry_index in range(
                self.policy.retries + 1
            ):
                started = time.perf_counter()
                tmp_path = (
                    work_dir
                    / (
                        "main.pdf.partial."
                        f"{location_index}.{retry_index}"
                    )
                )
                try:
                    headers = {
                        "User-Agent": (
                            self.policy.download_user_agent
                        ),
                        "Accept": (
                            "application/pdf,"
                            "application/octet-stream;q=0.9,"
                            "*/*;q=0.1"
                        ),
                        "Accept-Language": "en-US,en;q=0.8",
                    }
                    if (
                        self.policy.send_landing_page_referer
                        and location.url_for_landing_page
                    ):
                        headers["Referer"] = (
                            location.url_for_landing_page
                        )

                    request = Request(
                        url,
                        headers=headers,
                    )
                    with urlopen(
                        request,
                        timeout=(
                            self.policy
                            .request_timeout_seconds
                        ),
                    ) as response:
                        resolved_url = response.geturl()
                        content_type = str(
                            response.headers.get(
                                "Content-Type"
                            )
                            or ""
                        ).split(";", 1)[0].strip().lower()

                        digest = hashlib.sha256()
                        byte_count = 0
                        prefix = b""
                        with tmp_path.open("wb") as handle:
                            while True:
                                chunk = response.read(
                                    1024 * 1024
                                )
                                if not chunk:
                                    break
                                if not prefix:
                                    prefix = chunk[:5]
                                byte_count += len(chunk)
                                if (
                                    byte_count
                                    > self.policy.max_artifact_bytes
                                ):
                                    raise RuntimeError(
                                        "artifact_exceeds_max_bytes:"
                                        f"{self.policy.max_artifact_bytes}"
                                    )
                                digest.update(chunk)
                                handle.write(chunk)

                        if byte_count == 0:
                            raise RuntimeError(
                                "empty_download"
                            )
                        if (
                            self.policy.require_pdf_magic
                            and not _pdf_magic(prefix)
                        ):
                            raise RuntimeError(
                                "downloaded_content_is_not_pdf"
                            )

                        os.replace(tmp_path, final_path)
                        attempts.append(
                            ArtifactDownloadAttempt(
                                location_id=(
                                    location.location_id
                                ),
                                url=url,
                                host=(
                                    urlparse(url).hostname
                                ),
                                status="success",
                                elapsed_seconds=(
                                    time.perf_counter()
                                    - started
                                ),
                                resolved_url=resolved_url,
                                content_type=(
                                    content_type or None
                                ),
                                byte_count=byte_count,
                            )
                        )
                        return SourceArtifact(
                            artifact_id=_stable_id(
                                "source_artifact",
                                work.work_id,
                                "main",
                            ),
                            work_id=work.work_id,
                            role="main",
                            status="downloaded",
                            source_url=url,
                            resolved_url=resolved_url,
                            local_path=str(final_path),
                            sha256=digest.hexdigest(),
                            byte_count=byte_count,
                            content_type=(
                                content_type
                                or "application/pdf"
                            ),
                            license=location.license,
                            version=location.version,
                            host_type=location.host_type,
                            acquired_at_utc=(
                                datetime.now(timezone.utc)
                                .replace(microsecond=0)
                                .isoformat()
                            ),
                            acquisition_method=(
                                "public_oa_direct_http_"
                                "multi_location_fallback_v1"
                            ),
                            selected_location_id=(
                                location.location_id
                            ),
                            attempted_location_count=(
                                len(attempts)
                            ),
                            download_attempts=attempts,
                            positive_evidence_promotion_performed=False,
                        )
                except Exception as exc:
                    last_error = exc
                    if tmp_path.exists():
                        tmp_path.unlink()
                    attempts.append(
                        ArtifactDownloadAttempt(
                            location_id=(
                                location.location_id
                            ),
                            url=url,
                            host=urlparse(url).hostname,
                            status="failed",
                            elapsed_seconds=(
                                time.perf_counter()
                                - started
                            ),
                            error_code=_error_code(exc),
                            error=(
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        )
                    )
                    if (
                        retry_index
                        < self.policy.retries
                    ):
                        time.sleep(
                            self.policy
                            .retry_backoff_seconds
                            * (2**retry_index)
                        )

        return SourceArtifact(
            artifact_id=_stable_id(
                "source_artifact",
                work.work_id,
                "main",
            ),
            work_id=work.work_id,
            role="main",
            status="download_failed",
            source_url=(
                attempts[-1].url
                if attempts
                else resolution.selected_download_url
            ),
            acquisition_method=(
                "public_oa_direct_http_"
                "multi_location_fallback_v1"
            ),
            error=(
                f"{type(last_error).__name__}: "
                f"{last_error}"
                if last_error is not None
                else "all_direct_locations_failed"
            ),
            attempted_location_count=len(attempts),
            download_attempts=attempts,
            positive_evidence_promotion_performed=False,
        )
