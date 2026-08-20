from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from pipeline_core.literature.acquisition.access_contracts import SourceArtifact
from pipeline_core.literature.acquisition.supplementary_contracts import SupplementaryCandidate, SupplementaryDiscoveryPolicy


_ALLOWED_MIME = {
    "application/pdf",
    "application/zip",
    "application/x-zip-compressed",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/csv",
    "text/plain",
    "application/octet-stream",
}

_EXTENSION_BY_MIME = {
    "application/pdf": ".pdf",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/csv": ".csv",
    "text/plain": ".txt",
}


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(value) for value in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _safe_work_dir(work_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", work_id).strip("_")
    digest = hashlib.sha256(work_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:48]}__{digest}"


def _extension_from_url(url: str) -> str:
    suffix = Path(unquote(urlparse(url).path)).suffix.casefold()
    if suffix in {
        ".pdf", ".zip", ".xlsx", ".xls", ".csv",
        ".docx", ".doc", ".txt", ".pptx",
    }:
        return suffix
    return ""


def _detect_magic(prefix: bytes) -> str | None:
    if prefix.startswith(b"%PDF-"):
        return "pdf"
    if prefix.startswith(b"PK\x03\x04"):
        return "zip"
    if prefix.startswith(b"\xD0\xCF\x11\xE0"):
        return "ole"
    return None


def _validated_extension(
    *,
    url: str,
    content_type: str,
    prefix: bytes,
) -> str:
    mime = content_type.split(";", 1)[0].strip().casefold()
    if mime == "text/html":
        raise RuntimeError("supplementary_candidate_returned_html")
    if mime and mime not in _ALLOWED_MIME:
        raise RuntimeError(f"unsupported_supplementary_content_type:{mime}")

    url_ext = _extension_from_url(url)
    magic = _detect_magic(prefix)

    if url_ext == ".pdf" and magic != "pdf":
        raise RuntimeError("pdf_extension_without_pdf_magic")
    if mime == "application/pdf" and magic != "pdf":
        raise RuntimeError("pdf_mime_without_pdf_magic")
    if url_ext in {".zip", ".xlsx", ".docx", ".pptx"} and magic != "zip":
        raise RuntimeError("zip_family_extension_without_zip_magic")
    if mime in {
        "application/zip",
        "application/x-zip-compressed",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    } and magic != "zip":
        raise RuntimeError("zip_family_mime_without_zip_magic")

    if url_ext:
        return url_ext
    if mime in _EXTENSION_BY_MIME:
        return _EXTENSION_BY_MIME[mime]
    if magic == "pdf":
        return ".pdf"
    if magic == "zip":
        return ".zip"
    if magic == "ole":
        return ".bin"
    raise RuntimeError("cannot_validate_supplementary_artifact_type")


@dataclass(frozen=True)
class SupplementaryArtifactDownloader:
    policy: SupplementaryDiscoveryPolicy

    def acquire(
        self,
        *,
        candidate: SupplementaryCandidate,
        output_root: Path,
    ) -> SourceArtifact:
        eligible = (
            candidate.kind == "direct_file"
            and candidate.automatic_download_eligible
            and candidate.url
            and (
                candidate.confidence == "high"
                or (
                    candidate.confidence == "medium"
                    and self.policy.allow_medium_confidence_direct_files
                )
            )
            and self.policy.auto_download_high_confidence_direct_files
        )
        if not eligible:
            return SourceArtifact(
                artifact_id=_stable_id(
                    "source_artifact",
                    candidate.work_id,
                    "supporting_information",
                    candidate.candidate_id,
                ),
                work_id=candidate.work_id,
                role="supporting_information",
                status="not_attempted",
                source_url=candidate.url,
                acquisition_method=None,
                positive_evidence_promotion_performed=False,
            )

        url = str(candidate.url)
        work_dir = (
            output_root
            / "artifacts"
            / _safe_work_dir(candidate.work_id)
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        stem = hashlib.sha256(
            candidate.candidate_id.encode("utf-8")
        ).hexdigest()[:14]
        marker_path = work_dir / f"supplementary_{stem}.artifact.json"

        if marker_path.exists():
            marker = __import__("json").loads(
                marker_path.read_text(encoding="utf-8")
            )
            local_path = Path(marker["local_path"])
            if local_path.exists():
                digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
                if digest != marker["sha256"]:
                    raise RuntimeError(
                        f"Existing supplementary artifact hash drift: {local_path}"
                    )
                artifact_data = dict(marker["artifact"])
                artifact_data["acquisition_method"] = (
                    "resume_existing_verified_supplement"
                )
                return SourceArtifact.model_validate(artifact_data)

        last_error: Exception | None = None
        for attempt in range(self.policy.retries + 1):
            tmp_path: Path | None = None
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": self.policy.user_agent,
                        "Accept": (
                            "application/pdf,application/zip,"
                            "application/octet-stream,text/csv,text/plain;q=0.8,*/*;q=0.1"
                        ),
                    },
                )
                with urlopen(
                    request,
                    timeout=self.policy.request_timeout_seconds,
                ) as response:
                    resolved = response.geturl()
                    content_type = str(
                        response.headers.get("Content-Type") or ""
                    ).split(";", 1)[0].strip().casefold()

                    chunks: list[bytes] = []
                    total = 0
                    prefix = b""
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        if not prefix:
                            prefix = chunk[:16]
                        total += len(chunk)
                        if total > self.policy.max_artifact_bytes:
                            raise RuntimeError(
                                "supplementary_artifact_exceeds_max_bytes:"
                                f"{self.policy.max_artifact_bytes}"
                            )
                        chunks.append(chunk)
                    if total == 0:
                        raise RuntimeError("empty_supplementary_download")

                    extension = _validated_extension(
                        url=resolved,
                        content_type=content_type,
                        prefix=prefix,
                    )
                    final_path = work_dir / f"supplementary_{stem}{extension}"
                    tmp_path = final_path.with_suffix(final_path.suffix + ".partial")
                    digest = hashlib.sha256()
                    with tmp_path.open("wb") as handle:
                        for chunk in chunks:
                            digest.update(chunk)
                            handle.write(chunk)
                    os.replace(tmp_path, final_path)

                    artifact = SourceArtifact(
                        artifact_id=_stable_id(
                            "source_artifact",
                            candidate.work_id,
                            "supporting_information",
                            candidate.candidate_id,
                        ),
                        work_id=candidate.work_id,
                        role="supporting_information",
                        status="downloaded",
                        source_url=url,
                        resolved_url=resolved,
                        local_path=str(final_path),
                        sha256=digest.hexdigest(),
                        byte_count=total,
                        content_type=content_type or None,
                        acquired_at_utc=(
                            datetime.now(timezone.utc)
                            .replace(microsecond=0)
                            .isoformat()
                        ),
                        acquisition_method="public_supplementary_direct_http",
                        positive_evidence_promotion_performed=False,
                    )
                    marker = {
                        "candidate_id": candidate.candidate_id,
                        "local_path": str(final_path),
                        "sha256": artifact.sha256,
                        "artifact": artifact.model_dump(mode="json"),
                    }
                    marker_path.write_text(
                        __import__("json").dumps(
                            marker,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    return artifact
            except Exception as exc:
                last_error = exc
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
                if attempt < self.policy.retries:
                    time.sleep(
                        self.policy.retry_backoff_seconds * (2**attempt)
                    )

        return SourceArtifact(
            artifact_id=_stable_id(
                "source_artifact",
                candidate.work_id,
                "supporting_information",
                candidate.candidate_id,
            ),
            work_id=candidate.work_id,
            role="supporting_information",
            status="download_failed",
            source_url=url,
            acquisition_method="public_supplementary_direct_http",
            error=(
                f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "unknown_supplementary_download_failure"
            ),
            positive_evidence_promotion_performed=False,
        )
