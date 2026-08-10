from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Severity = Literal["info", "warning", "error"]
DocumentRole = Literal["main", "supporting_information"]


@dataclass(frozen=True)
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str | None = None
    size: int | None = None
    md5_checksum: str | None = None
    parent_id: str | None = None
    folder_path: str | None = None

    def fingerprint(self) -> str:
        if self.md5_checksum:
            return f"md5:{self.md5_checksum}"
        return "meta:" + "|".join(
            [
                self.file_id,
                self.modified_time or "",
                str(self.size or ""),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DriveFile":
        return cls(**value)


@dataclass(frozen=True)
class ArticleRow:
    title: str
    reason: str
    annotator: str
    date: str
    redundancy: str
    flag: str
    file_name: str
    si_existence: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestionIssue:
    code: str
    message: str
    severity: Severity = "warning"
    paper_id: str | None = None
    file_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveredPaper:
    paper_id: str
    article: ArticleRow
    main_file: DriveFile | None
    si_files: list[DriveFile] = field(default_factory=list)
    issues: list[IngestionIssue] = field(default_factory=list)

    @property
    def ready_for_download(self) -> bool:
        return self.main_file is not None and not any(
            issue.severity == "error" for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "article": self.article.to_dict(),
            "main_file": self.main_file.to_dict() if self.main_file else None,
            "si_files": [item.to_dict() for item in self.si_files],
            "issues": [item.to_dict() for item in self.issues],
            "ready_for_download": self.ready_for_download,
        }


@dataclass(frozen=True)
class MarkerResult:
    document_id: str
    document_role: DocumentRole
    input_pdf: str
    output_dir: str
    raw_markdown: str | None
    normalized_markdown: str | None
    marker_version: str
    return_code: int
    stdout_path: str | None = None
    stderr_path: str | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.return_code == 0
            and self.raw_markdown is not None
            and self.normalized_markdown is not None
            and self.error is None
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["succeeded"] = self.succeeded
        return result


@dataclass
class PaperRegistryEntry:
    paper_id: str
    title: str
    annotator: str
    source_file_name: str
    main_drive_file: dict[str, Any]
    si_drive_files: list[dict[str, Any]] = field(default_factory=list)
    local_main_pdf: str | None = None
    local_si_pdfs: list[str] = field(default_factory=list)
    marker_version: str | None = None
    main_markdown: str | None = None
    si_markdown: list[str] = field(default_factory=list)
    qc_status: str = "not_run"
    issues: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str | None = None

    @property
    def source_fingerprint(self) -> dict[str, Any]:
        return {
            "main": self.main_drive_file.get("md5_checksum")
            or self.main_drive_file.get("modified_time"),
            "si": [
                item.get("md5_checksum") or item.get("modified_time")
                for item in self.si_drive_files
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_fingerprint"] = self.source_fingerprint
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PaperRegistryEntry":
        cleaned = dict(value)
        cleaned.pop("source_fingerprint", None)
        return cls(**cleaned)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
