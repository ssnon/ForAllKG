"""Shared document-source provenance helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_core.corpus.extraction.document_config import PaperConfig
from pipeline_core.runtime.serialization_primitives import sha256_bytes




def sha256_file(path: str | Path) -> str:
    path = Path(path)
    return sha256_bytes(path.read_bytes())


def document_source_fingerprints(
    paper: PaperConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    allowed_suffixes = {
        ".md",
        ".json",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
    }
    for document in paper.documents:
        files: list[dict[str, str]] = []
        if document.package_dir.exists():
            for path in sorted(
                document.package_dir.rglob("*")
            ):
                if (
                    not path.is_file()
                    or path.suffix.lower()
                    not in allowed_suffixes
                ):
                    continue

                files.append(
                    {
                        "relative_path": str(
                            path.relative_to(
                                document.package_dir
                            )
                        ),
                        "sha256": sha256_file(
                            path
                        ),
                    }
                )

        records.append(
            {
                "document_id": (
                    document.document_id
                ),
                "role": document.role,
                "markdown_sha256": (
                    sha256_file(
                        document.markdown_path
                    )
                ),
                "metadata_sha256": (
                    sha256_file(
                        document.metadata_path
                    )
                    if (
                        document.metadata_path
                        is not None
                        and document.metadata_path.exists()
                    )
                    else None
                ),
                "package_files": files,
            }
        )

    return records
