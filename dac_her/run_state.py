from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dac_her.config import PaperConfig, paper_config_fingerprint_payload
from dac_her.extraction_policy import ExtractionPolicy
from dac_her.prompts import PROMPT_VERSION, SYSTEM_PROMPT


RUN_STATE_VERSION = "semantic-si-assets-run-v4"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def document_source_fingerprints(paper: PaperConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    allowed_suffixes = {
        ".md", ".json", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"
    }
    for document in paper.documents:
        files: list[dict[str, str]] = []
        if document.package_dir.exists():
            for path in sorted(document.package_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                    continue
                files.append({
                    "relative_path": str(path.relative_to(document.package_dir)),
                    "sha256": sha256_file(path),
                })
        records.append({
            "document_id": document.document_id,
            "role": document.role,
            "markdown_sha256": sha256_file(document.markdown_path),
            "metadata_sha256": (
                sha256_file(document.metadata_path)
                if document.metadata_path is not None and document.metadata_path.exists()
                else None
            ),
            "package_files": files,
        })
    return records


def compute_run_metadata(
    *,
    project_root: str | Path,
    paper: PaperConfig,
    policy: ExtractionPolicy,
    model: str,
    provider: str | None,
    schemas_path: str | Path,
    chunking_path: str | Path,
    runtime_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    schemas_path = Path(schemas_path).resolve()
    chunking_path = Path(chunking_path).resolve()

    metadata: dict[str, Any] = {
        "run_state_version": RUN_STATE_VERSION,
        "paper": paper_config_fingerprint_payload(paper),
        "document_sources": document_source_fingerprints(paper),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "schema_sha256": sha256_file(schemas_path),
        "chunking_sha256": sha256_file(chunking_path),
        "policy": asdict(policy),
        "model": model,
        "provider": provider,
        "runtime_options": runtime_options or {},
        "vocabularies": [
            {
                "relative_path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
            }
            for path in sorted(
                (project_root / "configs" / "vocabularies").glob("*.yaml")
            )
        ],
        "project_root": str(project_root),
    }

    fingerprint = sha256_text(canonical_json(metadata))
    metadata["run_fingerprint"] = fingerprint
    metadata["run_id"] = fingerprint[:16]
    metadata["created_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    return metadata


def paper_output_root(
    project_root: str | Path,
    paper_id: str,
) -> Path:
    return (
        Path(project_root).resolve()
        / "data_dac"
        / "extracted"
        / paper_id
    )


def run_directory(
    project_root: str | Path,
    paper_id: str,
    run_id: str,
) -> Path:
    return paper_output_root(project_root, paper_id) / "runs" / run_id


def write_json(
    path: str | Path,
    payload: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_latest_run_pointer(
    *,
    project_root: str | Path,
    paper_id: str,
    run_metadata: dict[str, Any],
) -> Path:
    root = paper_output_root(project_root, paper_id)
    return write_json(
        root / "latest_run.json",
        {
            "paper_id": paper_id,
            "run_id": run_metadata["run_id"],
            "run_fingerprint": run_metadata["run_fingerprint"],
            "run_directory": str(
                run_directory(
                    project_root,
                    paper_id,
                    str(run_metadata["run_id"]),
                )
            ),
            "updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    )


def resolve_run_directory(
    *,
    project_root: str | Path,
    paper_id: str,
    run_id: str | None,
) -> Path:
    if run_id:
        path = run_directory(project_root, paper_id, run_id)
    else:
        pointer_path = (
            paper_output_root(project_root, paper_id)
            / "latest_run.json"
        )
        if not pointer_path.exists():
            raise FileNotFoundError(
                "No latest run pointer found for "
                f"{paper_id!r}: {pointer_path}"
            )
        pointer = read_json(pointer_path)
        path = Path(pointer["run_directory"])

    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")

    return path.resolve()
