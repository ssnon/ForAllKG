from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


FROZEN_SCHEMA = "graphagentsdac-frozen-corpus-v01"
GENERATED_SCHEMA = "graphagentsdac-generated-paper-config-v02"


class KGConfigAdapterError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(raw: str, project_root: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _frontmatter_body(text: str) -> str:
    """Return the Marker body from our ingestion normalized Markdown.

    Ingestion v0.1 creates normalized.md by prepending YAML frontmatter to the
    exact Marker Markdown (apart from an optional UTF-8 BOM).  KG extraction
    should consume the raw Marker Markdown, not that operational frontmatter.
    """
    text = text.lstrip("\ufeff")
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        raise KGConfigAdapterError(
            "normalized.md does not start with the expected ingestion frontmatter"
        )

    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise KGConfigAdapterError("Malformed ingestion frontmatter")

    closing_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise KGConfigAdapterError("Unterminated ingestion frontmatter")

    # The closing frontmatter delimiter already owns its line terminator.
    # Everything after that line is the raw Marker body verbatim.  Do not
    # strip an initial newline here: some Marker outputs legitimately start
    # with one, and removing it breaks exact raw/normalized matching.
    body = "".join(lines[closing_index + 1 :])
    return body.lstrip("\ufeff")


def _raw_markdown_candidates(normalized_path: Path) -> list[Path]:
    return sorted(
        path
        for path in normalized_path.parent.glob("*.md")
        if path.name != normalized_path.name
    )


def resolve_raw_marker_markdown(normalized_path: Path) -> Path:
    """Find the exact raw Marker Markdown underlying normalized.md.

    Matching by content is deliberate: file names differ across contributors,
    especially for SI files.  This also guarantees that the generated
    papers.yaml points to the document Marker actually produced.
    """
    if not normalized_path.is_file():
        raise KGConfigAdapterError(f"Normalized Markdown not found: {normalized_path}")

    normalized_text = normalized_path.read_text(encoding="utf-8", errors="strict")
    expected_body = _frontmatter_body(normalized_text)
    candidates = _raw_markdown_candidates(normalized_path)
    exact: list[Path] = []
    for candidate in candidates:
        candidate_text = candidate.read_text(encoding="utf-8", errors="strict").lstrip("\ufeff")
        if candidate_text == expected_body:
            exact.append(candidate)

    if len(exact) == 1:
        return exact[0]
    if not exact:
        raise KGConfigAdapterError(
            "Could not find raw Marker Markdown matching normalized.md in "
            f"{normalized_path.parent}. Candidates: {[p.name for p in candidates]!r}"
        )
    raise KGConfigAdapterError(
        "Multiple raw Marker Markdown files exactly match normalized.md in "
        f"{normalized_path.parent}: {[p.name for p in exact]!r}"
    )


def _verify_normalized_hash(
    normalized_path: Path,
    expected_sha256: str | None,
    *,
    label: str,
) -> str:
    actual = _sha256_file(normalized_path)
    if expected_sha256 and actual != expected_sha256:
        raise KGConfigAdapterError(
            f"Frozen Markdown changed for {label}: expected {expected_sha256}, got {actual}"
        )
    return actual


def _metadata_file_for(raw_markdown: Path) -> str | None:
    candidates = [
        raw_markdown.with_name(raw_markdown.stem + "_meta.json"),
    ]
    for path in candidates:
        if path.is_file():
            return path.name
    return None


def _document_config(
    *,
    raw_markdown: Path,
    project_root: Path,
    document_id: str,
    role: str,
    is_main: bool,
) -> dict[str, Any]:
    package_dir = raw_markdown.parent
    value: dict[str, Any] = {
        "document_id": document_id,
        "role": role,
        # package_dir + markdown_file preserves Marker-relative image paths and
        # exactly matches the repository's existing document-package contract.
        "package_dir": _project_path(package_dir, project_root),
        "markdown_file": raw_markdown.name,
    }
    metadata_file = _metadata_file_for(raw_markdown)
    if metadata_file:
        value["metadata_file"] = metadata_file

    if is_main:
        value["selection"] = {
            "mode": "whole_document",
        }
    else:
        value["selection"] = {
            "mode": "referenced_blocks",
            "fallback": "skip",
            "reference_scope": "whole_main",
        }

    value["figure_processing"] = {
        "mode": "caption_first",
        "vision_assets": [],
    }
    return value


@dataclass(frozen=True)
class GeneratedPaperConfig:
    papers_yaml: Path
    adapter_manifest: Path
    paper_ids: tuple[str, ...]


def build_generated_paper_config(
    frozen: dict[str, Any],
    *,
    frozen_manifest_path: str | Path,
    output_yaml: str | Path,
    project_root: str | Path = ".",
) -> GeneratedPaperConfig:
    if frozen.get("schema_version") != FROZEN_SCHEMA:
        raise KGConfigAdapterError(
            f"Expected schema_version={FROZEN_SCHEMA!r}, got {frozen.get('schema_version')!r}"
        )

    root = Path(project_root).resolve()
    output_path = Path(output_yaml)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_papers: dict[str, Any] = {}
    adapter_papers: list[dict[str, Any]] = []

    documents = list(frozen.get("documents") or [])
    if not documents:
        raise KGConfigAdapterError("Frozen corpus contains no documents")

    for paper in documents:
        paper_id = str(paper.get("paper_id") or "").strip()
        if not paper_id:
            raise KGConfigAdapterError("Frozen document has an empty paper_id")
        if paper_id in raw_papers:
            raise KGConfigAdapterError(f"Duplicate paper_id in frozen corpus: {paper_id}")

        expected = dict(paper.get("content_fingerprint") or {})
        main_normalized = _resolve(str(paper.get("main_markdown") or ""), root)
        main_normalized_hash = _verify_normalized_hash(
            main_normalized,
            expected.get("main_markdown_sha256"),
            label=f"{paper_id}/main",
        )
        main_raw = resolve_raw_marker_markdown(main_normalized)

        paper_documents: list[dict[str, Any]] = [
            _document_config(
                raw_markdown=main_raw,
                project_root=root,
                document_id="main",
                role="main",
                is_main=True,
            )
        ]
        adapter_documents: list[dict[str, Any]] = [
            {
                "document_id": "main",
                "role": "main",
                "normalized_markdown": _project_path(main_normalized, root),
                "normalized_sha256": main_normalized_hash,
                "raw_markdown": _project_path(main_raw, root),
                "raw_sha256": _sha256_file(main_raw),
                "package_dir": _project_path(main_raw.parent, root),
            }
        ]

        expected_si_rows = [
            row
            for row in (expected.get("supporting_markdown") or [])
            if isinstance(row, dict)
        ]
        expected_si_by_path = {
            str(row.get("path")): row.get("sha256")
            for row in expected_si_rows
        }

        for index, raw_path in enumerate(paper.get("supporting_markdown") or [], start=1):
            normalized = _resolve(str(raw_path), root)
            normalized_hash = _verify_normalized_hash(
                normalized,
                expected_si_by_path.get(str(raw_path)),
                label=f"{paper_id}/si{index}",
            )
            raw_marker = resolve_raw_marker_markdown(normalized)
            document_id = f"si{index}"
            paper_documents.append(
                _document_config(
                    raw_markdown=raw_marker,
                    project_root=root,
                    document_id=document_id,
                    role="supporting_information",
                    is_main=False,
                )
            )
            adapter_documents.append(
                {
                    "document_id": document_id,
                    "role": "supporting_information",
                    "normalized_markdown": _project_path(normalized, root),
                    "normalized_sha256": normalized_hash,
                    "raw_markdown": _project_path(raw_marker, root),
                    "raw_sha256": _sha256_file(raw_marker),
                    "package_dir": _project_path(raw_marker.parent, root),
                }
            )

        raw_papers[paper_id] = {
            "enabled": True,
            "documents": paper_documents,
            "resolution_file": None,
        }
        adapter_papers.append(
            {
                "paper_id": paper_id,
                "title": paper.get("title"),
                "annotator": paper.get("annotator"),
                "qc_status": paper.get("qc_status"),
                "alias_paper_ids": list(paper.get("alias_paper_ids") or []),
                "documents": adapter_documents,
            }
        )

    yaml_payload = {
        "version": 3,
        "papers": raw_papers,
    }
    output_path.write_text(
        yaml.safe_dump(
            yaml_payload,
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        ),
        encoding="utf-8",
    )

    adapter_manifest_path = output_path.with_suffix(".adapter.json")
    adapter_manifest = {
        "schema_version": GENERATED_SCHEMA,
        "created_at": _utc_now(),
        "corpus_id": str(frozen.get("corpus_id") or ""),
        "source_frozen_manifest": str(frozen_manifest_path),
        "generated_papers_yaml": _project_path(output_path, root),
        "paper_count": len(adapter_papers),
        "paper_ids": [row["paper_id"] for row in adapter_papers],
        "papers": adapter_papers,
    }
    adapter_manifest_path.write_text(
        json.dumps(adapter_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return GeneratedPaperConfig(
        papers_yaml=output_path,
        adapter_manifest=adapter_manifest_path,
        paper_ids=tuple(adapter_manifest["paper_ids"]),
    )


def load_and_generate_paper_config(
    frozen_manifest_path: str | Path,
    output_yaml: str | Path,
    *,
    project_root: str | Path = ".",
) -> GeneratedPaperConfig:
    path = Path(frozen_manifest_path)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    return build_generated_paper_config(
        frozen,
        frozen_manifest_path=str(frozen_manifest_path),
        output_yaml=output_yaml,
        project_root=project_root,
    )
