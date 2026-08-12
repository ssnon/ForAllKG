from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import yaml

from .selection import SelectedLiterature


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def extraction_paper_id(literature_paper_id: str) -> str:
    """Filesystem-safe stable paper ID for downstream GraphAgents stages."""
    suffix = literature_paper_id.split(":", 1)[-1]
    safe = _SAFE_ID_RE.sub("_", suffix).strip("._-")
    if not safe:
        raise ValueError(f"cannot derive extraction paper ID from {literature_paper_id!r}")
    return f"broad_{safe}"


def build_abstract_packages(
    selected: Iterable[SelectedLiterature],
    *,
    output_dir: str | Path,
    project_root: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    root = Path(project_root).resolve()
    documents_root = output / "documents"
    documents_root.mkdir(parents=True, exist_ok=True)
    papers_yaml_path = output / "papers.yaml"
    manifest_path = output / "package_manifest.json"

    paper_configs: dict[str, dict] = {}
    manifest_rows: list[dict] = []

    for item in sorted(selected, key=lambda row: row.record.paper_id):
        record = item.record
        if not record.abstract:
            raise ValueError(f"selected paper lacks abstract: {record.paper_id}")
        extraction_id = extraction_paper_id(record.paper_id)
        package_dir = documents_root / extraction_id
        package_dir.mkdir(parents=True, exist_ok=True)

        markdown_path = package_dir / "main.md"
        metadata_path = package_dir / "metadata.json"
        markdown_path.write_text(
            f"# {record.title.strip()}\n\n## Abstract\n\n{record.abstract.strip()}\n",
            encoding="utf-8",
        )
        metadata = {
            "schema_version": "graphagentsdac-broad-abstract-package-v01",
            "paper_id": extraction_id,
            "literature_paper_id": record.paper_id,
            "source_depth": "abstract",
            "title": record.title,
            "doi": record.doi,
            "year": record.year,
            "venue": record.venue,
            "provider_references": [row.to_dict() for row in record.provider_references],
            "discovery_queries": list(record.discovery_queries),
            "mechanism_buckets": list(record.mechanism_buckets),
            "assigned_bucket": item.assigned_bucket,
            "selection_mode": item.selection_mode,
            "selection_score": item.assessment.total_score,
            "bucket_scores": item.assessment.bucket_scores,
            "source_metadata": record.metadata,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        package_value = _project_path(package_dir, root)
        paper_configs[extraction_id] = {
            "enabled": True,
            "documents": [
                {
                    "document_id": "abstract",
                    "role": "main",
                    "package_dir": package_value,
                    "markdown_file": "main.md",
                    "metadata_file": "metadata.json",
                    "selection": {"mode": "whole_document"},
                    "figure_processing": {"mode": "none", "vision_assets": []},
                }
            ],
            "resolution_file": None,
        }
        manifest_rows.append(
            {
                "paper_id": extraction_id,
                "literature_paper_id": record.paper_id,
                "package_dir": package_value,
                "markdown_path": str(markdown_path),
                "metadata_path": str(metadata_path),
                "assigned_bucket": item.assigned_bucket,
            }
        )

    papers_yaml_path.write_text(
        yaml.safe_dump(
            {"version": 3, "papers": paper_configs},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "graphagentsdac-broad-abstract-package-manifest-v01",
                "paper_count": len(manifest_rows),
                "papers_yaml": str(papers_yaml_path),
                "papers": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return papers_yaml_path, manifest_path


def _project_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()
