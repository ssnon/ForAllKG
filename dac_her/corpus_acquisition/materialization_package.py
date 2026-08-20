from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from dac_her.corpus_acquisition.access_contracts import SourceArtifact
from dac_her.corpus_acquisition.materialization_contracts import (
    MaterializationPolicy,
    MaterializedDocument,
)
from dac_her.corpus_acquisition.materializers import (
    copy_materializer_assets,
    materializer_for,
    sha256_file,
)
from pipeline_core.literature.catalog_contracts import CatalogWork


def stable_paper_id(
    *,
    prefix: str,
    work_id: str,
) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", prefix).strip("_")
    if not cleaned:
        raise ValueError("paper ID prefix must contain alphanumeric content")
    digest = hashlib.sha256(work_id.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned}_{digest}"


def _resolve_source_path(
    value: str,
    *,
    project_root: Path,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _project_path(
    path: Path,
    *,
    project_root: Path,
) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def materialize_artifact(
    *,
    materialization_id: str,
    paper_id: str,
    work: CatalogWork,
    document_id: str,
    role: str,
    artifact: SourceArtifact,
    package_dir: Path,
    policy: MaterializationPolicy,
    project_root: Path,
) -> MaterializedDocument:
    if artifact.status != "downloaded" or not artifact.local_path:
        return MaterializedDocument(
            materialization_id=materialization_id,
            paper_id=paper_id,
            work_id=work.work_id,
            document_id=document_id,
            role=role,
            source_artifact_id=artifact.artifact_id,
            source_artifact_sha256=artifact.sha256,
            source_path=str(artifact.local_path or ""),
            source_extension="",
            status="skipped",
            materializer="none",
            error="source_artifact_not_downloaded",
        )

    source_path = _resolve_source_path(
        artifact.local_path,
        project_root=project_root,
    )
    if not source_path.exists():
        return MaterializedDocument(
            materialization_id=materialization_id,
            paper_id=paper_id,
            work_id=work.work_id,
            document_id=document_id,
            role=role,
            source_artifact_id=artifact.artifact_id,
            source_artifact_sha256=artifact.sha256,
            source_path=str(source_path),
            source_extension=source_path.suffix.casefold(),
            status="failed",
            materializer="none",
            error="source_artifact_file_missing",
        )

    observed_sha = sha256_file(source_path)
    if artifact.sha256 and observed_sha != artifact.sha256:
        raise RuntimeError(
            f"Source artifact SHA drift for {artifact.artifact_id}: "
            f"{observed_sha} != {artifact.sha256}"
        )

    adapter = materializer_for(source_path, policy)
    if adapter is None:
        return MaterializedDocument(
            materialization_id=materialization_id,
            paper_id=paper_id,
            work_id=work.work_id,
            document_id=document_id,
            role=role,
            source_artifact_id=artifact.artifact_id,
            source_artifact_sha256=observed_sha,
            source_path=str(source_path),
            source_extension=source_path.suffix.casefold(),
            status="unsupported",
            materializer="none",
            error=f"unsupported_source_extension:{source_path.suffix.casefold()}",
        )

    try:
        output = adapter.materialize(
            source_path=source_path,
            policy=policy,
        )
        package_dir.mkdir(parents=True, exist_ok=True)
        copied_asset_count = copy_materializer_assets(
            output=output,
            package_dir=package_dir,
        )
        markdown = output.markdown.strip() + "\n"
        markdown_path = package_dir / "normalized.md"
        _atomic_write_text(markdown_path, markdown)

        metadata = {
            "schema_version": "materialized-document-metadata-v1",
            "materialization_id": materialization_id,
            "paper_id": paper_id,
            "work_id": work.work_id,
            "document_id": document_id,
            "role": role,
            "bibliographic": {
                "title": work.title,
                "doi": work.doi,
                "year": work.year,
                "publication_date": work.publication_date,
                "authors": list(work.authors),
                "venue": work.venue,
                "providers": list(work.providers),
                "provider_ids": dict(work.provider_ids),
                "retrieval_query_ids": list(work.retrieval_query_ids),
                "retrieval_axis_ids": list(work.retrieval_axis_ids),
            },
            "source_artifact": {
                "artifact_id": artifact.artifact_id,
                "role": artifact.role,
                "source_url": artifact.source_url,
                "resolved_url": artifact.resolved_url,
                "local_path": str(source_path),
                "sha256": observed_sha,
                "byte_count": artifact.byte_count,
                "content_type": artifact.content_type,
                "license": artifact.license,
                "version": artifact.version,
                "host_type": artifact.host_type,
                "acquisition_method": artifact.acquisition_method,
            },
            "materializer": output.materializer,
            "markdown_sha256": hashlib.sha256(
                markdown.encode("utf-8")
            ).hexdigest(),
            "source_scientific_content_modified": False,
            "scientific_result_inferred": False,
            "positive_evidence_promotion_performed": False,
        }
        metadata_path = package_dir / "metadata.json"
        _atomic_write_text(
            metadata_path,
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
        )

        return MaterializedDocument(
            materialization_id=materialization_id,
            paper_id=paper_id,
            work_id=work.work_id,
            document_id=document_id,
            role=role,
            source_artifact_id=artifact.artifact_id,
            source_artifact_sha256=observed_sha,
            source_path=str(source_path),
            source_extension=source_path.suffix.casefold(),
            status="materialized",
            materializer=output.materializer,
            package_dir=_project_path(
                package_dir,
                project_root=project_root,
            ),
            markdown_path=_project_path(
                markdown_path,
                project_root=project_root,
            ),
            metadata_path=_project_path(
                metadata_path,
                project_root=project_root,
            ),
            markdown_sha256=hashlib.sha256(
                markdown.encode("utf-8")
            ).hexdigest(),
            markdown_char_count=len(markdown),
            copied_asset_count=copied_asset_count,
            source_scientific_content_modified=False,
            scientific_result_inferred=False,
            positive_evidence_promotion_performed=False,
        )
    except Exception as exc:
        return MaterializedDocument(
            materialization_id=materialization_id,
            paper_id=paper_id,
            work_id=work.work_id,
            document_id=document_id,
            role=role,
            source_artifact_id=artifact.artifact_id,
            source_artifact_sha256=observed_sha,
            source_path=str(source_path),
            source_extension=source_path.suffix.casefold(),
            status="failed",
            materializer=getattr(adapter, "materializer_id", type(adapter).__name__),
            error=f"{type(exc).__name__}: {exc}",
            source_scientific_content_modified=False,
            scientific_result_inferred=False,
            positive_evidence_promotion_performed=False,
        )


def generated_paper_config_entry(
    *,
    paper_id: str,
    documents: list[MaterializedDocument],
    policy: MaterializationPolicy,
) -> dict[str, Any] | None:
    materialized = [
        row for row in documents if row.status == "materialized"
    ]
    main = next(
        (
            row
            for row in materialized
            if row.role == "main"
        ),
        None,
    )
    if main is None:
        return None

    ordered = [main] + sorted(
        [
            row
            for row in materialized
            if row.role == "supporting_information"
        ],
        key=lambda row: row.document_id,
    )

    config_documents = []
    for row in ordered:
        if not row.package_dir:
            raise ValueError("Materialized document lacks package_dir")
        if row.role == "main":
            selection = {
                "mode": policy.main_selection_mode,
            }
        else:
            selection = {
                "mode": policy.si_selection_mode,
                "fallback": policy.si_fallback,
                "reference_scope": policy.si_reference_scope,
            }
        config_documents.append(
            {
                "document_id": row.document_id,
                "role": row.role,
                "package_dir": row.package_dir,
                "markdown_file": "normalized.md",
                "metadata_file": "metadata.json",
                "selection": selection,
                "figure_processing": {
                    "mode": policy.figure_processing_mode,
                    "vision_assets": [],
                },
            }
        )

    return {
        "enabled": True,
        "documents": config_documents,
        "resolution_file": None,
    }


def write_generated_config(
    *,
    config_path: Path,
    papers: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "version": 3,
        "papers": {
            key: papers[key]
            for key in sorted(papers)
        },
    }
    text = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
    )
    _atomic_write_text(config_path, text)


def write_extraction_plan(
    *,
    path: Path,
    paper_ids: list[str],
    generated_config_path: Path,
    domain_profile_id: str,
    data_root: str,
    project_root: Path,
) -> None:
    config_value = _project_path(
        generated_config_path,
        project_root=project_root,
    )
    lines = []
    for paper_id in paper_ids:
        lines.append(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "command": [
                        "python",
                        "-m",
                        "scripts.extract_paper",
                        "--paper-id",
                        paper_id,
                        "--config",
                        config_value,
                        "--domain-profile",
                        domain_profile_id,
                        "--data-root",
                        data_root,
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    _atomic_write_text(
        path,
        "\n".join(lines) + ("\n" if lines else ""),
    )
