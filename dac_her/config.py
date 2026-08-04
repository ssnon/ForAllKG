from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


DocumentRole = Literal[
    "main",
    "supporting_information",
    "other",
]
SelectionMode = Literal[
    "sections",
    "whole_document",
    "referenced_blocks",
]
ReferenceScope = Literal[
    "selected_main",
    "whole_main",
]
FigureProcessingMode = Literal[
    "none",
    "caption_first",
    "always_vision",
]


@dataclass(frozen=True)
class DocumentSelection:
    mode: SelectionMode
    headings: tuple[str, ...] = ()
    fallback: Literal["error", "whole_document", "skip"] = "error"
    reference_scope: ReferenceScope = "selected_main"


@dataclass(frozen=True)
class FigureProcessingConfig:
    mode: FigureProcessingMode = "caption_first"
    vision_assets: tuple[str, ...] = ()
    vision_model: str | None = None


@dataclass(frozen=True)
class DocumentConfig:
    document_id: str
    role: DocumentRole
    package_dir: Path
    markdown_path: Path
    metadata_path: Path | None
    selection: DocumentSelection
    figure_processing: FigureProcessingConfig


@dataclass(frozen=True)
class PaperConfig:
    paper_id: str
    documents: tuple[DocumentConfig, ...]
    enabled: bool = True
    resolution_file: Path | None = None

    @property
    def main_document(self) -> DocumentConfig:
        for document in self.documents:
            if document.role == "main":
                return document
        return self.documents[0]

    # Backward-compatible accessors used by older code.
    @property
    def markdown_path(self) -> Path:
        return self.main_document.markdown_path

    @property
    def sections(self) -> tuple[str, ...]:
        selection = self.main_document.selection
        return selection.headings if selection.mode == "sections" else ()


def _resolve_project_path(
    project_root: Path,
    value: str | Path,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _parse_selection(
    raw: Any,
    *,
    paper_id: str,
    document_id: str,
    legacy_sections: list[str] | None = None,
) -> DocumentSelection:
    if raw is None:
        if legacy_sections:
            return DocumentSelection(
                mode="sections",
                headings=tuple(section.strip() for section in legacy_sections),
            )
        return DocumentSelection(mode="whole_document")

    if not isinstance(raw, dict):
        raise ValueError(
            f"Paper {paper_id!r}, document {document_id!r}: "
            "selection must be a mapping."
        )

    mode = raw.get("mode", "sections")
    if mode not in {"sections", "whole_document", "referenced_blocks"}:
        raise ValueError(
            f"Unsupported selection mode for {paper_id}/{document_id}: {mode!r}"
        )

    raw_headings = raw.get("headings", [])
    if raw_headings is None:
        raw_headings = []
    if not isinstance(raw_headings, list):
        raise ValueError(
            f"Paper {paper_id!r}, document {document_id!r}: "
            "selection.headings must be a list."
        )

    headings: list[str] = []
    for heading in raw_headings:
        if not isinstance(heading, str) or not heading.strip():
            raise ValueError(
                f"Paper {paper_id!r}, document {document_id!r} "
                "contains an invalid heading."
            )
        headings.append(heading.strip())

    if mode == "sections" and not headings:
        raise ValueError(
            f"Paper {paper_id!r}, document {document_id!r}: "
            "sections mode requires at least one heading."
        )

    fallback = raw.get("fallback", "error")
    if fallback not in {"error", "whole_document", "skip"}:
        raise ValueError(
            f"Unsupported selection fallback for {paper_id}/{document_id}: "
            f"{fallback!r}"
        )

    reference_scope = raw.get("reference_scope", "selected_main")
    if reference_scope not in {"selected_main", "whole_main"}:
        raise ValueError(
            f"Unsupported reference_scope for {paper_id}/{document_id}: "
            f"{reference_scope!r}"
        )

    return DocumentSelection(
        mode=mode,
        headings=tuple(headings),
        fallback=fallback,
        reference_scope=reference_scope,
    )


def _parse_figure_processing(
    raw: Any,
    *,
    paper_id: str,
    document_id: str,
) -> FigureProcessingConfig:
    if raw is None:
        return FigureProcessingConfig()
    if not isinstance(raw, dict):
        raise ValueError(
            f"Paper {paper_id!r}, document {document_id!r}: "
            "figure_processing must be a mapping."
        )

    mode = raw.get("mode", "caption_first")
    if mode not in {"none", "caption_first", "always_vision"}:
        raise ValueError(
            f"Unsupported figure_processing mode for "
            f"{paper_id}/{document_id}: {mode!r}"
        )

    raw_assets = raw.get("vision_assets", []) or []
    if not isinstance(raw_assets, list):
        raise ValueError(
            f"Paper {paper_id!r}, document {document_id!r}: "
            "vision_assets must be a list."
        )
    assets = tuple(
        str(value).strip()
        for value in raw_assets
        if str(value).strip()
    )

    vision_model = raw.get("vision_model")
    if vision_model is not None and not isinstance(vision_model, str):
        raise ValueError("vision_model must be a string or null.")

    return FigureProcessingConfig(
        mode=mode,
        vision_assets=assets,
        vision_model=(vision_model.strip() if vision_model else None),
    )


def _parse_document(
    raw: dict[str, Any],
    *,
    paper_id: str,
    project_root: Path,
) -> DocumentConfig:
    document_id = raw.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError(
            f"Paper {paper_id!r}: every document requires document_id."
        )
    document_id = document_id.strip()

    role = raw.get("role", "main")
    if role not in {"main", "supporting_information", "other"}:
        raise ValueError(
            f"Unsupported document role for {paper_id}/{document_id}: {role!r}"
        )

    package_value = raw.get("package_dir")
    markdown_value = raw.get("markdown_path")
    markdown_file = raw.get("markdown_file")

    if package_value is not None:
        if not isinstance(package_value, str) or not package_value.strip():
            raise ValueError("package_dir must be a non-empty string.")
        package_dir = _resolve_project_path(project_root, package_value)
        if markdown_file is None:
            raise ValueError(
                f"Paper {paper_id!r}, document {document_id!r}: "
                "markdown_file is required when package_dir is used."
            )
        markdown_path = package_dir / str(markdown_file)
    elif markdown_value is not None:
        if not isinstance(markdown_value, str) or not markdown_value.strip():
            raise ValueError("markdown_path must be a non-empty string.")
        markdown_path = _resolve_project_path(project_root, markdown_value)
        package_dir = markdown_path.parent
    else:
        raise ValueError(
            f"Paper {paper_id!r}, document {document_id!r}: "
            "requires package_dir + markdown_file, or markdown_path."
        )

    metadata_value = raw.get("metadata_file")
    metadata_path: Path | None = None
    if metadata_value:
        metadata_path = package_dir / str(metadata_value)

    return DocumentConfig(
        document_id=document_id,
        role=role,
        package_dir=package_dir.resolve(),
        markdown_path=markdown_path.resolve(),
        metadata_path=(metadata_path.resolve() if metadata_path else None),
        selection=_parse_selection(
            raw.get("selection"),
            paper_id=paper_id,
            document_id=document_id,
        ),
        figure_processing=_parse_figure_processing(
            raw.get("figure_processing"),
            paper_id=paper_id,
            document_id=document_id,
        ),
    )


def load_paper_configs(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, PaperConfig]:
    config_path = Path(config_path).resolve()
    project_root = Path(project_root).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Paper config not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("papers.yaml must contain a mapping at the top level.")

    raw_papers = payload.get("papers")
    if not isinstance(raw_papers, dict):
        raise ValueError("papers.yaml must contain a 'papers' mapping.")

    configs: dict[str, PaperConfig] = {}

    for paper_id, raw in raw_papers.items():
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError("Every paper ID must be a non-empty string.")
        if not isinstance(raw, dict):
            raise ValueError(f"Paper entry {paper_id!r} must be a mapping.")

        documents: list[DocumentConfig] = []
        raw_documents = raw.get("documents")

        if raw_documents is not None:
            if not isinstance(raw_documents, list) or not raw_documents:
                raise ValueError(
                    f"Paper {paper_id!r}: documents must be a non-empty list."
                )
            for raw_document in raw_documents:
                if not isinstance(raw_document, dict):
                    raise ValueError(
                        f"Paper {paper_id!r}: each document must be a mapping."
                    )
                documents.append(
                    _parse_document(
                        raw_document,
                        paper_id=paper_id,
                        project_root=project_root,
                    )
                )
        else:
            # Milestone-1 compatibility.
            markdown_value = raw.get("markdown_path")
            raw_sections = raw.get("sections")
            if not isinstance(markdown_value, str) or not markdown_value.strip():
                raise ValueError(
                    f"Paper {paper_id!r} requires documents or markdown_path."
                )
            if not isinstance(raw_sections, list) or not raw_sections:
                raise ValueError(
                    f"Paper {paper_id!r} requires a non-empty sections list."
                )
            markdown_path = _resolve_project_path(project_root, markdown_value)
            documents.append(
                DocumentConfig(
                    document_id="main",
                    role="main",
                    package_dir=markdown_path.parent,
                    markdown_path=markdown_path,
                    metadata_path=None,
                    selection=_parse_selection(
                        None,
                        paper_id=paper_id,
                        document_id="main",
                        legacy_sections=raw_sections,
                    ),
                    figure_processing=FigureProcessingConfig(),
                )
            )

        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(
                f"Paper {paper_id!r} contains duplicate document IDs."
            )
        if not any(document.role == "main" for document in documents):
            raise ValueError(
                f"Paper {paper_id!r} requires at least one main document."
            )

        resolution_value = raw.get("resolution_file")
        resolution_file: Path | None = None
        if resolution_value is not None:
            if not isinstance(resolution_value, str):
                raise ValueError(
                    f"Paper {paper_id!r} resolution_file must be a string."
                )
            resolution_file = _resolve_project_path(
                project_root,
                resolution_value,
            )

        configs[paper_id] = PaperConfig(
            paper_id=paper_id,
            documents=tuple(documents),
            enabled=bool(raw.get("enabled", True)),
            resolution_file=resolution_file,
        )

    return configs


def get_paper_config(
    config_path: str | Path,
    *,
    project_root: str | Path,
    paper_id: str,
) -> PaperConfig:
    configs = load_paper_configs(config_path, project_root=project_root)
    try:
        config = configs[paper_id]
    except KeyError as error:
        available = ", ".join(sorted(configs))
        raise KeyError(
            f"Unknown paper_id {paper_id!r}. Available: {available}"
        ) from error

    if not config.enabled:
        raise ValueError(f"Paper {paper_id!r} is disabled in the config.")
    return config


def paper_config_fingerprint_payload(config: PaperConfig) -> dict[str, Any]:
    return {
        "paper_id": config.paper_id,
        "enabled": config.enabled,
        "resolution_file": (
            str(config.resolution_file)
            if config.resolution_file is not None
            else None
        ),
        "documents": [
            {
                "document_id": document.document_id,
                "role": document.role,
                "package_dir": str(document.package_dir),
                "markdown_path": str(document.markdown_path),
                "metadata_path": (
                    str(document.metadata_path)
                    if document.metadata_path is not None
                    else None
                ),
                "selection": {
                    "mode": document.selection.mode,
                    "headings": list(document.selection.headings),
                    "fallback": document.selection.fallback,
                    "reference_scope": document.selection.reference_scope,
                },
                "figure_processing": {
                    "mode": document.figure_processing.mode,
                    "vision_assets": list(document.figure_processing.vision_assets),
                    "vision_model": document.figure_processing.vision_model,
                },
            }
            for document in config.documents
        ],
    }
