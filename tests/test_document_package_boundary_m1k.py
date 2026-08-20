from __future__ import annotations

import ast
import json
from dataclasses import fields
from pathlib import Path

import pytest

from pipeline_core.document_config import (
    DocumentConfig,
    DocumentSelection,
    FigureProcessingConfig,
)
from pipeline_core.document_package import (
    DocumentPackage,
    SelectedSource,
    extract_supplementary_references,
    load_document_package,
    select_document_sources,
)


def _document_config(
    tmp_path: Path,
    *,
    selection: DocumentSelection,
    metadata_path: Path | None = None,
) -> DocumentConfig:
    return DocumentConfig(
        document_id="main",
        role="main",
        package_dir=tmp_path,
        markdown_path=(
            tmp_path / "main.md"
        ),
        metadata_path=metadata_path,
        selection=selection,
        figure_processing=(
            FigureProcessingConfig()
        ),
    )


def _package(
    tmp_path: Path,
    *,
    markdown: str,
) -> DocumentPackage:
    return DocumentPackage(
        paper_id="paper_a",
        document_id="main",
        role="main",
        package_dir=tmp_path,
        markdown_path=(
            tmp_path / "main.md"
        ),
        metadata_path=None,
        markdown=markdown,
        metadata=None,
        assets=(),
    )


def test_document_package_dataclass_surface_is_frozen():
    assert [
        field.name
        for field in fields(
            DocumentPackage
        )
    ] == [
        "paper_id",
        "document_id",
        "role",
        "package_dir",
        "markdown_path",
        "metadata_path",
        "markdown",
        "metadata",
        "assets",
    ]

    assert [
        field.name
        for field in fields(
            SelectedSource
        )
    ] == [
        "paper_id",
        "document_id",
        "document_role",
        "selection_id",
        "section",
        "text",
    ]


def test_load_document_package_preserves_source_and_metadata(
    tmp_path: Path,
):
    markdown_path = (
        tmp_path / "main.md"
    )
    metadata_path = (
        tmp_path / "metadata.json"
    )

    markdown_path.write_text(
        "# Results\n\nObserved result.\n",
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(
            {
                "title": "Paper A",
                "year": 2026,
            }
        ),
        encoding="utf-8",
    )

    config = _document_config(
        tmp_path,
        selection=DocumentSelection(
            mode="whole_document"
        ),
        metadata_path=metadata_path,
    )

    package = load_document_package(
        paper_id="paper_a",
        config=config,
    )

    assert package.paper_id == "paper_a"
    assert package.document_id == "main"
    assert package.role == "main"

    assert (
        package.package_dir
        == tmp_path
    )

    assert (
        package.markdown_path
        == markdown_path
    )

    assert (
        package.metadata_path
        == metadata_path
    )

    assert (
        package.markdown
        == "# Results\n\nObserved result.\n"
    )

    assert package.metadata == {
        "title": "Paper A",
        "year": 2026,
    }

    assert isinstance(
        package.assets,
        tuple,
    )


def test_load_document_package_wraps_non_mapping_metadata(
    tmp_path: Path,
):
    markdown_path = (
        tmp_path / "main.md"
    )
    metadata_path = (
        tmp_path / "metadata.json"
    )

    markdown_path.write_text(
        "# Main\n\nText.\n",
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(
            ["one", "two"]
        ),
        encoding="utf-8",
    )

    config = _document_config(
        tmp_path,
        selection=DocumentSelection(
            mode="whole_document"
        ),
        metadata_path=metadata_path,
    )

    package = load_document_package(
        paper_id="paper_a",
        config=config,
    )

    assert package.metadata == {
        "value": [
            "one",
            "two",
        ]
    }


def test_supplementary_reference_normalization_and_range_expansion():
    references = (
        extract_supplementary_references(
            (
                (
                    "See Supplementary Figures "
                    "S1-S3 for details."
                ),
                (
                    "Supporting Information "
                    "Table 2 and 3 reports "
                    "the values. Fig. S5 "
                    "confirms this."
                ),
            )
        )
    )

    assert references == (
        "Supplementary Fig S1",
        "Supplementary Fig S2",
        "Supplementary Fig S3",
        "Supplementary Fig S5",
        "Supplementary Table 2",
        "Supplementary Table 3",
    )


def test_select_document_sources_whole_document(
    tmp_path: Path,
):
    package = _package(
        tmp_path,
        markdown=(
            "\n# Main\n\n"
            "Whole document text.\n"
        ),
    )

    config = _document_config(
        tmp_path,
        selection=DocumentSelection(
            mode="whole_document"
        ),
    )

    selected = select_document_sources(
        package=package,
        config=config,
    )

    assert len(selected) == 1

    source = selected[0]

    assert source.paper_id == "paper_a"
    assert source.document_id == "main"
    assert source.document_role == "main"

    assert (
        source.selection_id
        == "whole_document"
    )

    assert (
        source.section
        == "[main] whole document"
    )

    assert source.text == (
        "# Main\n\n"
        "Whole document text."
    )


def test_select_document_sources_sections(
    tmp_path: Path,
):
    package = _package(
        tmp_path,
        markdown=(
            "# Introduction\n\n"
            "Intro.\n\n"
            "## Results\n\n"
            "Result body.\n\n"
            "### Detail\n\n"
            "Nested detail.\n\n"
            "## Methods\n\n"
            "Method body.\n"
        ),
    )

    config = _document_config(
        tmp_path,
        selection=DocumentSelection(
            mode="sections",
            headings=(
                "## Results",
                "## Methods",
            ),
        ),
    )

    selected = select_document_sources(
        package=package,
        config=config,
    )

    assert [
        source.selection_id
        for source in selected
    ] == [
        "section:0",
        "section:1",
    ]

    assert [
        source.section
        for source in selected
    ] == [
        "## Results",
        "## Methods",
    ]

    assert (
        "Result body."
        in selected[0].text
    )

    assert (
        "Nested detail."
        in selected[0].text
    )

    assert (
        "Method body."
        not in selected[0].text
    )

    assert (
        "Method body."
        in selected[1].text
    )


def test_select_referenced_blocks_and_fallback_contract(
    tmp_path: Path,
):
    markdown = (
        "# Supporting Information\n\n"
        "Overview.\n\n"
        "## Figure S3\n\n"
        "Target S3 evidence.\n\n"
        "## Figure S4\n\n"
        "Other evidence.\n"
    )

    package = _package(
        tmp_path,
        markdown=markdown,
    )

    referenced_config = (
        _document_config(
            tmp_path,
            selection=DocumentSelection(
                mode="referenced_blocks",
                fallback="error",
            ),
        )
    )

    selected = select_document_sources(
        package=package,
        config=referenced_config,
        supplementary_references=(
            "Supplementary Fig S3",
        ),
    )

    assert len(selected) == 1

    assert (
        selected[0].selection_id
        == "referenced:0"
    )

    assert (
        selected[0].section
        == "Supplementary Fig S3"
    )

    assert (
        "Target S3 evidence."
        in selected[0].text
    )

    assert (
        "Other evidence."
        not in selected[0].text
    )

    whole_fallback = (
        _document_config(
            tmp_path,
            selection=DocumentSelection(
                mode="referenced_blocks",
                fallback="whole_document",
            ),
        )
    )

    selected = select_document_sources(
        package=package,
        config=whole_fallback,
        supplementary_references=(
            "Supplementary Fig S99",
        ),
    )

    assert len(selected) == 1

    assert (
        selected[0].selection_id
        == "fallback:whole_document"
    )

    assert (
        selected[0].text
        == markdown.strip()
    )

    skip_fallback = (
        _document_config(
            tmp_path,
            selection=DocumentSelection(
                mode="referenced_blocks",
                fallback="skip",
            ),
        )
    )

    assert (
        select_document_sources(
            package=package,
            config=skip_fallback,
            supplementary_references=(
                "Supplementary Fig S99",
            ),
        )
        == []
    )

    error_fallback = (
        _document_config(
            tmp_path,
            selection=DocumentSelection(
                mode="referenced_blocks",
                fallback="error",
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "No referenced blocks found"
        ),
    ):
        select_document_sources(
            package=package,
            config=error_fallback,
            supplementary_references=(
                "Supplementary Fig S99",
            ),
        )


def test_shared_document_package_dependency_boundary():
    path = Path(
        "pipeline_core/document_package.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if (
            isinstance(
                node,
                ast.ImportFrom,
            )
            and node.module is not None
        )
    }

    dac_modules = {
        module
        for module in imported_modules
        if module.startswith(
            "dac_her."
        )
    }

    assert dac_modules == set()

    assert {
        "pipeline_core.asset_index",
        "pipeline_core.document_config",
        "pipeline_core.markdown",
    }.issubset(
        imported_modules
    )
