from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import yaml

import dac_her.config as config
import dac_her.paper_config as legacy_paper_config


def _write_yaml(
    path: Path,
    payload: dict,
) -> None:
    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_document_config_dataclass_surface_is_frozen():
    assert [
        field.name
        for field in fields(
            config.DocumentSelection
        )
    ] == [
        "mode",
        "headings",
        "fallback",
        "reference_scope",
    ]

    assert [
        field.name
        for field in fields(
            config.FigureProcessingConfig
        )
    ] == [
        "mode",
        "vision_assets",
        "vision_model",
    ]

    assert [
        field.name
        for field in fields(
            config.DocumentConfig
        )
    ] == [
        "document_id",
        "role",
        "package_dir",
        "markdown_path",
        "metadata_path",
        "selection",
        "figure_processing",
    ]

    assert [
        field.name
        for field in fields(
            config.PaperConfig
        )
    ] == [
        "paper_id",
        "documents",
        "enabled",
        "resolution_file",
    ]


def test_document_style_yaml_parse_and_fingerprint_payload(
    tmp_path: Path,
):
    package = tmp_path / "paper_package"

    config_path = (
        tmp_path / "papers.yaml"
    )

    _write_yaml(
        config_path,
        {
            "papers": {
                "paper_a": {
                    "enabled": True,
                    "resolution_file": (
                        "resolution.json"
                    ),
                    "documents": [
                        {
                            "document_id": "main",
                            "role": "main",
                            "package_dir": (
                                "paper_package"
                            ),
                            "markdown_file": (
                                "main.md"
                            ),
                            "metadata_file": (
                                "metadata.json"
                            ),
                            "selection": {
                                "mode": "sections",
                                "headings": [
                                    "Results",
                                    "Methods",
                                ],
                                "fallback": "error",
                                "reference_scope": (
                                    "whole_main"
                                ),
                            },
                            "figure_processing": {
                                "mode": (
                                    "always_vision"
                                ),
                                "vision_assets": [
                                    "figure_1.png",
                                ],
                                "vision_model": (
                                    "vision-test"
                                ),
                            },
                        },
                        {
                            "document_id": "si",
                            "role": (
                                "supporting_information"
                            ),
                            "package_dir": (
                                "paper_package"
                            ),
                            "markdown_file": (
                                "si.md"
                            ),
                        },
                    ],
                },
            },
        },
    )

    paper = config.get_paper_config(
        config_path,
        project_root=tmp_path,
        paper_id="paper_a",
    )

    assert paper.paper_id == "paper_a"
    assert paper.enabled is True

    assert paper.resolution_file == (
        tmp_path
        / "resolution.json"
    ).resolve()

    assert len(paper.documents) == 2

    main = paper.documents[0]
    si = paper.documents[1]

    assert main.document_id == "main"
    assert main.role == "main"

    assert main.package_dir == (
        package.resolve()
    )

    assert main.markdown_path == (
        package / "main.md"
    ).resolve()

    assert main.metadata_path == (
        package / "metadata.json"
    ).resolve()

    assert main.selection.mode == "sections"
    assert main.selection.headings == (
        "Results",
        "Methods",
    )
    assert (
        main.selection.fallback
        == "error"
    )
    assert (
        main.selection.reference_scope
        == "whole_main"
    )

    assert (
        main.figure_processing.mode
        == "always_vision"
    )
    assert (
        main.figure_processing.vision_assets
        == ("figure_1.png",)
    )
    assert (
        main.figure_processing.vision_model
        == "vision-test"
    )

    # Defaults on an ordinary SI document.
    assert si.selection.mode == (
        "whole_document"
    )
    assert si.selection.headings == ()
    assert (
        si.figure_processing.mode
        == "caption_first"
    )
    assert (
        si.figure_processing.vision_assets
        == ()
    )
    assert (
        si.figure_processing.vision_model
        is None
    )

    payload = (
        config.paper_config_fingerprint_payload(
            paper
        )
    )

    assert payload == {
        "paper_id": "paper_a",
        "enabled": True,
        "resolution_file": str(
            (
                tmp_path
                / "resolution.json"
            ).resolve()
        ),
        "documents": [
            {
                "document_id": "main",
                "role": "main",
                "package_dir": str(
                    package.resolve()
                ),
                "markdown_path": str(
                    (
                        package
                        / "main.md"
                    ).resolve()
                ),
                "metadata_path": str(
                    (
                        package
                        / "metadata.json"
                    ).resolve()
                ),
                "selection": {
                    "mode": "sections",
                    "headings": [
                        "Results",
                        "Methods",
                    ],
                    "fallback": "error",
                    "reference_scope": (
                        "whole_main"
                    ),
                },
                "figure_processing": {
                    "mode": (
                        "always_vision"
                    ),
                    "vision_assets": [
                        "figure_1.png",
                    ],
                    "vision_model": (
                        "vision-test"
                    ),
                },
            },
            {
                "document_id": "si",
                "role": (
                    "supporting_information"
                ),
                "package_dir": str(
                    package.resolve()
                ),
                "markdown_path": str(
                    (
                        package
                        / "si.md"
                    ).resolve()
                ),
                "metadata_path": None,
                "selection": {
                    "mode": (
                        "whole_document"
                    ),
                    "headings": [],
                    "fallback": "error",
                    "reference_scope": (
                        "selected_main"
                    ),
                },
                "figure_processing": {
                    "mode": (
                        "caption_first"
                    ),
                    "vision_assets": [],
                    "vision_model": None,
                },
            },
        ],
    }


def test_legacy_paper_yaml_shape_is_preserved(
    tmp_path: Path,
):
    config_path = (
        tmp_path / "papers.yaml"
    )

    _write_yaml(
        config_path,
        {
            "papers": {
                "legacy": {
                    "markdown_path": (
                        "legacy.md"
                    ),
                    "sections": [
                        "Results",
                        "Discussion",
                    ],
                },
            },
        },
    )

    paper = config.get_paper_config(
        config_path,
        project_root=tmp_path,
        paper_id="legacy",
    )

    assert len(paper.documents) == 1

    document = paper.main_document

    assert document.document_id == "main"
    assert document.role == "main"
    assert (
        document.markdown_path
        == (
            tmp_path
            / "legacy.md"
        ).resolve()
    )

    assert (
        document.selection.mode
        == "sections"
    )

    assert (
        document.selection.headings
        == (
            "Results",
            "Discussion",
        )
    )

    # Historical convenience accessors.
    assert (
        paper.markdown_path
        == document.markdown_path
    )
    assert paper.sections == (
        "Results",
        "Discussion",
    )


def test_paper_config_compatibility_module_preserves_identity():
    names = (
        "DocumentConfig",
        "DocumentSelection",
        "FigureProcessingConfig",
        "PaperConfig",
        "get_paper_config",
        "load_paper_configs",
        "paper_config_fingerprint_payload",
    )

    for name in names:
        assert (
            getattr(
                legacy_paper_config,
                name,
            )
            is getattr(
                config,
                name,
            )
        )


def test_current_shared_core_still_depends_on_dac_config():
    path = Path(
        "pipeline_core/"
        "strict_bridge_corpus_pipeline.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
    }

    # M1k.0 characterization:
    # this is the architecture dependency
    # that M1k config extraction will remove.
    assert "dac_her.config" in (
        imported_modules
    )
