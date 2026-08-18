from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from dac_her.config import (
    DocumentConfig,
    DocumentSelection,
    FigureProcessingConfig,
    PaperConfig,
)
from dac_her.run_state import (
    document_source_fingerprints,
    sha256_file,
)


def _document(
    *,
    document_id: str,
    role: str,
    package_dir: Path,
    markdown_path: Path,
    metadata_path: Path | None,
) -> DocumentConfig:
    return DocumentConfig(
        document_id=document_id,
        role=role,
        package_dir=package_dir,
        markdown_path=markdown_path,
        metadata_path=metadata_path,
        selection=DocumentSelection(
            mode="whole_document"
        ),
        figure_processing=(
            FigureProcessingConfig()
        ),
    )


def test_sha256_file_matches_exact_file_bytes(
    tmp_path: Path,
):
    path = tmp_path / "payload.bin"
    payload = b"\x00document-provenance\xff"

    path.write_bytes(payload)

    assert sha256_file(path) == (
        hashlib.sha256(
            payload
        ).hexdigest()
    )


def test_document_source_fingerprints_preserve_document_and_package_semantics(
    tmp_path: Path,
):
    main_dir = tmp_path / "main"
    si_dir = tmp_path / "si"

    main_dir.mkdir()
    si_dir.mkdir()

    main_markdown = (
        main_dir / "main.md"
    )
    main_metadata = (
        main_dir / "metadata.json"
    )
    main_figure = (
        main_dir / "figure.png"
    )
    ignored = (
        main_dir / "notes.txt"
    )

    main_markdown.write_text(
        "# Main\n\nEvidence.\n",
        encoding="utf-8",
    )

    main_metadata.write_text(
        json.dumps(
            {
                "title": "Paper A",
            }
        ),
        encoding="utf-8",
    )

    main_figure.write_bytes(
        b"fake-png"
    )

    ignored.write_text(
        "not part of package fingerprint",
        encoding="utf-8",
    )

    si_markdown = (
        si_dir / "si.md"
    )

    si_markdown.write_text(
        "# SI\n\nSupplementary evidence.\n",
        encoding="utf-8",
    )

    main = _document(
        document_id="main",
        role="main",
        package_dir=main_dir,
        markdown_path=main_markdown,
        metadata_path=main_metadata,
    )

    si = _document(
        document_id="si",
        role="supporting_information",
        package_dir=si_dir,
        markdown_path=si_markdown,
        metadata_path=None,
    )

    paper = PaperConfig(
        paper_id="paper_a",
        documents=(
            main,
            si,
        ),
        enabled=True,
        resolution_file=None,
    )

    records = (
        document_source_fingerprints(
            paper
        )
    )

    assert records == [
        {
            "document_id": "main",
            "role": "main",
            "markdown_sha256": (
                sha256_file(
                    main_markdown
                )
            ),
            "metadata_sha256": (
                sha256_file(
                    main_metadata
                )
            ),
            "package_files": [
                {
                    "relative_path": (
                        "figure.png"
                    ),
                    "sha256": (
                        sha256_file(
                            main_figure
                        )
                    ),
                },
                {
                    "relative_path": (
                        "main.md"
                    ),
                    "sha256": (
                        sha256_file(
                            main_markdown
                        )
                    ),
                },
                {
                    "relative_path": (
                        "metadata.json"
                    ),
                    "sha256": (
                        sha256_file(
                            main_metadata
                        )
                    ),
                },
            ],
        },
        {
            "document_id": "si",
            "role": (
                "supporting_information"
            ),
            "markdown_sha256": (
                sha256_file(
                    si_markdown
                )
            ),
            "metadata_sha256": None,
            "package_files": [
                {
                    "relative_path": (
                        "si.md"
                    ),
                    "sha256": (
                        sha256_file(
                            si_markdown
                        )
                    ),
                },
            ],
        },
    ]

    assert all(
        item["relative_path"]
        != "notes.txt"
        for item
        in records[0]["package_files"]
    )


def test_current_strict_bridge_depends_only_on_document_fingerprints_from_run_state():
    path = Path(
        "pipeline_core/"
        "strict_bridge_corpus_pipeline.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    run_state_imports = []

    for node in ast.walk(tree):
        if (
            isinstance(
                node,
                ast.ImportFrom,
            )
            and node.module
            == "dac_her.run_state"
        ):
            run_state_imports.extend(
                alias.name
                for alias in node.names
            )

    assert run_state_imports == [
        "document_source_fingerprints"
    ]


def test_shared_document_provenance_owner_does_not_exist_yet():
    assert not Path(
        "pipeline_core/"
        "document_provenance.py"
    ).exists()
