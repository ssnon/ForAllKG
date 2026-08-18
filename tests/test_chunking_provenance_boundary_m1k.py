from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import dac_her.chunking as chunking

from dac_her.config import (
    DocumentConfig,
    DocumentSelection,
    FigureProcessingConfig,
    PaperConfig,
)
from dac_her.extraction_policy import (
    ExtractionPolicy,
)
from dac_her.run_state import (
    compute_run_metadata,
    sha256_file,
)


def test_chunk_spec_surface_is_frozen():
    assert [
        field.name
        for field in fields(
            chunking.ChunkSpec
        )
    ] == [
        "paper_id",
        "section",
        "index",
        "core_text",
        "left_context",
        "right_context",
        "chunk_id",
        "document_id",
        "document_role",
        "page_ids",
        "asset_ids",
        "asset_paths",
        "asset_pages",
        "asset_locators",
        "asset_context",
        "split_depth",
    ]


def test_chunk_identity_is_deterministic_and_document_aware():
    first = chunking.make_chunk_id(
        "paper_a",
        "## Results",
        "same core text",
        document_id="main",
    )

    repeated = chunking.make_chunk_id(
        "paper_a",
        "## Results",
        "same core text",
        document_id="main",
    )

    other_document = (
        chunking.make_chunk_id(
            "paper_a",
            "## Results",
            "same core text",
            document_id="si",
        )
    )

    other_core = chunking.make_chunk_id(
        "paper_a",
        "## Results",
        "different core text",
        document_id="main",
    )

    assert first == repeated
    assert first != other_document
    assert first != other_core

    assert first.startswith(
        "paper_a:main:"
    )


def test_run_metadata_hashes_exact_chunking_file(
    tmp_path: Path,
):
    package_dir = (
        tmp_path / "package"
    )
    package_dir.mkdir()

    markdown_path = (
        package_dir / "main.md"
    )
    markdown_path.write_text(
        "# Main\n\nSource text.\n",
        encoding="utf-8",
    )

    document = DocumentConfig(
        document_id="main",
        role="main",
        package_dir=package_dir,
        markdown_path=markdown_path,
        metadata_path=None,
        selection=DocumentSelection(
            mode="whole_document"
        ),
        figure_processing=(
            FigureProcessingConfig()
        ),
    )

    paper = PaperConfig(
        paper_id="paper_a",
        documents=(document,),
        enabled=True,
        resolution_file=None,
    )

    schemas_path = (
        tmp_path / "schemas.py"
    )
    schemas_path.write_text(
        "SCHEMA = 1\n",
        encoding="utf-8",
    )

    chunking_path = (
        tmp_path / "chunking_impl.py"
    )
    chunking_path.write_text(
        "CHUNKING = 1\n",
        encoding="utf-8",
    )

    (
        tmp_path
        / "configs"
        / "vocabularies"
    ).mkdir(
        parents=True
    )

    first = compute_run_metadata(
        project_root=tmp_path,
        paper=paper,
        policy=ExtractionPolicy(),
        model="test-model",
        provider=None,
        schemas_path=schemas_path,
        chunking_path=chunking_path,
        runtime_options={},
        implementation_paths=(),
        prompt_version="test-prompt-v1",
        system_prompt="test prompt",
        domain_profile_id="dac_her",
        data_root="data_dac",
    )

    assert (
        first["chunking_sha256"]
        == sha256_file(
            chunking_path
        )
    )

    first_fingerprint = (
        first["run_fingerprint"]
    )

    chunking_path.write_text(
        "CHUNKING = 2\n",
        encoding="utf-8",
    )

    second = compute_run_metadata(
        project_root=tmp_path,
        paper=paper,
        policy=ExtractionPolicy(),
        model="test-model",
        provider=None,
        schemas_path=schemas_path,
        chunking_path=chunking_path,
        runtime_options={},
        implementation_paths=(),
        prompt_version="test-prompt-v1",
        system_prompt="test prompt",
        domain_profile_id="dac_her",
        data_root="data_dac",
    )

    assert (
        second["chunking_sha256"]
        == sha256_file(
            chunking_path
        )
    )

    assert (
        second["chunking_sha256"]
        != first["chunking_sha256"]
    )

    assert (
        second["run_fingerprint"]
        != first_fingerprint
    )


def test_extraction_provenance_points_at_shared_chunking_owner():
    source = Path(
        "scripts/extract_paper.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "import pipeline_core.chunking "
        "as chunking_module"
        in source
    )

    assert (
        "import dac_her.chunking "
        "as chunking_module"
        not in source
    )

    assert (
        "chunking_path="
        "chunking_module.__file__"
        in source
    )


def test_incremental_full_freshness_points_at_shared_chunking_owner():
    source = Path(
        "dac_her/incremental_reconcile.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "import pipeline_core.chunking "
        "as chunking_module"
        in source
    )

    assert (
        "import dac_her.chunking "
        "as chunking_module"
        not in source
    )

    assert (
        "_sha256_file("
        "Path(chunking_module.__file__)"
        ")"
        in source
    )


def test_shared_chunking_dependency_boundary():
    path = Path(
        "pipeline_core/chunking.py"
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
        "pipeline_core.extraction_policy",
    }.issubset(
        imported_modules
    )

def test_chunking_facade_reexports_shared_core_identity():
    import dac_her.chunking as legacy
    import pipeline_core.chunking as core

    names = (
        "ChunkSpec",
        "count_tokens",
        "first_tokens",
        "last_tokens",
        "make_chunk_id",
        "split_long_unit",
        "paragraph_units",
        "build_core_chunks",
        "create_chunks",
        "split_chunk_in_half",
    )

    for name in names:
        assert (
            getattr(
                legacy,
                name,
            )
            is getattr(
                core,
                name,
            )
        )


def test_shared_chunking_owner_is_real_provenance_file():
    import pipeline_core.chunking as core

    expected = (
        Path(
            "pipeline_core/chunking.py"
        )
        .resolve()
    )

    assert (
        Path(core.__file__).resolve()
        == expected
    )
