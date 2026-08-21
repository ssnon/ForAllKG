from __future__ import annotations

import ast
import inspect
from pathlib import Path

from pipeline_core.document_config import (
    DocumentConfig,
    DocumentSelection,
    FigureProcessingConfig,
    PaperConfig,
)
from pipeline_core.extraction_policy import (
    ExtractionPolicy,
)
from pipeline_core.serialization_primitives import (
    canonical_json,
    sha256_text,
)
import pipeline_core.run_metadata as runtime


def _paper(
    tmp_path: Path,
) -> PaperConfig:
    package_dir = (
        tmp_path / "package"
    )
    package_dir.mkdir(
        parents=True
    )

    markdown_path = (
        package_dir / "main.md"
    )
    markdown_path.write_text(
        "# Main\n\nSource text.\n",
        encoding="utf-8",
    )

    return PaperConfig(
        paper_id="paper_a",
        documents=(
            DocumentConfig(
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
            ),
        ),
        enabled=True,
        resolution_file=None,
    )


def test_run_metadata_boundary_has_no_reverse_dependency():
    path = Path(
        "pipeline_core/run_metadata.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    violations = []

    for node in ast.walk(tree):
        names = []

        if isinstance(
            node,
            ast.Import,
        ):
            names = [
                alias.name
                for alias in node.names
            ]

        elif (
            isinstance(
                node,
                ast.ImportFrom,
            )
            and node.level == 0
        ):
            names = [
                node.module or ""
            ]

        for name in names:
            if (
                name == "dac_her"
                or name.startswith(
                    "dac_her."
                )
                or name == "domains"
                or name.startswith(
                    "domains."
                )
                or name == "scripts"
                or name.startswith(
                    "scripts."
                )
            ):
                violations.append(
                    (
                        node.lineno,
                        name,
                    )
                )

    assert violations == []


def test_run_metadata_domain_inputs_are_explicit():
    signature = inspect.signature(
        runtime.compute_run_metadata
    )

    for name in (
        "prompt_version",
        "system_prompt",
        "domain_profile_id",
        "data_root",
    ):
        assert (
            signature.parameters[
                name
            ].default
            is inspect.Parameter.empty
        )

    assert (
        runtime.RUN_STATE_VERSION
        == "semantic-si-assets-run-v5-strict-recovery"
    )


def test_compute_run_metadata_fingerprint_contract(
    tmp_path: Path,
):
    paper = _paper(
        tmp_path
    )

    schemas_path = (
        tmp_path / "schemas.py"
    )
    schemas_path.write_text(
        "SCHEMA = 1\n",
        encoding="utf-8",
    )

    chunking_path = (
        tmp_path / "chunking.py"
    )
    chunking_path.write_text(
        "CHUNKING = 1\n",
        encoding="utf-8",
    )

    vocab_dir = (
        tmp_path
        / "configs"
        / "vocabularies"
    )
    vocab_dir.mkdir(
        parents=True
    )

    (
        vocab_dir / "z.yaml"
    ).write_text(
        "z: 1\n",
        encoding="utf-8",
    )

    (
        vocab_dir / "a.yaml"
    ).write_text(
        "a: 1\n",
        encoding="utf-8",
    )

    impl_z = (
        tmp_path / "z_impl.py"
    )
    impl_a = (
        tmp_path / "a_impl.py"
    )

    impl_z.write_text(
        "Z = 1\n",
        encoding="utf-8",
    )
    impl_a.write_text(
        "A = 1\n",
        encoding="utf-8",
    )

    metadata = (
        runtime.compute_run_metadata(
            project_root=tmp_path,
            paper=paper,
            policy=ExtractionPolicy(),
            model="test-model",
            provider="test-provider",
            schemas_path=schemas_path,
            chunking_path=chunking_path,
            prompt_version="prompt-v1",
            system_prompt="system prompt",
            domain_profile_id="test-domain",
            data_root="test-data",
            runtime_options={
                "beta": 2,
                "alpha": 1,
            },
            implementation_paths=(
                impl_z,
                impl_a,
            ),
        )
    )

    assert (
        metadata["domain_profile_id"]
        == "test-domain"
    )
    assert (
        metadata["prompt_version"]
        == "prompt-v1"
    )

    assert [
        row["relative_path"]
        for row
        in metadata[
            "implementation_files"
        ]
    ] == [
        "a_impl.py",
        "z_impl.py",
    ]

    payload = dict(
        metadata
    )

    payload.pop(
        "run_fingerprint"
    )
    payload.pop(
        "run_id"
    )
    payload.pop(
        "created_at_utc"
    )

    expected = sha256_text(
        canonical_json(
            payload
        )
    )

    assert (
        metadata["run_fingerprint"]
        == expected
    )
    assert (
        metadata["run_id"]
        == expected[:16]
    )
