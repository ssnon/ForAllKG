from __future__ import annotations

from dataclasses import asdict
from datetime import (
    datetime as RealDateTime,
    timezone,
)
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from pipeline_core.document_config import (
    DocumentConfig,
    DocumentSelection,
    FigureProcessingConfig,
    PaperConfig,
    paper_config_fingerprint_payload,
)
from pipeline_core.extraction_policy import ExtractionPolicy
from domains.dac_her.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
)
import dac_her.run_state as legacy
import pipeline_core.document_provenance as provenance
import pipeline_core.evaluation_runtime.artifacts as evaluation_artifacts
import pipeline_core.serialization_primitives as serialization


def _paper(
    tmp_path: Path,
) -> PaperConfig:
    package_dir = (
        tmp_path / "package"
    )
    package_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    return PaperConfig(
        paper_id="paper_a",
        documents=(document,),
        enabled=True,
        resolution_file=None,
    )


def test_current_run_state_versions_and_domain_defaults_are_frozen():
    assert legacy.RUN_STATE_VERSION == (
        "semantic-si-assets-run-v5-strict-recovery"
    )

    assert legacy.ATTEMPT_LAYOUT_VERSION == (
        "run-attempt-provenance-v1"
    )

    signature = inspect.signature(
        legacy.compute_run_metadata
    )

    assert (
        signature.parameters[
            "runtime_options"
        ].default
        is None
    )

    assert (
        signature.parameters[
            "implementation_paths"
        ].default
        == ()
    )

    assert (
        signature.parameters[
            "prompt_version"
        ].default
        == PROMPT_VERSION
    )

    assert (
        signature.parameters[
            "system_prompt"
        ].default
        == SYSTEM_PROMPT
    )

    assert (
        signature.parameters[
            "domain_profile_id"
        ].default
        == "dac_her"
    )

    assert (
        signature.parameters[
            "data_root"
        ].default
        == "data_dac"
    )


def test_current_run_state_ownership_and_provenance_compatibility():
    assert (
        legacy.compute_run_metadata.__module__
        == "dac_her.run_state"
    )

    for name in (
        "paper_output_root",
        "run_directory",
        "attempt_directory",
        "write_latest_attempt_pointer",
        "write_latest_run_pointer",
        "resolve_run_directory",
    ):
        assert (
            getattr(
                legacy,
                name,
            ).__module__
            == "dac_her.run_state"
        )

    assert (
        legacy.document_source_fingerprints
        is provenance.document_source_fingerprints
    )

    assert (
        legacy.sha256_file
        is provenance.sha256_file
    )

    for name in (
        "sha256_bytes",
        "sha256_text",
        "canonical_json",
        "read_json",
        "write_json",
    ):
        assert (
            getattr(legacy, name)
            is getattr(
                serialization,
                name,
            )
        )

        assert (
            getattr(
                serialization,
                name,
            ).__module__
            == (
                "pipeline_core."
                "serialization_primitives"
            )
        )

    assert (
        provenance.sha256_bytes
        is serialization.sha256_bytes
    )


def test_hash_and_canonical_json_contract():
    raw = b"abc\x00\xff"

    assert legacy.sha256_bytes(
        raw
    ) == hashlib.sha256(
        raw
    ).hexdigest()

    text = "한글 η Δ"

    assert legacy.sha256_text(
        text
    ) == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    value = {
        "z": "한글",
        "a": [
            2,
            1,
        ],
    }

    assert legacy.canonical_json(
        value
    ) == (
        '{"a":[2,1],"z":"한글"}'
    )


def test_write_json_exact_byte_contract(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "nested"
        / "value.json"
    )

    result = legacy.write_json(
        path,
        {
            "z": "한글",
            "a": 1,
        },
    )

    assert result == path

    assert path.read_text(
        encoding="utf-8"
    ) == (
        '{\n'
        '  "a": 1,\n'
        '  "z": "한글"\n'
        '}'
    )

    assert legacy.read_json(
        path
    ) == {
        "a": 1,
        "z": "한글",
    }


def test_compute_run_metadata_exact_composition_and_fingerprint(
    tmp_path: Path,
):
    paper = _paper(
        tmp_path
    )

    policy = ExtractionPolicy()

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

    # Deliberately create in reverse lexical order.
    vocab_z = (
        vocab_dir / "z.yaml"
    )
    vocab_z.write_text(
        "z: 1\n",
        encoding="utf-8",
    )

    vocab_a = (
        vocab_dir / "a.yaml"
    )
    vocab_a.write_text(
        "a: 1\n",
        encoding="utf-8",
    )

    impl_z = (
        tmp_path / "z_impl.py"
    )
    impl_z.write_text(
        "Z = 1\n",
        encoding="utf-8",
    )

    impl_a = (
        tmp_path / "a_impl.py"
    )
    impl_a.write_text(
        "A = 1\n",
        encoding="utf-8",
    )

    metadata = (
        legacy.compute_run_metadata(
            project_root=tmp_path,
            paper=paper,
            policy=policy,
            model="test-model",
            provider="test-provider",
            schemas_path=schemas_path,
            chunking_path=chunking_path,
            runtime_options={
                "beta": 2,
                "alpha": 1,
            },
            implementation_paths=(
                impl_z,
                impl_a,
            ),
            prompt_version=(
                "test-prompt-v1"
            ),
            system_prompt=(
                "test system prompt"
            ),
            domain_profile_id=(
                "test-domain"
            ),
            data_root="test-data",
        )
    )

    assert set(
        metadata
    ) == {
        "run_state_version",
        "paper",
        "document_sources",
        "domain_profile_id",
        "data_root",
        "prompt_version",
        "prompt_sha256",
        "schema_sha256",
        "chunking_sha256",
        "policy",
        "model",
        "provider",
        "runtime_options",
        "vocabularies",
        "project_root",
        "implementation_files",
        "run_fingerprint",
        "run_id",
        "created_at_utc",
    }

    assert (
        metadata[
            "run_state_version"
        ]
        == legacy.RUN_STATE_VERSION
    )

    assert (
        metadata["paper"]
        == paper_config_fingerprint_payload(
            paper
        )
    )

    assert (
        metadata[
            "document_sources"
        ]
        == legacy.document_source_fingerprints(
            paper
        )
    )

    assert (
        metadata[
            "domain_profile_id"
        ]
        == "test-domain"
    )

    assert (
        metadata["data_root"]
        == "test-data"
    )

    assert (
        metadata[
            "prompt_version"
        ]
        == "test-prompt-v1"
    )

    assert (
        metadata[
            "prompt_sha256"
        ]
        == legacy.sha256_text(
            "test system prompt"
        )
    )

    assert (
        metadata[
            "schema_sha256"
        ]
        == provenance.sha256_file(
            schemas_path
        )
    )

    assert (
        metadata[
            "chunking_sha256"
        ]
        == provenance.sha256_file(
            chunking_path
        )
    )

    assert (
        metadata["policy"]
        == asdict(policy)
    )

    assert (
        metadata["model"]
        == "test-model"
    )

    assert (
        metadata["provider"]
        == "test-provider"
    )

    assert (
        metadata[
            "runtime_options"
        ]
        == {
            "beta": 2,
            "alpha": 1,
        }
    )

    assert [
        row[
            "relative_path"
        ]
        for row
        in metadata[
            "vocabularies"
        ]
    ] == [
        "configs/vocabularies/a.yaml",
        "configs/vocabularies/z.yaml",
    ]

    assert [
        row["sha256"]
        for row
        in metadata[
            "vocabularies"
        ]
    ] == [
        provenance.sha256_file(
            vocab_a
        ),
        provenance.sha256_file(
            vocab_z
        ),
    ]

    assert [
        row[
            "relative_path"
        ]
        for row
        in metadata[
            "implementation_files"
        ]
    ] == [
        "a_impl.py",
        "z_impl.py",
    ]

    assert [
        row["sha256"]
        for row
        in metadata[
            "implementation_files"
        ]
    ] == [
        provenance.sha256_file(
            impl_a
        ),
        provenance.sha256_file(
            impl_z
        ),
    ]

    assert (
        metadata[
            "project_root"
        ]
        == str(
            tmp_path.resolve()
        )
    )

    fingerprint_payload = dict(
        metadata
    )

    fingerprint_payload.pop(
        "run_fingerprint"
    )
    fingerprint_payload.pop(
        "run_id"
    )
    fingerprint_payload.pop(
        "created_at_utc"
    )

    expected_fingerprint = (
        legacy.sha256_text(
            legacy.canonical_json(
                fingerprint_payload
            )
        )
    )

    assert (
        metadata[
            "run_fingerprint"
        ]
        == expected_fingerprint
    )

    assert (
        metadata["run_id"]
        == expected_fingerprint[:16]
    )

    created_at = (
        RealDateTime.fromisoformat(
            metadata[
                "created_at_utc"
            ]
        )
    )

    assert (
        created_at.utcoffset()
        is not None
    )


def test_run_layout_exact_paths(
    tmp_path: Path,
):
    project_root = (
        tmp_path / "project"
    )
    project_root.mkdir()

    expected_paper_root = (
        project_root.resolve()
        / "data_dac"
        / "extracted"
        / "paper-A"
    )

    assert (
        legacy.paper_output_root(
            project_root,
            "paper-A",
        )
        == expected_paper_root
    )

    assert (
        legacy.run_directory(
            project_root,
            "paper-A",
            "run-A",
        )
        == (
            expected_paper_root
            / "runs"
            / "run-A"
        )
    )

    assert (
        legacy.attempt_directory(
            project_root,
            "paper-A",
            "run-A",
            "attempt-A",
        )
        == (
            expected_paper_root
            / "runs"
            / "run-A"
            / "attempts"
            / "attempt-A"
        )
    )

    absolute_root = (
        tmp_path / "absolute-data"
    )

    assert (
        legacy.paper_output_root(
            project_root,
            "paper-B",
            data_root=absolute_root,
        )
        == (
            absolute_root.resolve()
            / "extracted"
            / "paper-B"
        )
    )


def test_pointer_payload_and_resolution_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FrozenDateTime:
        @classmethod
        def now(
            cls,
            tz=None,
        ):
            return RealDateTime(
                2026,
                8,
                19,
                1,
                2,
                3,
                tzinfo=timezone.utc,
            )

    monkeypatch.setattr(
        legacy,
        "datetime",
        FrozenDateTime,
    )

    project_root = (
        tmp_path / "project"
    )
    project_root.mkdir()

    data_root = (
        tmp_path / "data"
    )

    metadata = {
        "run_id": "run-A",
        "run_fingerprint": "fp-A",
    }

    family_dir = (
        legacy.run_directory(
            project_root,
            "paper-A",
            "run-A",
            data_root=data_root,
        )
    )

    concrete_dir = (
        legacy.attempt_directory(
            project_root,
            "paper-A",
            "run-A",
            "attempt-A",
            data_root=data_root,
        )
    )

    concrete_dir.mkdir(
        parents=True
    )

    latest_attempt_path = (
        legacy.write_latest_attempt_pointer(
            project_root=project_root,
            paper_id="paper-A",
            run_metadata=metadata,
            attempt_id="attempt-A",
            data_root=data_root,
        )
    )

    latest_run_path = (
        legacy.write_latest_run_pointer(
            project_root=project_root,
            paper_id="paper-A",
            run_metadata=metadata,
            data_root=data_root,
            attempt_id="attempt-A",
        )
    )

    assert legacy.read_json(
        latest_attempt_path
    ) == {
        "paper_id": "paper-A",
        "run_id": "run-A",
        "run_fingerprint": "fp-A",
        "attempt_layout_version": (
            legacy.ATTEMPT_LAYOUT_VERSION
        ),
        "attempt_id": "attempt-A",
        "attempt_directory": str(
            concrete_dir
        ),
        "updated_at_utc": (
            "2026-08-19T01:02:03+00:00"
        ),
    }

    assert legacy.read_json(
        latest_run_path
    ) == {
        "paper_id": "paper-A",
        "run_id": "run-A",
        "run_fingerprint": "fp-A",
        "run_directory": str(
            family_dir
        ),
        "updated_at_utc": (
            "2026-08-19T01:02:03+00:00"
        ),
        "attempt_layout_version": (
            legacy.ATTEMPT_LAYOUT_VERSION
        ),
        "attempt_id": "attempt-A",
        "attempt_directory": str(
            concrete_dir
        ),
    }

    assert (
        legacy.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id=None,
            data_root=data_root,
        )
        == concrete_dir.resolve()
    )


def test_stale_latest_run_attempt_falls_back_to_latest_attempt(
    tmp_path: Path,
):
    project_root = (
        tmp_path / "project"
    )
    project_root.mkdir()

    data_root = (
        tmp_path / "data"
    )

    paper_id = "paper-A"
    run_id = "run-A"

    family_dir = (
        legacy.run_directory(
            project_root,
            paper_id,
            run_id,
            data_root=data_root,
        )
    )

    current_attempt = (
        legacy.attempt_directory(
            project_root,
            paper_id,
            run_id,
            "attempt-current",
            data_root=data_root,
        )
    )

    current_attempt.mkdir(
        parents=True
    )

    stale_attempt = (
        family_dir
        / "attempts"
        / "attempt-stale"
    )

    legacy.write_json(
        family_dir
        / "latest_attempt.json",
        {
            "attempt_directory": str(
                current_attempt
            ),
        },
    )

    legacy.write_json(
        (
            legacy.paper_output_root(
                project_root,
                paper_id,
                data_root=data_root,
            )
            / "latest_run.json"
        ),
        {
            "run_directory": str(
                family_dir
            ),
            "attempt_directory": str(
                stale_attempt
            ),
        },
    )

    assert (
        legacy.resolve_run_directory(
            project_root=project_root,
            paper_id=paper_id,
            run_id=None,
            data_root=data_root,
        )
        == current_attempt.resolve()
    )


def test_resolver_error_contracts_are_frozen(
    tmp_path: Path,
):
    project_root = (
        tmp_path / "project"
    )
    project_root.mkdir()

    data_root = (
        tmp_path / "data"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            r"No latest run pointer "
            r"found for 'paper-A':"
        ),
    ):
        legacy.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id=None,
            data_root=data_root,
        )

    family_dir = (
        legacy.run_directory(
            project_root,
            "paper-A",
            "run-A",
            data_root=data_root,
        )
    )

    family_dir.mkdir(
        parents=True
    )

    missing_attempt = (
        family_dir
        / "attempts"
        / "missing"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "^Run directory not found: "
            + str(
                missing_attempt
            )
            + "$"
        ),
    ):
        legacy.resolve_run_directory(
            project_root=project_root,
            paper_id="paper-A",
            run_id="run-A",
            data_root=data_root,
            attempt_id="missing",
        )


def test_run_canonical_json_remains_stricter_than_evaluation_artifact_helper():
    class Dumpable:
        def model_dump(
            self,
            *,
            mode: str,
        ):
            assert mode == "json"
            return {
                "value": 1,
            }

    assert (
        evaluation_artifacts.canonical_json(
            Dumpable()
        )
        == '{"value":1}'
    )

    with pytest.raises(
        TypeError
    ):
        serialization.canonical_json(
            Dumpable()
        )
