from __future__ import annotations

from pathlib import Path

import yaml

from dac_her.corpus_acquisition.materialization_contracts import (
    MaterializationPolicy,
    MaterializedDocument,
)
from dac_her.corpus_acquisition.materialization_package import (
    generated_paper_config_entry,
    stable_paper_id,
    write_generated_config,
)


def _doc(document_id, role, package_dir):
    return MaterializedDocument(
        materialization_id="m",
        paper_id="API_x",
        work_id="w",
        document_id=document_id,
        role=role,
        source_artifact_id="a:" + document_id,
        source_path="/tmp/source.pdf",
        source_extension=".pdf",
        status="materialized",
        materializer="test",
        package_dir=str(package_dir),
        markdown_path=str(package_dir / "normalized.md"),
        metadata_path=str(package_dir / "metadata.json"),
        markdown_sha256="x",
        markdown_char_count=10,
    )


def test_stable_paper_id_is_deterministic_and_prefix_scoped():
    assert stable_paper_id(prefix="SERS_API", work_id="w1") == stable_paper_id(
        prefix="SERS_API", work_id="w1"
    )
    assert stable_paper_id(prefix="HER_API", work_id="w1") != stable_paper_id(
        prefix="SERS_API", work_id="w1"
    )


def test_generated_config_uses_existing_main_and_si_selection_semantics(tmp_path):
    policy = MaterializationPolicy(policy_id="p")
    main = _doc("main", "main", tmp_path / "main")
    si = _doc("si1", "supporting_information", tmp_path / "si1")
    entry = generated_paper_config_entry(
        paper_id="API_x",
        documents=[main, si],
        policy=policy,
    )
    assert entry is not None
    assert entry["documents"][0]["selection"]["mode"] == "whole_document"
    assert entry["documents"][1]["selection"] == {
        "mode": "referenced_blocks",
        "fallback": "skip",
        "reference_scope": "whole_main",
    }


def test_no_main_means_not_extraction_ready(tmp_path):
    policy = MaterializationPolicy(policy_id="p")
    si = _doc("si1", "supporting_information", tmp_path / "si1")
    assert generated_paper_config_entry(
        paper_id="API_x",
        documents=[si],
        policy=policy,
    ) is None


def test_generated_config_version_three(tmp_path):
    path = tmp_path / "generated.yaml"
    write_generated_config(
        config_path=path,
        papers={
            "API_x": {
                "enabled": True,
                "documents": [],
                "resolution_file": None,
            }
        },
    )
    loaded = yaml.safe_load(path.read_text())
    assert loaded["version"] == 3
    assert "API_x" in loaded["papers"]
