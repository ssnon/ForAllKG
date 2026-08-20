from __future__ import annotations

import hashlib

from dac_her.corpus_acquisition.access_contracts import SourceArtifact
from dac_her.corpus_acquisition.materialization_contracts import (
    MaterializationPolicy,
)
from dac_her.corpus_acquisition.materialization_package import (
    materialize_artifact,
)
from pipeline_core.literature.catalog_contracts import CatalogWork


def test_text_artifact_materializes_with_metadata(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("scientific source text", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    artifact = SourceArtifact(
        artifact_id="a",
        work_id="w",
        role="supporting_information",
        status="downloaded",
        local_path=str(source),
        sha256=digest,
    )
    work = CatalogWork(
        work_id="w",
        title="Example",
        doi="10.1/example",
    )
    package = tmp_path / "package"
    row = materialize_artifact(
        materialization_id="m",
        paper_id="API_x",
        work=work,
        document_id="si1",
        role="supporting_information",
        artifact=artifact,
        package_dir=package,
        policy=MaterializationPolicy(policy_id="p"),
        project_root=tmp_path,
    )
    assert row.status == "materialized"
    assert (package / "normalized.md").exists()
    assert (package / "metadata.json").exists()
    assert row.scientific_result_inferred is False
    assert row.positive_evidence_promotion_performed is False


def test_source_sha_drift_fails_closed(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("changed", encoding="utf-8")
    artifact = SourceArtifact(
        artifact_id="a",
        work_id="w",
        role="supporting_information",
        status="downloaded",
        local_path=str(source),
        sha256="0" * 64,
    )
    work = CatalogWork(work_id="w", title="Example")
    import pytest
    with pytest.raises(RuntimeError):
        materialize_artifact(
            materialization_id="m",
            paper_id="API_x",
            work=work,
            document_id="si1",
            role="supporting_information",
            artifact=artifact,
            package_dir=tmp_path / "package",
            policy=MaterializationPolicy(policy_id="p"),
            project_root=tmp_path,
        )
