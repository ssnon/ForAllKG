from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from scripts.materialize_corpus_documents_incremental import (
    _assign_si_document_ids,
    _can_reuse_document,
)


def _artifact(artifact_id: str, *, role: str = "supporting_information"):
    return SimpleNamespace(
        artifact_id=artifact_id,
        role=role,
        sha256=f"sha-{artifact_id}",
    )


def test_assign_si_document_ids_preserves_existing_ids_and_appends_new():
    prior = [
        SimpleNamespace(
            role="supporting_information",
            source_artifact_id="artifact-b",
            document_id="si2",
        ),
        SimpleNamespace(
            role="supporting_information",
            source_artifact_id="artifact-d",
            document_id="si5",
        ),
    ]
    artifacts = (
        _artifact("artifact-a"),
        _artifact("artifact-b"),
        _artifact("artifact-d"),
    )

    assigned = _assign_si_document_ids(artifacts=artifacts, prior=prior)

    assert assigned["artifact-b"] == "si2"
    assert assigned["artifact-d"] == "si5"
    assert assigned["artifact-a"] == "si6"


def test_verified_materialized_document_is_reused(tmp_path):
    markdown = tmp_path / "normalized.md"
    markdown.write_text("# paper\n", encoding="utf-8")
    markdown_sha = hashlib.sha256(markdown.read_bytes()).hexdigest()
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "paper_id": "P1",
                "work_id": "W1",
                "bibliographic": {
                    "title": "Title",
                    "doi": "10.1/example",
                    "year": 2026,
                },
                "source_artifact": {
                    "artifact_id": "A1",
                    "sha256": "abc",
                },
            }
        ),
        encoding="utf-8",
    )
    row = SimpleNamespace(
        materialization_id="M1",
        paper_id="P1",
        work_id="W1",
        document_id="main",
        role="main",
        source_artifact_id="A1",
        source_artifact_sha256="abc",
        status="materialized",
        markdown_path=str(markdown),
        metadata_path=str(metadata),
        markdown_sha256=markdown_sha,
    )
    artifact = SimpleNamespace(
        artifact_id="A1",
        role="main",
        sha256="abc",
    )
    work = SimpleNamespace(
        work_id="W1",
        title="Title",
        doi="10.1/example",
        year=2026,
    )

    reusable, reason = _can_reuse_document(
        row=row,
        artifact=artifact,
        work=work,
        materialization_id="M1",
        context_compatible=True,
        retry_failed=False,
        force=False,
    )

    assert reusable is True
    assert reason == "reuse_verified"


def test_markdown_drift_invalidates_cache(tmp_path):
    markdown = tmp_path / "normalized.md"
    markdown.write_text("changed\n", encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    row = SimpleNamespace(
        materialization_id="M1",
        paper_id="P1",
        work_id="W1",
        document_id="main",
        role="main",
        source_artifact_id="A1",
        source_artifact_sha256="abc",
        status="materialized",
        markdown_path=str(markdown),
        metadata_path=str(metadata),
        markdown_sha256="not-the-current-sha",
    )
    artifact = SimpleNamespace(artifact_id="A1", role="main", sha256="abc")
    work = SimpleNamespace(work_id="W1", title="Title", doi=None, year=2026)

    reusable, reason = _can_reuse_document(
        row=row,
        artifact=artifact,
        work=work,
        materialization_id="M1",
        context_compatible=True,
        retry_failed=False,
        force=False,
    )

    assert reusable is False
    assert reason == "markdown_sha_changed"


def test_retry_failed_only_invalidates_failed_document():
    row = SimpleNamespace(
        materialization_id="M1",
        role="main",
        source_artifact_id="A1",
        source_artifact_sha256="abc",
        status="failed",
    )
    artifact = SimpleNamespace(artifact_id="A1", role="main", sha256="abc")
    work = SimpleNamespace(work_id="W1", title="Title", doi=None, year=2026)

    reusable, reason = _can_reuse_document(
        row=row,
        artifact=artifact,
        work=work,
        materialization_id="M1",
        context_compatible=True,
        retry_failed=True,
        force=False,
    )

    assert reusable is False
    assert reason == "retry_failed"
