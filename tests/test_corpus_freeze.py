from __future__ import annotations

from pathlib import Path

from dac_her.corpus_freeze import FreezeOptions, freeze_ingestion_manifest


def _doc(pid: str, title: str, main_fp: str, path: Path, qc: str = "passed", si=None):
    return {
        "paper_id": pid,
        "title": title,
        "annotator": "tester",
        "main_markdown": str(path),
        "supporting_markdown": list(si or []),
        "source_file_name": f"{pid}.pdf",
        "source_fingerprint": {"main": main_fp, "si": []},
        "marker_version": "2.0.0",
        "qc_status": qc,
    }


def test_exact_fingerprint_dedupes_but_title_only_duplicate_is_kept(tmp_path: Path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    c = tmp_path / "c.md"
    for p, text in [(a, "alpha"), (b, "alpha"), (c, "different")]:
        p.write_text(text, encoding="utf-8")

    manifest = {
        "schema_version": "graphagentsdac-ingestion-corpus-v01",
        "corpus_id": "demo",
        "document_count": 3,
        "documents": [
            _doc("A", "Same title", "fp1", a),
            _doc("B", "Same title", "fp1", b),
            _doc("C", "Same title", "fp2", c),
        ],
    }
    frozen = freeze_ingestion_manifest(
        manifest,
        source_manifest_path="input.json",
        project_root=tmp_path,
        options=FreezeOptions(verify_paths=True),
    )
    assert frozen["document_count"] == 2
    assert frozen["deduplicated_document_count"] == 1
    assert len(frozen["exact_duplicate_groups"]) == 1
    assert any(g["review_required"] for g in frozen["title_review_groups"])
    ids = {d["paper_id"] for d in frozen["documents"]}
    assert "C" in ids


def test_warning_can_be_excluded(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("hello", encoding="utf-8")
    manifest = {
        "schema_version": "graphagentsdac-ingestion-corpus-v01",
        "corpus_id": "demo",
        "document_count": 1,
        "documents": [_doc("A", "A", "fp", p, qc="passed_with_warnings")],
    }
    frozen = freeze_ingestion_manifest(
        manifest,
        source_manifest_path="input.json",
        project_root=tmp_path,
        options=FreezeOptions(include_warnings=False, verify_paths=True),
    )
    assert frozen["document_count"] == 0
    assert frozen["excluded_qc_count"] == 1


def test_exact_duplicate_merges_same_si_fingerprint_once(tmp_path: Path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    si1 = tmp_path / "si1.md"
    si2 = tmp_path / "si2.md"
    for p, text in [(a, "a"), (b, "b"), (si1, "same-si"), (si2, "same-si")]:
        p.write_text(text, encoding="utf-8")
    da = _doc("A", "A", "same-main", a, si=[str(si1)])
    db = _doc("B", "A", "same-main", b, si=[str(si2)])
    da["source_fingerprint"]["si"] = ["same-si-fp"]
    db["source_fingerprint"]["si"] = ["same-si-fp"]
    manifest = {
        "schema_version": "graphagentsdac-ingestion-corpus-v01",
        "corpus_id": "demo",
        "document_count": 2,
        "documents": [da, db],
    }
    frozen = freeze_ingestion_manifest(
        manifest,
        source_manifest_path="input.json",
        project_root=tmp_path,
        options=FreezeOptions(verify_paths=True),
    )
    assert frozen["document_count"] == 1
    assert len(frozen["documents"][0]["supporting_markdown"]) == 1
    assert frozen["exact_duplicate_groups"][0]["merged_supporting_markdown_count"] == 1
