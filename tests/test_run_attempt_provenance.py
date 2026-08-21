from __future__ import annotations

import json
from pathlib import Path

from domains.dac_her.bridge_run_state import compute_bridge_extraction_metadata
import pipeline_core.run_lifecycle as run_lifecycle
from pipeline_core.serialization_primitives import write_json


attempt_directory = run_lifecycle.attempt_directory
resolve_run_directory = run_lifecycle.resolve_run_directory
run_directory = run_lifecycle.run_directory

_ATTEMPT_LAYOUT_VERSION = "run-attempt-provenance-v1"
_TEST_UPDATED_AT_UTC = "2026-08-21T00:00:00+00:00"


def write_latest_attempt_pointer(
    *,
    project_root,
    paper_id,
    run_metadata,
    attempt_id,
    data_root,
):
    return run_lifecycle.write_latest_attempt_pointer(
        project_root=project_root,
        paper_id=paper_id,
        run_metadata=run_metadata,
        attempt_id=attempt_id,
        data_root=data_root,
        attempt_layout_version=_ATTEMPT_LAYOUT_VERSION,
        updated_at_utc=_TEST_UPDATED_AT_UTC,
    )


def write_latest_run_pointer(
    *,
    project_root,
    paper_id,
    run_metadata,
    data_root,
    attempt_id=None,
):
    return run_lifecycle.write_latest_run_pointer(
        project_root=project_root,
        paper_id=paper_id,
        run_metadata=run_metadata,
        data_root=data_root,
        attempt_layout_version=_ATTEMPT_LAYOUT_VERSION,
        updated_at_utc=_TEST_UPDATED_AT_UTC,
        attempt_id=attempt_id,
    )


def _metadata(run_id: str = "run-abc") -> dict[str, str]:
    return {
        "run_id": run_id,
        "run_fingerprint": f"fingerprint-{run_id}",
    }


def test_attempt_pointer_resolves_concrete_attempt_and_preserves_family_identity(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    data_root = tmp_path / "data"
    paper_id = "paper_A"
    meta = _metadata()
    family_dir = run_directory(
        project_root,
        paper_id,
        meta["run_id"],
        data_root=data_root,
    )
    attempt_id = "20260812T120000.000000Z"
    concrete_dir = attempt_directory(
        project_root,
        paper_id,
        meta["run_id"],
        attempt_id,
        data_root=data_root,
    )
    concrete_dir.mkdir(parents=True)
    write_json(
        concrete_dir / "run.json",
        {**meta, "attempt_id": attempt_id},
    )

    write_latest_attempt_pointer(
        project_root=project_root,
        paper_id=paper_id,
        run_metadata=meta,
        attempt_id=attempt_id,
        data_root=data_root,
    )
    write_latest_run_pointer(
        project_root=project_root,
        paper_id=paper_id,
        run_metadata=meta,
        data_root=data_root,
        attempt_id=attempt_id,
    )

    latest = json.loads(
        (data_root / "extracted" / paper_id / "latest_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert Path(latest["run_directory"]) == family_dir
    assert Path(latest["attempt_directory"]) == concrete_dir
    assert latest["run_id"] == meta["run_id"]
    assert latest["attempt_id"] == attempt_id
    assert resolve_run_directory(
        project_root=project_root,
        paper_id=paper_id,
        run_id=None,
        data_root=data_root,
    ) == concrete_dir.resolve()
    assert resolve_run_directory(
        project_root=project_root,
        paper_id=paper_id,
        run_id=meta["run_id"],
        data_root=data_root,
        attempt_id=attempt_id,
    ) == concrete_dir.resolve()


def test_resolve_run_directory_keeps_legacy_flat_run_support(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    data_root = tmp_path / "data"
    paper_id = "legacy_A"
    meta = _metadata("legacy-run")
    family_dir = run_directory(
        project_root,
        paper_id,
        meta["run_id"],
        data_root=data_root,
    )
    family_dir.mkdir(parents=True)
    write_json(family_dir / "active_chunks.json", {"run_id": meta["run_id"]})
    write_latest_run_pointer(
        project_root=project_root,
        paper_id=paper_id,
        run_metadata=meta,
        data_root=data_root,
    )

    assert resolve_run_directory(
        project_root=project_root,
        paper_id=paper_id,
        run_id=None,
        data_root=data_root,
    ) == family_dir.resolve()
    assert resolve_run_directory(
        project_root=project_root,
        paper_id=paper_id,
        run_id=meta["run_id"],
        data_root=data_root,
    ) == family_dir.resolve()


def _broad_pipeline(tmp_path: Path) -> BroadCorpusPilotPipeline:
    papers_yaml = tmp_path / "papers.yaml"
    papers_yaml.write_text(
        "version: 3\npapers:\n  broad_A: {}\n",
        encoding="utf-8",
    )
    return BroadCorpusPilotPipeline(
        project_root=tmp_path,
        papers_yaml=papers_yaml,
        corpus_id="attempt-provenance",
        options=BroadPilotOptions(data_root=str(tmp_path / "data_broad")),
        requested_paper_ids=["broad_A"],
    )








def test_bridge_fingerprint_ignores_attempt_identity(tmp_path: Path) -> None:
    attempt_a = tmp_path / "attempt-A"
    attempt_b = tmp_path / "attempt-B"
    attempt_a.mkdir()
    attempt_b.mkdir()
    strict_a = attempt_a / "chunk.json"
    strict_b = attempt_b / "chunk.json"
    source_a = attempt_a / "source.json"
    source_b = attempt_b / "source.json"
    strict_a.write_text('{"x": 1}', encoding="utf-8")
    strict_b.write_text('{"x": 1}', encoding="utf-8")
    source_a.write_text('{"source": "same"}', encoding="utf-8")
    source_b.write_text('{"source": "same"}', encoding="utf-8")
    implementation = tmp_path / "impl.py"
    implementation.write_text("x = 1\n", encoding="utf-8")

    common = {
        "run_id": "run-A",
        "run_fingerprint": "fp-A",
    }
    meta_a = compute_bridge_extraction_metadata(
        strict_run_dir=attempt_a,
        active_payload={**common, "attempt_id": "attempt-A"},
        model="model",
        provider="provider",
        strict_chunk_paths=[strict_a],
        source_chunk_paths=[source_a],
        implementation_paths=[implementation],
        runtime_options={"x": 1},
    )
    meta_b = compute_bridge_extraction_metadata(
        strict_run_dir=attempt_b,
        active_payload={**common, "attempt_id": "attempt-B"},
        model="model",
        provider="provider",
        strict_chunk_paths=[strict_b],
        source_chunk_paths=[source_b],
        implementation_paths=[implementation],
        runtime_options={"x": 1},
    )

    assert meta_a["bridge_extraction_fingerprint"] == meta_b[
        "bridge_extraction_fingerprint"
    ]
    assert meta_a["bridge_extraction_id"] == meta_b["bridge_extraction_id"]
    assert meta_a["strict_run_directory"] != meta_b["strict_run_directory"]
