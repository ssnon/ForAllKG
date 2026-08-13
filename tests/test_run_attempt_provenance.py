from __future__ import annotations

import json
from pathlib import Path

from dac_her.bridge_run_state import compute_bridge_extraction_metadata
from dac_her.broad_corpus_pipeline import (
    BroadCorpusPilotPipeline,
    BroadPilotOptions,
)
from dac_her.broad_extraction_diagnostics import _latest_run_dir
import dac_her.llm_telemetry_backfill as telemetry_backfill
from dac_her.run_state import (
    attempt_directory,
    resolve_run_directory,
    run_directory,
    write_json,
    write_latest_attempt_pointer,
    write_latest_run_pointer,
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


def test_broad_freshness_binds_new_artifacts_to_attempt_id(tmp_path: Path) -> None:
    pipeline = _broad_pipeline(tmp_path)
    paper_root = pipeline._paper_root("broad_A")
    family_dir = paper_root / "runs" / "run-A"
    attempt_dir = family_dir / "attempts" / "attempt-A"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "active_chunks.json").write_text(
        json.dumps(
            {
                "paper_id": "broad_A",
                "run_id": "run-A",
                "run_fingerprint": "fp-A",
                "attempt_id": "attempt-A",
                "graph_materialization_status": "complete",
            }
        ),
        encoding="utf-8",
    )
    (attempt_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-A",
                "run_fingerprint": "fp-A",
                "attempt_id": "attempt-A",
            }
        ),
        encoding="utf-8",
    )
    paper_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "latest_run.json").write_text(
        json.dumps(
            {
                "paper_id": "broad_A",
                "run_id": "run-A",
                "run_fingerprint": "fp-A",
                "run_directory": str(family_dir),
                "attempt_id": "attempt-A",
                "attempt_directory": str(attempt_dir),
            }
        ),
        encoding="utf-8",
    )

    identity = pipeline._latest_extraction_identity("broad_A")
    assert identity is not None
    assert identity["attempt_id"] == "attempt-A"
    assert Path(identity["run_directory"]) == attempt_dir
    assert pipeline._identity_matches(identity, "run-A", "fp-A", "attempt-A")
    assert not pipeline._identity_matches(identity, "run-A", "fp-A", "attempt-B")

    import networkx as nx

    graph_path = pipeline._expected_output("broad_A", "paper_graph")
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(
        nx.MultiDiGraph(
            run_id="run-A",
            run_fingerprint="fp-A",
            source_extraction_attempt_id="attempt-B",
        ),
        graph_path,
    )
    assert not pipeline._paper_graph_matches_latest_extraction("broad_A")
    nx.write_graphml(
        nx.MultiDiGraph(
            run_id="run-A",
            run_fingerprint="fp-A",
            source_extraction_attempt_id="attempt-A",
        ),
        graph_path,
    )
    assert pipeline._paper_graph_matches_latest_extraction("broad_A")

    projection_summary = (
        paper_root / "graphagents" / "mechanism" / "summary.json"
    )
    projection_summary.parent.mkdir(parents=True, exist_ok=True)
    projection_summary.write_text(
        json.dumps(
            {
                "source_extraction_run_id": "run-A",
                "source_extraction_run_fingerprint": "fp-A",
                "source_extraction_attempt_id": "attempt-B",
            }
        ),
        encoding="utf-8",
    )
    assert not pipeline._projection_matches_latest_extraction("broad_A")
    projection_summary.write_text(
        json.dumps(
            {
                "source_extraction_run_id": "run-A",
                "source_extraction_run_fingerprint": "fp-A",
                "source_extraction_attempt_id": "attempt-A",
            }
        ),
        encoding="utf-8",
    )
    assert pipeline._projection_matches_latest_extraction("broad_A")

    legacy_identity = {
        "run_id": "run-A",
        "run_fingerprint": "fp-A",
    }
    assert pipeline._identity_matches(legacy_identity, "run-A", "fp-A", "")


def test_broad_diagnostics_prefers_attempt_directory(tmp_path: Path) -> None:
    paper_root = tmp_path / "extracted" / "broad_A"
    family_dir = paper_root / "runs" / "run-A"
    attempt_dir = family_dir / "attempts" / "attempt-A"
    attempt_dir.mkdir(parents=True)
    paper_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "latest_run.json").write_text(
        json.dumps(
            {
                "run_directory": str(family_dir),
                "attempt_directory": str(attempt_dir),
            }
        ),
        encoding="utf-8",
    )
    assert _latest_run_dir(paper_root) == attempt_dir



def test_telemetry_backfill_reads_latest_attempt_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data_broad"
    paper_root = data_root / "extracted" / "broad_A"
    family_dir = paper_root / "runs" / "run-A"
    attempt_dir = family_dir / "attempts" / "attempt-A"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "active_chunks.json").write_text(
        json.dumps(
            {
                "paper_id": "broad_A",
                "run_id": "run-A",
                "attempt_id": "attempt-A",
                "graph_materialization_status": "complete",
                "chunks": [],
                "quarantined_chunks": [],
                "failed_chunks": [],
            }
        ),
        encoding="utf-8",
    )
    paper_root.mkdir(parents=True, exist_ok=True)
    (paper_root / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": "run-A",
                "run_fingerprint": "fp-A",
                "run_directory": str(family_dir),
                "attempt_id": "attempt-A",
                "attempt_directory": str(attempt_dir),
            }
        ),
        encoding="utf-8",
    )
    telemetry_path = tmp_path / "telemetry.jsonl"
    telemetry_path.write_text(
        json.dumps({"record_type": "call", "call_id": "call-A"}) + "\n",
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_append(path, **kwargs):
        seen["path"] = path
        seen.update(kwargs)
        return 1

    monkeypatch.setattr(
        telemetry_backfill,
        "append_extraction_artifact_resolutions",
        fake_append,
    )
    result = telemetry_backfill.backfill_extraction_resolutions(
        data_root=data_root,
        paper_ids=["broad_A"],
        telemetry_path=telemetry_path,
    )

    assert result["missing_runs"] == []
    assert result["papers_resolved"] == 1
    assert seen["run_id"] == "run-A"
    assert seen["paper_id"] == "broad_A"

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
