from __future__ import annotations

import json
from pathlib import Path

from dac_her.broad_binding_sidecar import (
    binding_from_attempt,
    build_bound_diagnostics_view,
    load_bindings,
    write_bindings,
)


def _write_attempt(root: Path, paper: str, run: str, attempt: str) -> Path:
    path = root / "extracted" / paper / "runs" / run / "attempts" / attempt
    path.mkdir(parents=True)
    (path / "active_chunks.json").write_text(
        json.dumps({
            "paper_id": paper,
            "run_id": run,
            "run_fingerprint": f"fp-{run}",
            "attempt_id": attempt,
            "graph_materialization_status": "complete",
            "chunks": [],
            "quarantined_chunks": [],
            "failed_chunks": [],
        }),
        encoding="utf-8",
    )
    (path / "summary.json").write_text(
        json.dumps({
            "paper_id": paper,
            "run_id": run,
            "run_fingerprint": f"fp-{run}",
            "attempt_id": attempt,
            "graph_materialization_status": "complete",
        }),
        encoding="utf-8",
    )
    return path


def test_binding_roundtrip_and_view_points_to_exact_attempt(tmp_path: Path):
    data_root = tmp_path / "data"
    attempt = _write_attempt(data_root, "broad_A", "run-full", "a1")
    binding = binding_from_attempt(data_root, "broad_A", "run-full", "a1")
    sidecar = write_bindings(
        tmp_path / "bindings.json",
        corpus_id="c1",
        data_root=data_root,
        bindings={"broad_A": binding},
    )
    _, loaded = load_bindings(sidecar)
    view = tmp_path / "view"
    missing = build_bound_diagnostics_view(
        view_root=view,
        data_root=data_root,
        paper_ids=["broad_A"],
        bindings=loaded,
    )
    assert missing == []
    pointer = json.loads(
        (view / "extracted" / "broad_A" / "latest_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert Path(pointer["attempt_directory"]) == attempt.resolve()
    assert pointer["run_id"] == "run-full"
    assert pointer["attempt_id"] == "a1"


def test_missing_binding_never_falls_back_to_source_latest(tmp_path: Path):
    data_root = tmp_path / "data"
    attempt = _write_attempt(data_root, "broad_A", "run-latest", "latest-a")
    paper_root = data_root / "extracted" / "broad_A"
    (paper_root / "latest_run.json").write_text(
        json.dumps({
            "run_id": "run-latest",
            "attempt_id": "latest-a",
            "run_directory": str(attempt.parent.parent),
            "attempt_directory": str(attempt),
        }),
        encoding="utf-8",
    )
    view = tmp_path / "view"
    missing = build_bound_diagnostics_view(
        view_root=view,
        data_root=data_root,
        paper_ids=["broad_A"],
        bindings={},
    )
    assert missing == ["broad_A"]
    assert not (view / "extracted" / "broad_A" / "latest_run.json").exists()
