from __future__ import annotations

import json
from pathlib import Path

from scripts.run_corpus_preprocessing import (
    _infer_materialization_id,
    _latest_m3_from_manifest,
    _next_round_dir,
)


def test_infer_materialization_id_prefers_incremental_report(tmp_path: Path) -> None:
    m4 = tmp_path / "m4"
    m4.mkdir()
    (m4 / "materialization_report.json").write_text(
        json.dumps({"materialization_id": "legacy"}), encoding="utf-8"
    )
    (m4 / "incremental_materialization_report.json").write_text(
        json.dumps({"materialization_id": "incremental"}), encoding="utf-8"
    )
    assert _infer_materialization_id(m4, None) == "incremental"
    assert _infer_materialization_id(m4, "explicit") == "explicit"


def test_latest_m3_manifest_requires_completed_snapshot(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    m3 = tmp_path / "m3"
    m3.mkdir()
    (run_root / "preprocess_run.json").write_text(
        json.dumps({"latest_m3_dir": str(m3)}), encoding="utf-8"
    )
    assert _latest_m3_from_manifest(run_root) is None
    (m3 / "acquisition_report.json").write_text("{}\n", encoding="utf-8")
    assert _latest_m3_from_manifest(run_root) == m3.resolve()


def test_next_round_is_monotonic(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "round_001").mkdir(parents=True)
    (run_root / "round_003").mkdir()
    (run_root / "round_misc").mkdir()
    index, path = _next_round_dir(run_root)
    assert index == 4
    assert path == run_root / "round_004"
