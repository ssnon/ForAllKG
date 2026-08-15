from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_sers_alpha4c4b1_trend_holdout import (
    FrozenTrendHoldoutError,
    _require_equal,
    _resolution_decisions_snapshot,
)


def test_preexisting_resolution_decisions_are_snapshotted_not_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.run_sers_alpha4c4b1_trend_holdout as runner

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    paper_root = tmp_path / "data_sers" / "extracted" / "P"
    path = paper_root / "resolution" / "decisions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")

    snap = _resolution_decisions_snapshot(paper_root)
    assert snap["present"] is True
    assert snap["nonempty_line_count"] == 2
    assert len(snap["sha256"]) == 64


def test_missing_resolution_file_is_explicitly_snapshotted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import scripts.run_sers_alpha4c4b1_trend_holdout as runner

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    paper_root = tmp_path / "data_sers" / "extracted" / "P"
    paper_root.mkdir(parents=True)

    snap = _resolution_decisions_snapshot(paper_root)
    assert snap["present"] is False
    assert snap["sha256"] == ""
    assert snap["nonempty_line_count"] == 0


def test_protocol_forbids_post_lock_resolution_changes():
    payload = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4b1_trend_holdout_run.json"
        ).read_text(encoding="utf-8")
    )
    policy = payload["canonical_input_policy"]
    assert policy["preexisting_resolution_decisions_allowed"] is True
    assert policy["post_snapshot_resolution_changes_allowed"] is False
    assert (
        policy["resolution_decisions_hash_locked_at_campaign_start"]
        is True
    )
    assert (
        policy["first_successful_preflight_creates_persistent_input_lock"]
        is True
    )


def test_scientific_split_and_semantics_are_unchanged():
    old = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4b_trend_holdout_run.json"
        ).read_text(encoding="utf-8")
    )
    new = json.loads(
        Path(
            "configs/heldout/"
            "sers_alpha4c4b1_trend_holdout_run.json"
        ).read_text(encoding="utf-8")
    )
    assert new["holdout_papers"] == old["holdout_papers"]
    assert new["future_reserve_papers"] == old["future_reserve_papers"]
    assert new["source_split_protocol"] == old["source_split_protocol"]
    assert new["frozen_semantics"] == old["frozen_semantics"]
    assert new["acceptance_policy"]["count_thresholds_used"] is False


def test_fail_closed_helper_still_fails():
    with pytest.raises(FrozenTrendHoldoutError):
        _require_equal("x", "left", "right")
