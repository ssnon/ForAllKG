from __future__ import annotations

import inspect
from pathlib import Path

from dac_her.alpha4c4d2_holdout_support import (
    verify_strict_source_unchanged,
)
from scripts.prepare_sers_alpha4c4d2_holdout_inputs import copy_if_exists


def test_locator_is_not_an_immutable_strict_source_gate():
    source = inspect.getsource(verify_strict_source_unchanged)
    assert "locator_index appeared" not in source
    assert "locator_index disappeared" not in source
    assert "latest_run" in source
    assert "active_chunks" in source
    assert "chunk_inputs" in source


def test_resume_safe_snapshot_does_not_overwrite(tmp_path: Path):
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("new", encoding="utf-8")
    dst.write_text("original", encoding="utf-8")

    copy_if_exists(src, dst)

    assert dst.read_text(encoding="utf-8") == "original"
