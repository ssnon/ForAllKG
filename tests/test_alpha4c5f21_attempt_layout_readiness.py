from __future__ import annotations

import json
from pathlib import Path

from dac_her.alpha4c5f2_strict_source import (
    STRICT_SOURCE_LAYOUT_SEMANTICS_ID,
    _resolve_concrete_run_directory,
)


def test_attempt_directory_from_latest_run_pointer(tmp_path: Path):
    family = tmp_path / "runs" / "abc"
    attempt = family / "attempts" / "attempt-1"
    attempt.mkdir(parents=True)

    pointer = {
        "paper_id": "P1",
        "run_id": "abc",
        "run_directory": str(family),
        "attempt_id": "attempt-1",
        "attempt_directory": str(attempt),
    }
    resolved, latest = _resolve_concrete_run_directory(
        paper_id="P1",
        pointer=pointer,
        family_dir=family,
    )
    assert resolved == attempt.resolve()
    assert latest is None


def test_attempt_directory_from_latest_attempt_pointer(tmp_path: Path):
    family = tmp_path / "runs" / "abc"
    attempt = family / "attempts" / "attempt-2"
    attempt.mkdir(parents=True)
    (family / "latest_attempt.json").write_text(
        json.dumps(
            {
                "paper_id": "P1",
                "run_id": "abc",
                "attempt_id": "attempt-2",
                "attempt_directory": str(attempt),
            }
        ),
        encoding="utf-8",
    )

    resolved, latest = _resolve_concrete_run_directory(
        paper_id="P1",
        pointer={
            "paper_id": "P1",
            "run_id": "abc",
            "run_directory": str(family),
        },
        family_dir=family,
    )
    assert resolved == attempt.resolve()
    assert latest["attempt_id"] == "attempt-2"


def test_legacy_flat_layout_falls_back_to_family(tmp_path: Path):
    family = tmp_path / "runs" / "legacy"
    family.mkdir(parents=True)

    resolved, latest = _resolve_concrete_run_directory(
        paper_id="P1",
        pointer={
            "paper_id": "P1",
            "run_id": "legacy",
            "run_directory": str(family),
        },
        family_dir=family,
    )
    assert resolved == family
    assert latest is None


def test_layout_semantics_is_explicit():
    assert (
        STRICT_SOURCE_LAYOUT_SEMANTICS_ID
        == "strict_source_attempt_layout_v1_alpha4c5f21"
    )
