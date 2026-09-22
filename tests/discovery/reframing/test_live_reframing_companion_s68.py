from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.reframing.live_reframing_companion import (
    common_scope_args,
    extract_run_dir_from_e2e_args,
    resolve_live_reframing_paths,
    root_args,
    strip_remainder_separator,
)


def test_live_paths_use_fixed_e2e_artifact_contract(tmp_path: Path):
    paths = resolve_live_reframing_paths(tmp_path)
    assert paths.packet.name == "explorer.packet.json"
    assert paths.explorer_report.name == "explorer.report.json"
    assert paths.context.name == "hypothesis.context.json"
    assert paths.reframe_shadow.name == "scientific_reframing_shadow.live.json"
    assert paths.reasoning_portfolio.name == (
        "scientific_reframing_reasoning_portfolio.live.json"
    )


def test_root_args_requires_at_least_one_root():
    with pytest.raises(ValueError, match="at least one"):
        root_args([])


def test_root_args_preserves_multiple_roots(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    argv = root_args([first, second])
    assert argv == [
        "--root", str(first.resolve()),
        "--root", str(second.resolve()),
    ]


def test_missing_root_fails_closed(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        root_args([tmp_path / "missing"])


def test_common_scope_defaults_can_be_explicitly_frozen(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    argv = common_scope_args(
        roots=[root],
        duplicate_paper_policy="latest_attempt",
        cross_root_duplicate_policy="prefer_last_root",
    )
    flat = " ".join(argv)
    assert "--duplicate-paper-policy latest_attempt" in flat
    assert "--cross-root-duplicate-policy prefer_last_root" in flat


def test_extract_run_dir_supports_separate_value(tmp_path: Path):
    result = extract_run_dir_from_e2e_args(
        ["--run-dir", str(tmp_path), "--source", "x"]
    )
    assert result == tmp_path.resolve()


def test_extract_run_dir_supports_equals_form(tmp_path: Path):
    result = extract_run_dir_from_e2e_args(
        [f"--run-dir={tmp_path}", "--source", "x"]
    )
    assert result == tmp_path.resolve()


def test_extract_run_dir_requires_value():
    with pytest.raises(ValueError, match="requires a value"):
        extract_run_dir_from_e2e_args(["--run-dir"])


def test_extract_run_dir_requires_flag():
    with pytest.raises(ValueError, match="must include --run-dir"):
        extract_run_dir_from_e2e_args(["--source", "x"])


def test_remainder_separator_is_stripped_only_once():
    assert strip_remainder_separator(["--", "--run-dir", "/tmp/x"]) == [
        "--run-dir", "/tmp/x"
    ]
    assert strip_remainder_separator(["--run-dir", "/tmp/x"]) == [
        "--run-dir", "/tmp/x"
    ]
