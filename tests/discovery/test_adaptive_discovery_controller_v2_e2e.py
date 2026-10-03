from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.discovery.run_adaptive_discovery_controller_v2_shadow import (
    _validate_initial_local_controller_dir,
)
from scripts.discovery.run_dac_discovery_e2e import (
    parse_args,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def test_v2_accepts_complete_stage776_reuse_directory(
    tmp_path: Path,
) -> None:
    for name in (
        "adaptive_controller.summary.json",
        "adaptive_effective.portfolio.json",
        "graph_retraversal.handoff.json",
        "adaptive_search.history.json",
    ):
        _write(tmp_path / name, {})

    _validate_initial_local_controller_dir(tmp_path)


def test_v2_rejects_incomplete_stage776_reuse_directory(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "adaptive_controller.summary.json",
        {},
    )

    with pytest.raises(
        RuntimeError,
        match="complete Stage-7.76 artifact set",
    ):
        _validate_initial_local_controller_dir(
            tmp_path
        )


def test_stage777_flag_auto_enables_stage776_and_stage775(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    argv = [
        "run_dac_discovery_e2e.py",
        "--run-dir",
        str(tmp_path / "run"),
        "--source",
        "source",
        "--target",
        "target",
        "--question",
        "question",
        "--adaptive-graph-retraversal-shadow",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = parse_args()

    assert args.adaptive_graph_retraversal_shadow is True
    assert args.adaptive_discovery_controller_shadow is True
    assert args.scientific_portfolio_closed_loop_shadow is True
    assert args.scientific_portfolio_verification_shadow is True
    assert args.scientific_portfolio_selection_shadow is True


def test_stage777_invalid_selection_budget_fails_early(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    argv = [
        "run_dac_discovery_e2e.py",
        "--run-dir",
        str(tmp_path / "run"),
        "--source",
        "source",
        "--target",
        "target",
        "--question",
        "question",
        "--adaptive-graph-retraversal-shadow",
        "--adaptive-graph-candidate-top-k",
        "4",
        "--adaptive-graph-selected-top-k",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        parse_args()


def test_e2e_source_contains_exact_stage777_reuse_contract() -> None:
    source = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "scripts/discovery/run_dac_discovery_e2e.py"
    ).read_text(encoding="utf-8")

    assert "[7.77/13] Adaptive grounded context expansion shadow" in source
    assert "--initial-local-controller-dir" in source
    assert "stage_7_76_reused_as_epoch_0" in source
    assert '"stage8_input_changed": False' in source
    assert '"production_selection_changed": False' in source
