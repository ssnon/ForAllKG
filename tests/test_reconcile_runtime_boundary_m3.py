from __future__ import annotations

import ast
from pathlib import Path

import dac_her.incremental_reconcile as legacy
import pipeline_core.reconcile_runtime as core


ROOT = Path(__file__).resolve().parents[1]


def test_reconcile_state_types_are_shared() -> None:
    assert legacy.ReconcileError is core.ReconcileError
    assert legacy.StageState is core.StageState
    assert legacy.StrictRunState is core.StrictRunState


def test_shared_reconcile_runtime_has_no_domain_dependency() -> None:
    path = (
        ROOT
        / "pipeline_core"
        / "reconcile_runtime.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )
    tree = ast.parse(
        source,
        filename=str(path),
    )

    forbidden = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""

            if module.startswith(
                (
                    "dac_her",
                    "domains",
                    "campaigns",
                )
            ):
                forbidden.append(module)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(
                    (
                        "dac_her",
                        "domains",
                        "campaigns",
                    )
                ):
                    forbidden.append(
                        alias.name
                    )

    assert forbidden == []
    assert "dac_her" not in source
    assert "data_dac" not in source
    assert "sers_au_ag" not in source


def test_logged_runner_dry_run_is_side_effect_free(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"

    assert core.run_logged_command(
        ["not-executed"],
        cwd=tmp_path,
        label="dry-run",
        log_dir=log_dir,
        heartbeat_seconds=0,
        dry_run=True,
    )

    assert not log_dir.exists()
