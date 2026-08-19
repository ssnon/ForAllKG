from __future__ import annotations

import ast
import json
import os
from pathlib import Path

from pipeline_core.reconcile_recovery import (
    discover_strict_run_candidates,
    select_compatible_strict_run,
)
from pipeline_core.reconcile_runtime import (
    StageState,
    StrictRunState,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_shared_recovery_has_no_domain_or_quality_policy() -> None:
    path = (
        ROOT
        / "pipeline_core"
        / "reconcile_recovery.py"
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
        if isinstance(
            node,
            ast.ImportFrom,
        ):
            module = node.module or ""

            if module.startswith(
                (
                    "dac_her",
                    "domains",
                    "campaigns",
                )
            ):
                forbidden.append(
                    module
                )

        elif isinstance(
            node,
            ast.Import,
        ):
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

    assert "partial_critical" not in source
    assert '"rejected"' not in source
    assert "OPENROUTER" not in source

    # Pointer serialization identity stays outside.
    assert "recovered_by" not in source
    assert "incremental_reconciler" not in source


def test_candidate_discovery_prefers_pointer_then_mtime(
    tmp_path: Path,
) -> None:
    paper_root = (
        tmp_path / "P1"
    )

    runs = (
        paper_root / "runs"
    )

    old = runs / "run-old"
    mid = runs / "run-mid"
    new = runs / "run-new"

    for path in (
        old,
        mid,
        new,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    os.utime(
        old,
        ns=(
            1_000_000_000,
            1_000_000_000,
        ),
    )

    os.utime(
        mid,
        ns=(
            2_000_000_000,
            2_000_000_000,
        ),
    )

    os.utime(
        new,
        ns=(
            3_000_000_000,
            3_000_000_000,
        ),
    )

    candidates, reason = (
        discover_strict_run_candidates(
            paper_root=paper_root
        )
    )

    assert reason == (
        "no latest strict run"
    )

    assert [
        path.name
        for path in candidates
    ] == [
        "run-new",
        "run-mid",
        "run-old",
    ]

    _write_json(
        paper_root
        / "latest_run.json",
        {
            "paper_id": "P1",
            "run_id": "run-old",
        },
    )

    candidates, reason = (
        discover_strict_run_candidates(
            paper_root=paper_root
        )
    )

    assert reason == (
        "latest pointer=run-old"
    )

    assert [
        path.name
        for path in candidates
    ] == [
        "run-old",
        "run-new",
        "run-mid",
    ]


def test_selection_preserves_recovery_reason() -> None:
    first = Path(
        "/tmp/run-first"
    )

    second = Path(
        "/tmp/run-second"
    )

    def validate(
        run_dir: Path,
    ) -> StrictRunState:
        if run_dir == first:
            return StrictRunState(
                StageState.pending(
                    "bad-run-first"
                ),
                first.name,
                first,
            )

        return StrictRunState(
            StageState.ready(
                "semantic contract matches",
                second,
                {
                    "run_id":
                        second.name,
                },
            ),
            second.name,
            second,
        )

    state = (
        select_compatible_strict_run(
            candidates=[
                first,
                second,
            ],
            pointer_reason=(
                "latest pointer="
                "run-first"
            ),
            validate_run=validate,
        )
    )

    assert state.stage.valid
    assert state.run_id == (
        "run-second"
    )

    assert state.stage.reason == (
        "recovered compatible "
        "usable strict run "
        "from runs/*; "
        "latest pointer=run-first; "
        "selected=run-second"
    )
