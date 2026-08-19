from __future__ import annotations

import ast
import json
from pathlib import Path

import pipeline_core.reconcile_validation as core


ROOT = Path(__file__).resolve().parents[1]


def _write_json(
    path: Path,
    value: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def test_shared_validator_has_no_domain_or_quality_policy() -> None:
    path = (
        ROOT
        / "pipeline_core"
        / "reconcile_validation.py"
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

    # Quality meanings remain an application policy.
    assert "partial_critical" not in source
    assert '"rejected"' not in source


def test_shared_validator_uses_application_admission_callback(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "run-a"
    )

    _write_json(
        run_dir / "run.json",
        {
            "run_id": "run-a",
        },
    )

    _write_json(
        run_dir
        / "active_chunks.json",
        {
            "run_id": "run-a",
            "chunks": [
                {
                    "chunk_id": "c1",
                }
            ],
            "application_status":
                "custom-block",
        },
    )

    state = core.validate_strict_run(
        run_dir=run_dir,
        current={},
        active_payload_issue=(
            lambda active:
                (
                    "application rejected run"
                    if active.get(
                        "application_status"
                    )
                    == "custom-block"
                    else None
                )
        ),
        compatibility_check=(
            lambda run_meta, current:
                (
                    True,
                    "contract matches",
                )
        ),
    )

    assert not state.stage.valid
    assert (
        state.stage.reason
        == "application rejected run"
    )


def test_shared_validator_preserves_compatibility_metadata(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "run-b"
    )

    meta = {
        "run_id": "run-b",
        "marker": "meta",
    }

    _write_json(
        run_dir / "run.json",
        meta,
    )

    _write_json(
        run_dir
        / "active_chunks.json",
        {
            "run_id": "run-b",
            "chunks": [
                {
                    "chunk_id": "c1",
                }
            ],
        },
    )

    state = core.validate_strict_run(
        run_dir=run_dir,
        current={},
        active_payload_issue=(
            lambda active: None
        ),
        compatibility_check=(
            lambda run_meta, current:
                (
                    False,
                    "contract changed",
                )
        ),
    )

    assert not state.stage.valid
    assert (
        state.stage.reason
        == "contract changed"
    )
    assert (
        state.stage.metadata
        == meta
    )
