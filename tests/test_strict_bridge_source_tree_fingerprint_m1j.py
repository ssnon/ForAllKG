from __future__ import annotations

from pathlib import Path

from pipeline_core.strict_bridge_corpus_pipeline import (
    _sha256_source_tree,
)


def _write(
    root: Path,
    relative: str,
    content: str,
) -> Path:
    path = root / relative
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        content,
        encoding="utf-8",
    )
    return path


def _minimal_source_tree(
    root: Path,
) -> None:
    _write(
        root,
        "dac_her/example.py",
        "VALUE = 'dac-v1'\n",
    )

    _write(
        root,
        "pipeline_core/example.py",
        "VALUE = 'core-v1'\n",
    )

    _write(
        root,
        "scripts/extract_paper.py",
        "VALUE = 'script-v1'\n",
    )


def test_current_source_tree_hash_tracks_dac_her_python_changes(
    tmp_path: Path,
):
    _minimal_source_tree(tmp_path)

    baseline = _sha256_source_tree(
        tmp_path
    )

    _write(
        tmp_path,
        "dac_her/example.py",
        "VALUE = 'dac-v2'\n",
    )

    changed = _sha256_source_tree(
        tmp_path
    )

    assert changed != baseline


def test_current_source_tree_hash_tracks_selected_script_changes(
    tmp_path: Path,
):
    _minimal_source_tree(tmp_path)

    baseline = _sha256_source_tree(
        tmp_path
    )

    _write(
        tmp_path,
        "scripts/extract_paper.py",
        "VALUE = 'script-v2'\n",
    )

    changed = _sha256_source_tree(
        tmp_path
    )

    assert changed != baseline


def test_current_source_tree_hash_ignores_pipeline_core_python_changes(
    tmp_path: Path,
):
    _minimal_source_tree(tmp_path)

    baseline = _sha256_source_tree(
        tmp_path
    )

    _write(
        tmp_path,
        "pipeline_core/example.py",
        "VALUE = 'core-v2'\n",
    )

    changed = _sha256_source_tree(
        tmp_path
    )

    # Characterization of the current blind spot:
    # shared implementation moved under pipeline_core
    # does not currently participate in orchestration
    # resume fingerprinting.
    assert changed == baseline
