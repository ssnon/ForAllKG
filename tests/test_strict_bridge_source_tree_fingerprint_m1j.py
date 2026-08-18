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


def test_source_tree_hash_tracks_pipeline_core_python_changes(
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

    assert changed != baseline

def test_source_tree_hash_ignores_non_python_tree_files(
    tmp_path: Path,
):
    _minimal_source_tree(tmp_path)

    baseline = _sha256_source_tree(
        tmp_path
    )

    _write(
        tmp_path,
        "pipeline_core/notes.txt",
        "not implementation python\n",
    )

    _write(
        tmp_path,
        "dac_her/cache.json",
        "{\"value\": 1}\n",
    )

    changed = _sha256_source_tree(
        tmp_path
    )

    assert changed == baseline


def test_source_tree_hash_is_independent_of_file_creation_order(
    tmp_path: Path,
):
    left = tmp_path / "left"
    right = tmp_path / "right"

    files = (
        (
            "dac_her/example.py",
            "VALUE = 'dac'\n",
        ),
        (
            "pipeline_core/example.py",
            "VALUE = 'core'\n",
        ),
        (
            "pipeline_core/nested/runtime.py",
            "VALUE = 'nested'\n",
        ),
        (
            "scripts/extract_paper.py",
            "VALUE = 'script'\n",
        ),
    )

    for relative, content in files:
        _write(
            left,
            relative,
            content,
        )

    for relative, content in reversed(files):
        _write(
            right,
            relative,
            content,
        )

    assert (
        _sha256_source_tree(left)
        == _sha256_source_tree(right)
    )
