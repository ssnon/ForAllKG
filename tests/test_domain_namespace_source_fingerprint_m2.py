from __future__ import annotations

from pathlib import Path

from pipeline_core.strict_bridge_corpus_pipeline import (
    _sha256_source_tree,
)


def _write(
    root: Path,
    relative: str,
    content: str,
) -> None:
    path = root / relative
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        content,
        encoding="utf-8",
    )


def test_source_tree_hash_tracks_domain_namespace_python_changes(
    tmp_path: Path,
):
    _write(
        tmp_path,
        "dac_her/example.py",
        "VALUE = 'legacy'\n",
    )
    _write(
        tmp_path,
        "pipeline_core/example.py",
        "VALUE = 'core'\n",
    )
    _write(
        tmp_path,
        "domains/dac_her/profile.py",
        "VALUE = 'before'\n",
    )

    baseline = _sha256_source_tree(
        tmp_path
    )

    _write(
        tmp_path,
        "domains/dac_her/profile.py",
        "VALUE = 'after'\n",
    )

    changed = _sha256_source_tree(
        tmp_path
    )

    assert changed != baseline


def test_source_tree_hash_ignores_domain_namespace_non_python_changes(
    tmp_path: Path,
):
    _write(
        tmp_path,
        "dac_her/example.py",
        "VALUE = 'legacy'\n",
    )
    _write(
        tmp_path,
        "pipeline_core/example.py",
        "VALUE = 'core'\n",
    )
    _write(
        tmp_path,
        "domains/dac_her/profile.py",
        "VALUE = 'domain'\n",
    )
    _write(
        tmp_path,
        "domains/dac_her/notes.txt",
        "before\n",
    )

    baseline = _sha256_source_tree(
        tmp_path
    )

    _write(
        tmp_path,
        "domains/dac_her/notes.txt",
        "after\n",
    )

    changed = _sha256_source_tree(
        tmp_path
    )

    assert changed == baseline


def test_empty_domain_namespace_does_not_change_source_tree_hash(
    tmp_path: Path,
):
    _write(
        tmp_path,
        "dac_her/example.py",
        "VALUE = 'legacy'\n",
    )
    _write(
        tmp_path,
        "pipeline_core/example.py",
        "VALUE = 'core'\n",
    )

    baseline = _sha256_source_tree(
        tmp_path
    )

    (
        tmp_path
        / "domains"
        / "dac_her"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    changed = _sha256_source_tree(
        tmp_path
    )

    assert changed == baseline
