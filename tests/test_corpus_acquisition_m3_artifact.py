from __future__ import annotations

from pathlib import Path

from dac_her.corpus_acquisition.artifact_acquisition import (
    _pdf_magic,
    _safe_work_dir,
)


def test_pdf_magic():
    assert _pdf_magic(b"%PDF-") is True
    assert _pdf_magic(b"<html") is False


def test_safe_work_dir_is_stable_and_sanitized():
    left = _safe_work_dir("catalog_work:abc/def")
    right = _safe_work_dir("catalog_work:abc/def")
    assert left == right
    assert "/" not in left
    assert ":" not in left
