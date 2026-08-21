from __future__ import annotations

import pytest

from pipeline_core.literature.acquisition.supplementary_acquisition import _detect_magic, _validated_extension


def test_detect_pdf_and_zip_magic():
    assert _detect_magic(b"%PDF-1.7") == "pdf"
    assert _detect_magic(b"PK\x03\x04abc") == "zip"


def test_html_is_rejected():
    with pytest.raises(RuntimeError):
        _validated_extension(
            url="https://example.org/supplement",
            content_type="text/html",
            prefix=b"<html>",
        )


def test_pdf_requires_pdf_magic():
    with pytest.raises(RuntimeError):
        _validated_extension(
            url="https://example.org/supp.pdf",
            content_type="application/pdf",
            prefix=b"notpdf",
        )


def test_xlsx_zip_magic_is_accepted():
    assert (
        _validated_extension(
            url="https://example.org/supp.xlsx",
            content_type=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            prefix=b"PK\x03\x04abc",
        )
        == ".xlsx"
    )
