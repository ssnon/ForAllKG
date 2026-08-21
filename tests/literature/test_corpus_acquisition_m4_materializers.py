from __future__ import annotations

from pathlib import Path

from pipeline_core.literature.acquisition.materialization_contracts import (
    MaterializationPolicy,
)
from pipeline_core.literature.acquisition.materializers import (
    CsvMaterializer,
    TextMaterializer,
    materializer_for,
)


def test_text_materializer(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("hello\nworld", encoding="utf-8")
    output = TextMaterializer().materialize(
        source_path=source,
        policy=MaterializationPolicy(policy_id="p"),
    )
    assert "hello" in output.markdown
    assert output.asset_source_dir is None


def test_csv_materializer_produces_markdown_table(tmp_path):
    source = tmp_path / "table.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    output = CsvMaterializer().materialize(
        source_path=source,
        policy=MaterializationPolicy(policy_id="p"),
    )
    assert "| a | b |" in output.markdown
    assert "| 1 | 2 |" in output.markdown


def test_zip_is_explicitly_unsupported():
    source = Path("/tmp/example.zip")
    assert materializer_for(
        source,
        MaterializationPolicy(policy_id="p"),
    ) is None
