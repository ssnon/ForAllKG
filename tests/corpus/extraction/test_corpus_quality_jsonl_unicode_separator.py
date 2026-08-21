import json

import pytest
from pydantic import BaseModel

from scripts.corpus.apply_corpus_quality_gate import (
    _read_jsonl,
)


class _Row(BaseModel):
    text: str


@pytest.mark.parametrize(
    "separator",
    [
        "\u0085",
        "\u2028",
        "\u2029",
    ],
)
def test_jsonl_reader_preserves_unicode_line_separator(
    tmp_path,
    separator,
):
    path = tmp_path / "rows.jsonl"

    rows = [
        {"text": f"before{separator}after"},
        {"text": "second row"},
    ]

    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    observed = _read_jsonl(
        path,
        _Row,
    )

    assert [
        row.text
        for row in observed
    ] == [
        f"before{separator}after",
        "second row",
    ]


def test_splitlines_would_corrupt_same_valid_jsonl(
    tmp_path,
):
    path = tmp_path / "rows.jsonl"

    path.write_text(
        json.dumps(
            {
                "text": "before\u2028after",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    text = path.read_text(
        encoding="utf-8"
    )

    assert len(text.split("\n")) == 2
    assert len(text.splitlines()) == 2

    with pytest.raises(
        json.JSONDecodeError
    ):
        json.loads(
            text.splitlines()[0]
        )
