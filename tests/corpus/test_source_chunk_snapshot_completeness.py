from __future__ import annotations

import json

from pipeline_core.corpus.extraction.chunking import (
    ChunkSpec,
)
from scripts.corpus.extract_paper import (
    write_source_chunk,
)


def test_source_chunk_snapshot_preserves_rechunk_asset_state(
    tmp_path,
):
    chunk = ChunkSpec(
        paper_id="P",
        section="Methods",
        index=0,
        core_text=(
            "Preparation is shown in Figure 1. "
            "![](_page_3_Diagram_2.jpeg)"
        ),
        left_context="left",
        right_context="right",
        chunk_id="P:main:c",
        document_id="main",
        document_role="main",
        page_ids=(3,),
        asset_ids=("P:main:asset:1",),
        asset_paths=(
            "_page_3_Diagram_2.jpeg",
        ),
        asset_pages=(3,),
        asset_locators=("Figure 1",),
        asset_context=(
            "ASSET_ID: P:main:asset:1\n"
            "TYPE: figure\n"
            "PAGE_ID: 3\n"
            "CAPTION: Figure 1. Preparation."
        ),
        split_depth=1,
    )

    path = write_source_chunk(
        tmp_path,
        chunk,
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["asset_ids"] == [
        "P:main:asset:1"
    ]

    assert payload["asset_paths"] == [
        "_page_3_Diagram_2.jpeg"
    ]

    assert payload["asset_pages"] == [
        3
    ]

    assert payload["asset_locators"] == [
        "Figure 1"
    ]

    assert (
        payload["asset_context"]
        == chunk.asset_context
    )
