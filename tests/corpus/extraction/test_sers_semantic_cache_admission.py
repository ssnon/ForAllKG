from __future__ import annotations

import json

from domains.extraction_registry import (
    get_extraction_adapter,
)
from pipeline_core.corpus.extraction.chunking import (
    ChunkSpec,
)
from scripts.corpus.strict_extraction_runtime import (
    load_existing_result,
)


def _chunk() -> ChunkSpec:
    return ChunkSpec(
        paper_id="P",
        section="Methods",
        index=0,
        core_text=(
            "500 uL of NaBH4 solution was "
            "added as a reductant."
        ),
        left_context="",
        right_context="",
        chunk_id="P:main:c",
        document_id="main",
        document_role="main",
        page_ids=(3,),
        asset_ids=(),
    )


def _payload() -> dict:
    return {
        "paper_id": "P",
        "chunk_id": "P:main:c",
        "section": "Methods",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [3],
        "asset_ids": [],
        "entities": [
            {
                "id": "method",
                "type": "SynthesisMethod",
                "label": "Gold seed synthesis",
                "description": "Borohydride reduction.",
            },
            {
                "id": "nabh4",
                "type": "Precursor",
                "label": "NaBH4",
                "description": (
                    "Aqueous reductant used in "
                    "the gold-seed synthesis."
                ),
            },
        ],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [
            {
                "source": "method",
                "relation": "USES_PRECURSOR",
                "target": "nabh4",
                "evidence_type": "synthesis_procedure",
                "evidence_strength": "direct",
                "evidence_text": (
                    "500 uL of NaBH4 solution was "
                    "added as a reductant."
                ),
                "confidence": "high",
                "evidence_pointers": [
                    {
                        "document_id": "main",
                        "document_role": "main",
                        "page_id": 3,
                        "asset_ids": [],
                        "locator_text": "Methods",
                    }
                ],
                "subsection": "Methods",
            }
        ],
    }


def test_endpoint_valid_semantic_invalid_cache_is_rejected(
    tmp_path,
):
    adapter = get_extraction_adapter(
        "sers_au_ag"
    )

    path = tmp_path / "cached.json"

    path.write_text(
        json.dumps(_payload()),
        encoding="utf-8",
    )

    endpoint_only = load_existing_result(
        chunk=_chunk(),
        output_path=path,
        relation_constraints=(
            adapter.strict_relation_constraints
        ),
        semantic_issue_collector=None,
    )

    assert endpoint_only is not None

    current = load_existing_result(
        chunk=_chunk(),
        output_path=path,
        relation_constraints=(
            adapter.strict_relation_constraints
        ),
        semantic_issue_collector=(
            adapter.strict_semantic_issue_collector
        ),
    )

    assert current is None
