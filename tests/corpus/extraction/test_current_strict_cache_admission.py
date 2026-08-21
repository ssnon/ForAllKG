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


def _edge() -> dict:
    return {
        "source": "method",
        "relation": "USES_PRECURSOR",
        "target": "input",
        "evidence_type": "synthesis_procedure",
        "evidence_strength": "direct",
        "evidence_text": (
            "HAuCl4 was used to prepare the gold seed."
        ),
        "confidence": "high",
        "evidence_pointers": [
            {
                "document_id": "main",
                "document_role": "main",
                "page_id": None,
                "asset_ids": [],
                "locator_text": "Methods",
            }
        ],
        "subsection": "Methods",
    }


def _payload(
    *,
    target_type: str,
) -> dict:
    return {
        "paper_id": "P",
        "chunk_id": "P:main:c",
        "section": "Methods",
        "document_id": "main",
        "document_role": "main",
        "page_ids": [],
        "asset_ids": [],
        "entities": [
            {
                "id": "method",
                "type": "SynthesisMethod",
                "label": "Gold seed synthesis",
                "description": None,
            },
            {
                "id": "input",
                "type": target_type,
                "label": "HAuCl4 solution",
                "description": None,
            },
        ],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [
            _edge(),
        ],
    }


def _chunk() -> ChunkSpec:
    return ChunkSpec(
        paper_id="P",
        section="Methods",
        index=0,
        core_text=(
            "HAuCl4 was used to prepare the gold seed."
        ),
        left_context="",
        right_context="",
        chunk_id="P:main:c",
        document_id="main",
        document_role="main",
        page_ids=(),
        asset_ids=(),
    )


def _load(
    tmp_path,
    *,
    target_type: str,
):
    adapter = get_extraction_adapter(
        "sers_au_ag"
    )

    path = (
        tmp_path
        / f"{target_type}.json"
    )

    path.write_text(
        json.dumps(
            _payload(
                target_type=target_type,
            )
        ),
        encoding="utf-8",
    )

    return load_existing_result(
        chunk=_chunk(),
        output_path=path,
        relation_constraints=(
            adapter.strict_relation_constraints
        ),
        semantic_issue_collector=(
            adapter.strict_semantic_issue_collector
        ),
    )


def test_current_cache_rejects_historical_material_target(
    tmp_path,
):
    assert (
        _load(
            tmp_path,
            target_type="Material",
        )
        is None
    )


def test_current_cache_accepts_valid_precursor_target(
    tmp_path,
):
    result = _load(
        tmp_path,
        target_type="Precursor",
    )

    assert result is not None
    assert result.entities[1].type == "Precursor"
    assert (
        result.edges[0].relation
        == "USES_PRECURSOR"
    )
