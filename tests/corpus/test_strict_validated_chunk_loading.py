from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from pipeline_core.corpus.schemas import KnowledgeGraph
from pipeline_core.corpus.extraction.chunking import ChunkSpec
from scripts.corpus.strict_extraction_runtime import load_existing_result
from pipeline_core.corpus.graph.strict_chunk_loading import (
    load_strict_validated_chunk_graph,
)


def _edge(
    source: str,
    relation: str,
    target: str,
) -> dict:
    return {
        "source": source,
        "relation": relation,
        "target": target,
        "evidence_type": "synthesis_procedure",
        "evidence_strength": "direct",
        "evidence_text": "Source-grounded synthesis input.",
        "confidence": "high",
        "evidence_pointers": [{
            "document_id": "main",
            "document_role": "main",
            "page_id": None,
            "asset_ids": [],
            "locator_text": "Methods",
        }],
        "subsection": "Methods",
    }


def _payload(
    *,
    target_id: str = "material",
) -> dict:
    return {
        "paper_id": "sers-context-regression",
        "chunk_id": "chunk-1",
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
                "id": "material",
                "type": "Material",
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
            _edge(
                "method",
                "USES_PRECURSOR",
                target_id,
            )
        ],
    }


def test_default_reload_reproduces_legacy_dac_relation_failure(
    tmp_path,
):
    payload = _payload()
    raw = json.dumps(payload)

    # Historical no-context loading activates legacy DAC
    # USES_PRECURSOR endpoint validation.
    with pytest.raises(ValidationError):
        KnowledgeGraph.model_validate_json(raw)


def test_strict_validated_chunk_reload_skips_legacy_dac_relation_policy(
    tmp_path,
):
    path = tmp_path / "chunk.json"
    path.write_text(
        json.dumps(_payload()),
        encoding="utf-8",
    )

    graph = load_strict_validated_chunk_graph(
        path
    )

    assert graph.paper_id == "sers-context-regression"
    assert graph.edges[0].relation == "USES_PRECURSOR"
    assert graph.entities[1].type == "Material"


def test_strict_validated_chunk_reload_keeps_structural_integrity_checks(
    tmp_path,
):
    path = tmp_path / "chunk.json"
    path.write_text(
        json.dumps(
            _payload(
                target_id="missing-node",
            )
        ),
        encoding="utf-8",
    )

    # Only legacy relation semantics are bypassed.
    # Undefined endpoints must still fail.
    with pytest.raises(ValidationError):
        load_strict_validated_chunk_graph(
            path
        )



def test_cached_strict_result_uses_shared_validated_loader(
    tmp_path,
):
    path = tmp_path / "chunk.json"
    path.write_text(
        json.dumps(_payload()),
        encoding="utf-8",
    )

    chunk = ChunkSpec(
        paper_id="sers-context-regression",
        section="Methods",
        index=0,
        core_text="Source-grounded synthesis input.",
        left_context="",
        right_context="",
        chunk_id="chunk-1",
        document_id="main",
        document_role="main",
        page_ids=(),
        asset_ids=(),
    )

    result = load_existing_result(
        chunk=chunk,
        output_path=path,
    )

    assert result is not None
    assert result.paper_id == "sers-context-regression"
    assert result.chunk_id == "chunk-1"
    assert result.edges[0].relation == "USES_PRECURSOR"
