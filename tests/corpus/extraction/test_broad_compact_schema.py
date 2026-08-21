from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from domains.catalysis_mechanism.compact_schema import BroadMechanismGraphDraft
from pipeline_core.corpus.extraction.draft_schema import KnowledgeGraphDraft
from pipeline_core.llm.llm_telemetry import estimate_tokens, normalize_stage_name


def _empty_payload() -> dict:
    return {
        "paper_id": "broad_test",
        "chunk_id": "broad_test:abstract:chunk",
        "section": "Abstract",
        "document_id": "abstract",
        "document_role": "main",
        "page_ids": [],
        "asset_ids": [],
        "entities": [],
        "experiments": [],
        "calculations": [],
        "measurements": [],
        "measurement_groups": [],
        "observation_claims": [],
        "mechanism_claims": [],
        "edges": [],
    }


def test_compact_schema_expands_to_canonical_draft():
    compact = BroadMechanismGraphDraft.model_validate(_empty_payload())
    expanded = compact.to_knowledge_graph_draft()

    assert isinstance(expanded, KnowledgeGraphDraft)
    assert expanded.measurements == []
    assert expanded.measurement_groups == []
    assert expanded.model_dump(mode="json") == _empty_payload()


def test_compact_schema_rejects_disabled_measurement_payloads():
    payload = _empty_payload()
    payload["measurements"] = ["forbidden"]

    with pytest.raises(ValueError, match="measurements"):
        BroadMechanismGraphDraft.model_validate(payload)


def test_compact_schema_removes_heavy_measurement_definitions():
    full_schema = KnowledgeGraphDraft.model_json_schema()
    compact_schema = BroadMechanismGraphDraft.model_json_schema()
    full_defs = full_schema.get("$defs", {})
    compact_defs = compact_schema.get("$defs", {})

    full_json = json.dumps(
        full_schema,
        sort_keys=True,
        separators=(",", ":"),
    )
    compact_json = json.dumps(
        compact_schema,
        sort_keys=True,
        separators=(",", ":"),
    )

    full_tokens, _ = estimate_tokens(full_json)
    compact_tokens, _ = estimate_tokens(compact_json)

    # Inspect actual JSON Schema definitions rather than searching the whole
    # serialized document. Other retained model descriptions legitimately
    # mention the text "MeasurementNode" even when its heavy $defs entry has
    # been pruned from the compact response schema.
    assert "MeasurementNode" in full_defs
    assert "MeasurementNode" not in compact_defs
    assert "MeasurementGroupDraft" in full_defs
    assert "MeasurementGroupDraft" not in compact_defs
    assert compact_tokens < full_tokens


def test_compact_response_model_keeps_graph_generation_stage_name():
    assert normalize_stage_name(
        "BroadMechanismGraphDraft",
        response_model="BroadMechanismGraphDraft",
    ) == "graph_generation"


