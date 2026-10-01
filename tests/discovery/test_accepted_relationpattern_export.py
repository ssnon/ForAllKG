from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
import pytest

from pipeline_core.discovery.accepted_relationpattern_export import (
    export_accepted_relationpatterns_from_canonical_graph,
)


def _valid_attrs():
    phrase = "SERS intensity varies with Ag shell thickness."
    return {
        "type": "BridgeConcept",
        "concept_type": "RelationPattern",
        "label": "SERS intensity varies with shell thickness",
        "source_phrase": phrase,
        "description": "",
        "source_local_id": "local:1",
        "retention_lane": "accepted_pattern",
        "evidence_scope": "paper_result",
        "pattern_subject": "SERS intensity",
        "pattern_relation": "VARIES_WITH",
        "pattern_object": "Ag shell thickness",
        "relation_strength": "correlational",
        "qualifiers_json": "[]",
        "pattern_support_mode": "explicit_single_span",
        "supporting_phrases_json": '["SERS intensity varies with Ag shell thickness."]',
        "subject_evidence_phrase": "SERS intensity",
        "relation_evidence_phrase": "varies with",
        "object_evidence_phrase": "Ag shell thickness",
        "comparison_items_json": "[]",
        "paper_id": "paper:1",
        "chunk_id": "chunk:1",
        "document_id": "document:1",
    }


def test_canonical_export_preserves_validator_approved_relationpattern(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node("rp:1", **_valid_attrs())
    graph.add_node(
        "ordinary:1",
        type="Entity",
        concept_type="Material",
        retention_lane="",
    )
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)

    output = tmp_path / "accepted.csv"
    audit = export_accepted_relationpatterns_from_canonical_graph(
        graph_path=graph_path,
        output_csv=output,
    )

    assert audit.accepted_relationpattern_node_count == 1
    assert audit.validated_component_count == 1
    assert audit.validation_rejection_count == 0
    assert audit.production_selection_authority is False
    assert audit.canonical_graph_mutated is False

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["node_id"] == "rp:1"
    assert rows[0]["pattern_relation"] == "VARIES_WITH"


def test_canonical_export_fails_closed_on_invalid_accepted_pattern(tmp_path):
    graph = nx.MultiDiGraph()
    attrs = _valid_attrs()
    attrs["paper_id"] = ""
    graph.add_node("rp:bad", **attrs)
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)

    with pytest.raises(RuntimeError, match="validation failed closed"):
        export_accepted_relationpatterns_from_canonical_graph(
            graph_path=graph_path,
            output_csv=tmp_path / "accepted.csv",
        )


def test_canonical_export_rejects_graph_without_accepted_patterns(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node("x", type="Entity")
    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)

    with pytest.raises(RuntimeError, match="zero accepted RelationPattern"):
        export_accepted_relationpatterns_from_canonical_graph(
            graph_path=graph_path,
            output_csv=tmp_path / "accepted.csv",
        )
