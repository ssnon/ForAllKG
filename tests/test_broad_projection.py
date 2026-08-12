from __future__ import annotations

import json

import networkx as nx

from dac_her.broad_projection import (
    BROAD_EVIDENCE_STATUS,
    BROAD_GRAPH_LAYER,
    build_broad_mechanism_projection,
    summarize_broad_projection,
)


def _canonical_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        paper_id="broad_demo",
        domain_profile_id="catalysis_mechanism",
        extraction_quality_status="complete",
        extraction_complete=True,
        extraction_source_token_coverage=1.0,
        extraction_quarantine_token_fraction=0.0,
        extraction_active_chunk_count=1,
        extraction_quarantined_chunk_count=0,
        extraction_failed_chunk_count=0,
        extraction_coverage_exact=True,
        extraction_absence_claims_allowed=True,
        extraction_coverage_sensitive_queries_allowed=True,
        run_id="run-demo",
        run_fingerprint="fp-demo",
    )
    graph.add_node("catalyst", type="Catalyst", label="FeCo site")
    graph.add_node("env", type="InterfacialEnvironment", label="interfacial water")
    graph.add_node("barrier", type="MechanisticFactor", label="proton-transfer barrier")
    graph.add_node("experiment", type="Experiment", label="generic experiment")
    pointer = [{
        "document_id": "abstract",
        "document_role": "main",
        "page_id": None,
        "asset_ids": [],
    }]
    graph.add_edge(
        "catalyst",
        "env",
        key="e1",
        edge_id="edge_1",
        relation="HAS_ENVIRONMENT",
        paper_id="broad_demo",
        evidence_pointers_json=json.dumps(pointer),
    )
    graph.add_edge(
        "env",
        "barrier",
        key="e2",
        edge_id="edge_2",
        relation="MODULATES",
        paper_id="broad_demo",
        evidence_pointers_json=json.dumps(pointer),
    )
    graph.add_edge(
        "catalyst",
        "experiment",
        key="e3",
        edge_id="edge_3",
        relation="EVALUATED_IN",
        paper_id="broad_demo",
        evidence_pointers_json=json.dumps(pointer),
    )
    return graph


def test_broad_projection_keeps_mechanism_nodes_and_omits_evidence_plumbing():
    projection, node_rows, evidence_rows = build_broad_mechanism_projection(
        _canonical_graph()
    )

    assert set(projection.nodes) == {"catalyst", "env", "barrier"}
    assert projection.number_of_edges() == 2
    assert "experiment" not in projection
    assert len(node_rows) == 3
    assert len(evidence_rows) == 2
    assert projection.graph["source_extraction_run_id"] == "run-demo"
    assert projection.graph["source_extraction_run_fingerprint"] == "fp-demo"

    attrs = projection.edges["env", "barrier", next(iter(projection["env"]["barrier"]))]
    assert attrs["evidence_status"] == BROAD_EVIDENCE_STATUS
    assert attrs["graph_layer"] == BROAD_GRAPH_LAYER
    assert attrs["requires_verification"] is True
    assert json.loads(attrs["source_edge_ids_json"]) == ["edge_2"]


def test_broad_projection_summary_counts_direct_mechanism_edges():
    projection, _, _ = build_broad_mechanism_projection(_canonical_graph())
    summary = summarize_broad_projection(projection)
    assert summary["nodes"] == 3
    assert summary["edges"] == 2
    assert summary["direct_mechanism_edges"] == 1
    assert summary["mechanism_edge_fraction"] == 0.5
