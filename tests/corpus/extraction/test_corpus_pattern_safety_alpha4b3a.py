from __future__ import annotations

from pathlib import Path

import networkx as nx

from pipeline_core.corpus.corpus_graph import (
    ProjectionBundle,
    build_corpus_graph,
    is_confirmed_corpus_pattern,
)
from domains.sers.profile import SERS_AU_AG_PROFILE


def _bundle(paper_id: str) -> ProjectionBundle:
    graph = nx.MultiDiGraph()

    graph.add_node("metal", type="Metal", label="Ag")
    graph.add_node("nano", type="Nanostructure", label="nanostar")
    graph.add_node("reaction", type="Reaction", label="HER")

    graph.add_node(
        "confirmed",
        type="BridgeConcept",
        label="confirmed pattern",
        retention_lane="accepted_pattern",
        policy_lane="accepted_pattern",
        evidence_status="supported",
        graph_layer="bridge",
        requires_verification=False,
        pattern_subject="surface-enhanced Raman scattering enhancement",
        pattern_relation="VARIES_WITH",
        pattern_object="nanogap size",
    )
    graph.add_node(
        "candidate",
        type="BridgeConcept",
        label="candidate pattern",
        retention_lane="accepted_pattern",
        policy_lane="semantic_candidate",
        evidence_status="semantic_candidate",
        graph_layer="bridge_candidate",
        requires_verification=True,
        pattern_subject="local field",
        pattern_relation="VARIES_WITH",
        pattern_object="shell thickness",
    )

    return ProjectionBundle(
        paper_id=paper_id,
        mode="exploratory",
        root=Path("."),
        graph_path=Path(f"{paper_id}.graphml"),
        node_text_path=Path(f"{paper_id}.nodes.jsonl"),
        edge_evidence_path=Path(f"{paper_id}.edges.jsonl"),
        summary_path=Path(f"{paper_id}.summary.json"),
        graph=graph,
        node_rows=[],
        evidence_rows=[],
        summary={"domain_profile_id": "sers_au_ag"},
        sha256={},
    )


def test_alpha4b3a_confirmed_pattern_predicate_is_fail_closed() -> None:
    assert is_confirmed_corpus_pattern(
        {
            "type": "BridgeConcept",
            "retention_lane": "accepted_pattern",
            "policy_lane": "accepted_pattern",
            "evidence_status": "supported",
            "graph_layer": "bridge",
            "requires_verification": False,
        }
    )
    assert not is_confirmed_corpus_pattern(
        {
            "type": "BridgeConcept",
            "retention_lane": "accepted_pattern",
            "policy_lane": "semantic_candidate",
            "evidence_status": "semantic_candidate",
            "graph_layer": "bridge_candidate",
            "requires_verification": True,
        }
    )


def test_alpha4b3a_sers_corpus_uses_only_domain_safe_alignment_semantics() -> None:
    (
        graph,
        _node_rows,
        _evidence_rows,
        registry_rows,
        pattern_rows,
        candidate_rows,
        manifest,
    ) = build_corpus_graph(
        [_bundle("S1"), _bundle("S2")],
        corpus_id="sers-alpha4b3a",
        mode="exploratory",
        domain_profile=SERS_AU_AG_PROFILE,
    )

    assert len(registry_rows) == 1
    assert registry_rows[0]["entity_type"] == "Metal"
    assert not any(row["entity_type"] == "Reaction" for row in registry_rows)

    nano_candidates = [
        row
        for row in candidate_rows
        if row["node_type"] == "Nanostructure"
    ]
    assert len(nano_candidates) == 1
    assert nano_candidates[0]["review_priority"] == "high"
    assert not any(row["node_type"] == "Reaction" for row in candidate_rows)

    assert len(pattern_rows) == 1
    assert pattern_rows[0]["pattern_subject"] == "sers enhancement"
    assert pattern_rows[0]["pattern_relation"] == "varies with"
    assert pattern_rows[0]["pattern_object"] == "nanogap size"

    corpus_patterns = [
        attrs
        for _node_id, attrs in graph.nodes(data=True)
        if attrs.get("type") == "CorpusPattern"
    ]
    assert len(corpus_patterns) == 1

    assert manifest["domain_profile_id"] == "sers_au_ag"
    assert manifest["corpus_semantics_id"] == "sers_au_ag_corpus_v1_alpha4b3a"
    assert manifest["registry_alignment_types"] == ["Metal"]
    assert manifest["destructive_cross_paper_merges"] == 0
    assert manifest["pattern_alignment_mode"] == "confirmed_exact"
    assert manifest["pattern_alignment_enabled"] is True
