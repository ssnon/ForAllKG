from __future__ import annotations

from pathlib import Path

import networkx as nx

from pipeline_core.corpus.corpus_graph import ProjectionBundle, build_corpus_graph
from domains.catalysis_mechanism.profile import CATALYSIS_MECHANISM_PROFILE


def _bundle(paper_id: str, *, domain: str = "dac_her") -> ProjectionBundle:
    graph = nx.MultiDiGraph()
    graph.add_node("metal", type="Metal", label="Pt")
    graph.add_node("reaction", type="Reaction", label="hydrogen evolution reaction")
    graph.add_node("material", type="Material", label="carbon support")
    graph.add_node(
        "pattern",
        type="BridgeConcept",
        label="bridge pattern",
        retention_lane="accepted_pattern",
        policy_lane="accepted_pattern",
        evidence_status="supported",
        graph_layer="bridge",
        requires_verification=False,
        pattern_subject="hydrogen adsorption",
        pattern_relation="VARIES_WITH",
        pattern_object="coordination",
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
        summary={"domain_profile_id": domain},
        sha256={},
    )


def _edge_signature(graph: nx.MultiDiGraph) -> list[tuple[str, str, str, str]]:
    return sorted(
        (
            str(left),
            str(right),
            str(key),
            str(attrs.get("relation", "")),
        )
        for left, right, key, attrs in graph.edges(keys=True, data=True)
    )




def test_alpha4b3a_broad_profile_disables_pattern_alignment_capability() -> None:
    bundles = [
        _bundle("B1", domain="catalysis_mechanism"),
        _bundle("B2", domain="catalysis_mechanism"),
    ]
    result = build_corpus_graph(
        bundles,
        corpus_id="broad",
        mode="exploratory",
        domain_profile=CATALYSIS_MECHANISM_PROFILE,
    )
    manifest = result[-1]
    assert result[4] == []
    assert manifest["pattern_alignment_mode"] == "disabled"
    assert manifest["pattern_alignment_enabled"] is False
