from __future__ import annotations

from pathlib import Path

import networkx as nx

from dac_her.corpus_graph import ProjectionBundle, build_corpus_graph
from dac_her.domains.catalysis_mechanism import CATALYSIS_MECHANISM_PROFILE
from dac_her.domains.dac_her import DAC_HER_PROFILE


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


def test_alpha4b3a_explicit_her_profile_preserves_legacy_direct_call_outputs() -> None:
    bundles = [_bundle("H1"), _bundle("H2")]

    legacy = build_corpus_graph(
        bundles,
        corpus_id="her-parity",
        mode="exploratory",
    )
    explicit = build_corpus_graph(
        bundles,
        corpus_id="her-parity",
        mode="exploratory",
        domain_profile=DAC_HER_PROFILE,
    )

    legacy_graph = legacy[0]
    explicit_graph = explicit[0]

    assert sorted(legacy_graph.nodes) == sorted(explicit_graph.nodes)
    assert _edge_signature(legacy_graph) == _edge_signature(explicit_graph)
    assert legacy[3] == explicit[3]
    assert legacy[4] == explicit[4]
    assert legacy[5] == explicit[5]
    assert legacy[6] == explicit[6]


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
