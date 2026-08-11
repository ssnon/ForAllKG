from __future__ import annotations

import networkx as nx

from dac_her.domains.graph_registry import get_graph_adapter
from dac_her.graph_semantics import (
    apply_graph_domain_canonicalization,
    duplicate_label_groups,
)


def test_sers_relation_contract_flags_endpoint_mismatches():
    graph = nx.MultiDiGraph()
    graph.add_node("substrate", type="PlasmonicSubstrate", label="Au@Ag")
    graph.add_node("synthesis_exp", type="Experiment", label="synthesis")
    graph.add_node("calc_dda", type="Experiment", label="DDA calculation")
    graph.add_node("method", type="SynthesisMethod", label="method")
    graph.add_node("material", type="Material", label="Au")

    graph.add_edge("substrate", "synthesis_exp", key="e1", relation="PREPARED_BY")
    graph.add_edge("substrate", "calc_dda", key="e2", relation="SIMULATED_BY")
    graph.add_edge("method", "material", key="e3", relation="HAS_COMPONENT")

    issues = get_graph_adapter("sers_au_ag").diagnose_relation_contracts(graph)
    codes = [issue.code for issue in issues]
    assert codes.count("relation_target_type_mismatch") == 2
    assert "relation_source_type_mismatch" in codes
    assert {issue.relation for issue in issues} == {
        "PREPARED_BY",
        "SIMULATED_BY",
        "HAS_COMPONENT",
    }


def test_sers_relation_contract_accepts_canonical_endpoints():
    graph = nx.MultiDiGraph()
    graph.add_node("substrate", type="PlasmonicSubstrate", label="Au@Ag")
    graph.add_node("method", type="SynthesisMethod", label="growth")
    graph.add_node("exp", type="Experiment", label="SERS")
    graph.add_node("calc", type="Calculation", label="DDA")
    graph.add_node("analyte", type="Analyte", label="ATP")
    graph.add_node("reporter", type="RamanReporter", label="4-FBT")
    graph.add_node("cond", type="OpticalCondition", label="532 nm")

    graph.add_edge("substrate", "method", key="e1", relation="PREPARED_BY")
    graph.add_edge("substrate", "exp", key="e2", relation="TESTED_IN")
    graph.add_edge("substrate", "calc", key="e3", relation="SIMULATED_BY")
    graph.add_edge("exp", "analyte", key="e4", relation="USES_ANALYTE")
    graph.add_edge("exp", "reporter", key="e5", relation="USES_REPORTER")
    graph.add_edge("exp", "cond", key="e6", relation="USES_OPTICAL_CONDITION")

    assert get_graph_adapter("sers_au_ag").diagnose_relation_contracts(graph) == []


def test_same_paper_nodes_are_deterministically_merged():
    graph = nx.MultiDiGraph(paper_id="Kiwook_SERS_1")
    graph.add_node(
        "paper_title",
        type="Paper",
        label="Highly sensitive and reliable SERS probes",
        description="long description",
    )
    graph.add_node("paper_short", type="Paper", label="Kiwook_SERS_1")
    graph.add_node("substrate", type="PlasmonicSubstrate", label="Au@Ag")
    graph.add_edge("paper_short", "substrate", key="e1", relation="STUDIES")
    graph.add_edge("paper_title", "substrate", key="e2", relation="STUDIES")

    canonical, summary = apply_graph_domain_canonicalization(
        graph,
        graph_adapter=get_graph_adapter("sers_au_ag"),
        paper_id="Kiwook_SERS_1",
    )
    paper_nodes = [
        node_id for node_id, attrs in canonical.nodes(data=True)
        if attrs.get("type") == "Paper"
    ]
    assert paper_nodes == ["paper_title"]
    assert summary["paper_identity_merges"] == 1
    assert canonical.has_edge("paper_title", "substrate")


def test_duplicate_label_groups_are_review_not_auto_merge_for_substrates():
    graph = nx.MultiDiGraph()
    for node_id in ("a", "b"):
        graph.add_node(
            node_id,
            type="PlasmonicSubstrate",
            label="SiO2@Au@Ag nanoparticles",
        )
    groups = duplicate_label_groups(graph)
    assert groups[0]["node_type"] == "PlasmonicSubstrate"
    assert groups[0]["severity"] == "review"
