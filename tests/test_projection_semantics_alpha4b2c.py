import json
import networkx as nx

from domains.dac_her.profile import DAC_HER_PROFILE
from domains.sers.profile import SERS_AU_AG_PROFILE
from pipeline_core.graphagents_adapter import build_graphagents_projection


def _sers_graph():
    g = nx.MultiDiGraph()
    g.add_node("sub", type="PlasmonicSubstrate", label="Au-Ag substrate")
    g.add_node("exp", type="Experiment", label="SERS experiment")
    g.add_node("m", type="Measurement", label="SERS intensity")
    g.add_node("claim", type="ObservationClaim", label="SERS enhancement")
    g.add_node("reporter", type="RamanReporter", label="R6G")
    g.add_edge("sub", "exp", key="tested", relation="TESTED_IN",
               edge_id="edge:tested", paper_id="p")
    g.add_edge("exp", "m", key="measured", relation="HAS_MEASUREMENT",
               edge_id="edge:measured", paper_id="p")
    g.add_edge("m", "claim", key="supports", relation="SUPPORTS_CLAIM",
               edge_id="edge:supports", paper_id="p")
    g.add_edge("exp", "reporter", key="reporter", relation="USES_REPORTER",
               edge_id="edge:reporter", paper_id="p")
    return g


def _bridge(anchor):
    g = nx.MultiDiGraph(
        bridge_extraction_id="e",
        bridge_policy_run_id="p",
        bridge_policy_version="v",
    )
    g.add_node(
        "bridge::b",
        type="BridgeConcept",
        retention_lane="accepted_pattern",
        concept_type="RelationPattern",
        label="SERS enhancement varies with nanogap size",
        pattern_subject="SERS enhancement",
        pattern_relation="VARIES_WITH",
        pattern_object="nanogap size",
        evidence_scope="paper_result",
    )
    g.add_edge(
        anchor, "bridge::b", key="ground", relation="EXPRESSES_PATTERN",
        edge_id="edge:ground", evidence_pointers_json="[]", paper_id="p"
    )
    return g


def test_profiles_have_projection_semantics():
    assert DAC_HER_PROFILE.projection is not None
    assert SERS_AU_AG_PROFILE.projection is not None
    assert DAC_HER_PROFILE.projection.semantics_id == "dac_her_projection_v1_alpha4b2c"
    assert SERS_AU_AG_PROFILE.projection.semantics_id == "sers_au_ag_projection_v2_alpha4b2c3"


def test_sers_measurement_chain_lifts_to_substrate_claim():
    p, _, rows = build_graphagents_projection(
        _sers_graph(),
        mode="mechanism",
        projection_semantics=SERS_AU_AG_PROFILE.projection,
    )
    assert "sub" in p and "claim" in p
    assert "exp" not in p and "m" not in p and "reporter" not in p
    assert p.has_edge("sub", "claim")
    assert any(
        r["source"] == "sub"
        and r["target"] == "claim"
        and r["derivation_rule"] == "evidence_chain_to_claim"
        for r in rows
    )


def test_sers_bridge_measurement_anchor_lifts_to_substrate():
    p, _, rows = build_graphagents_projection(
        _sers_graph(),
        bridge_graph=_bridge("m"),
        mode="mechanism",
        projection_semantics=SERS_AU_AG_PROFILE.projection,
    )
    assert p.has_edge("sub", "bridge::b")
    assert any(
        r["source"] == "sub"
        and r["target"] == "bridge::b"
        and r["derivation_rule"] == "lift_removed_bridge_anchor"
        for r in rows
    )


def test_outgoing_measured_for_is_supported():
    g = nx.MultiDiGraph()
    g.add_node("m", type="Measurement", label="EF")
    g.add_node("sub", type="PlasmonicSubstrate", label="Au-Ag substrate")
    g.add_edge("m", "sub", key="for", relation="MEASURED_FOR",
               edge_id="edge:for", paper_id="p")
    p, _, _ = build_graphagents_projection(
        g,
        bridge_graph=_bridge("m"),
        mode="mechanism",
        projection_semantics=SERS_AU_AG_PROFILE.projection,
    )
    assert p.has_edge("sub", "bridge::b")


def test_default_projection_call_preserves_legacy_her_behavior():
    g = nx.MultiDiGraph()
    g.add_node("cat", type="Catalyst", label="M-N4")
    g.add_node("exp", type="Experiment", label="HER experiment")
    g.add_node("m", type="Measurement", label="overpotential")
    g.add_node("claim", type="ObservationClaim", label="improved HER")
    g.add_edge("cat", "exp", key="eval", relation="EVALUATED_IN",
               edge_id="edge:eval", paper_id="p")
    g.add_edge("exp", "m", key="measurement", relation="HAS_MEASUREMENT",
               edge_id="edge:measurement", paper_id="p")
    g.add_edge("m", "claim", key="support", relation="SUPPORTS_CLAIM",
               edge_id="edge:support", paper_id="p")

    default, _, _ = build_graphagents_projection(g, mode="mechanism")
    explicit, _, _ = build_graphagents_projection(
        g, mode="mechanism",
        projection_semantics=DAC_HER_PROFILE.projection,
    )
    assert set(default.nodes) == set(explicit.nodes)
    assert set(default.edges) == set(explicit.edges)
    assert default.has_edge("cat", "claim")
