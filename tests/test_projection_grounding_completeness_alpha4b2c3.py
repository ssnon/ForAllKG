import networkx as nx

from domains.sers.profile import SERS_AU_AG_PROFILE
from dac_her.graphagents_adapter import build_graphagents_projection


def _bridge(anchor: str) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(
        bridge_extraction_id="e",
        bridge_policy_run_id="p",
        bridge_policy_version="v",
    )
    graph.add_node(
        "bridge::b",
        type="BridgeConcept",
        retention_lane="accepted_pattern",
        concept_type="RelationPattern",
        label="calibration bridge",
        pattern_subject="property",
        pattern_relation="VARIES_WITH",
        pattern_object="condition",
        evidence_scope="paper_result",
    )
    graph.add_edge(
        anchor,
        "bridge::b",
        key="ground",
        relation="EXPRESSES_PATTERN",
        edge_id="edge:ground",
        evidence_pointers_json="[]",
        paper_id="p",
    )
    return graph


def test_alpha4b2c3_sers_profile_has_grounding_completion_rules():
    semantics = SERS_AU_AG_PROFILE.projection
    assert semantics is not None
    assert semantics.semantics_id == "sers_au_ag_projection_v2_alpha4b2c3"

    rules = {
        (rule.relation, rule.direction)
        for rule in semantics.backtrace_rules
    }
    assert ("IN_MEASUREMENT_GROUP", "incoming") in rules
    assert ("USES_PRECURSOR", "incoming") in rules


def test_alpha4b2c3_measurement_group_anchor_lifts_through_measurement():
    graph = nx.MultiDiGraph()
    graph.add_node("sub", type="PlasmonicSubstrate", label="Au-Ag substrate")
    graph.add_node("exp", type="Experiment", label="SERS characterization")
    graph.add_node(
        "m1",
        type="Measurement",
        label="particle diameter at low precursor",
    )
    graph.add_node(
        "group",
        type="MeasurementGroup",
        label="particle diameter versus precursor concentration",
    )

    graph.add_edge(
        "sub", "exp", key="tested", relation="TESTED_IN",
        edge_id="edge:tested", paper_id="p",
    )
    graph.add_edge(
        "exp", "m1", key="measurement", relation="HAS_MEASUREMENT",
        edge_id="edge:measurement", paper_id="p",
    )
    graph.add_edge(
        "m1", "group", key="group", relation="IN_MEASUREMENT_GROUP",
        edge_id="edge:group", paper_id="p",
    )

    projection, _, evidence = build_graphagents_projection(
        graph,
        bridge_graph=_bridge("group"),
        mode="mechanism",
        projection_semantics=SERS_AU_AG_PROFILE.projection,
    )

    assert projection.has_edge("sub", "bridge::b")
    rows = [
        row
        for row in evidence
        if row["source"] == "sub"
        and row["target"] == "bridge::b"
        and row["derivation_rule"] == "lift_removed_bridge_anchor"
    ]
    assert rows
    assert "edge:group" in rows[0]["source_edge_ids"]
    assert "edge:measurement" in rows[0]["source_edge_ids"]
    assert "edge:tested" in rows[0]["source_edge_ids"]
    assert "edge:ground" in rows[0]["source_edge_ids"]


def test_alpha4b2c3_precursor_anchor_lifts_to_synthesis_method():
    graph = nx.MultiDiGraph()
    graph.add_node("method", type="SynthesisMethod", label="Ag shell growth")
    graph.add_node("precursor", type="Precursor", label="AgNO3")
    graph.add_edge(
        "method", "precursor", key="precursor", relation="USES_PRECURSOR",
        edge_id="edge:precursor", paper_id="p",
    )

    projection, _, evidence = build_graphagents_projection(
        graph,
        bridge_graph=_bridge("precursor"),
        mode="mechanism",
        projection_semantics=SERS_AU_AG_PROFILE.projection,
    )

    assert projection.has_edge("method", "bridge::b")
    rows = [
        row
        for row in evidence
        if row["source"] == "method"
        and row["target"] == "bridge::b"
        and row["derivation_rule"] == "lift_removed_bridge_anchor"
    ]
    assert rows
    assert set(rows[0]["source_edge_ids"]) == {
        "edge:precursor",
        "edge:ground",
    }
