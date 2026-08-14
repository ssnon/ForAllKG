from __future__ import annotations

import networkx as nx

from dac_her.measurement_result_identity import (
    IDENTITY_AWARE_DOMAIN_RECONSTRUCTION_ID,
    MeasurementResultIdentity,
    build_identity_interpretation_graph,
)


def _identity():
    return MeasurementResultIdentity(
        identity_id="measurement-result:test",
        semantics_id="measurement_result_identity_v1_alpha4b4a1",
        paper_id="P1",
        representative_measurement_id="m",
        source_mention_ids=("m", "m__mention_measurement_x"),
        origin_local_id="m",
        status="consolidated_exact",
        consolidation_reasons=("test",),
    )


def _graph():
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node(
        "m",
        type="Measurement",
        metric_id="detection_limit",
        value_numeric=2.4,
        value_text="",
        unit="nM",
        subject_id="s1",
        source_expression="theoretical LOD 2.4 nM",
    )
    graph.add_node(
        "m__mention_measurement_x",
        type="Measurement",
        metric_id="detection_limit",
        value_numeric=2.4,
        value_text="",
        unit="nM",
        subject_id="s2",
        source_local_id="m",
        source_expression="LOD 2.4 nM",
        id_collision_reason="measurement_payload_conflict",
    )
    graph.add_node(
        "exp_mix",
        type="Experiment",
        method_label="nanoparticles were mixed with ATP",
    )
    graph.add_node(
        "exp_drop",
        type="Experiment",
        method_label="drop-cast ATP-modified nanoparticles",
    )
    graph.add_node("s1", type="PlasmonicSubstrate", label="SiO2@Au@Ag")
    graph.add_node("s2", type="PlasmonicSubstrate", label="SiO2@Au@Ag for ATP")
    graph.add_edge("exp_mix", "m", key="a", relation="HAS_MEASUREMENT")
    graph.add_edge(
        "exp_drop",
        "m__mention_measurement_x",
        key="b",
        relation="HAS_MEASUREMENT",
    )
    graph.add_edge("m", "s1", key="c", relation="MEASURED_FOR")
    graph.add_edge(
        "m__mention_measurement_x",
        "s2",
        key="d",
        relation="MEASURED_FOR",
    )
    return graph


def test_transient_graph_rehomes_all_provenance_without_mutating_canonical():
    canonical = _graph()
    overlay = build_identity_interpretation_graph(canonical, [_identity()])

    assert "m__mention_measurement_x" in canonical
    assert "m__mention_measurement_x" not in overlay
    assert "m" in overlay

    producers = {
        str(source)
        for source, _, data in overlay.in_edges("m", data=True)
        if data.get("relation") == "HAS_MEASUREMENT"
    }
    assert producers == {"exp_mix", "exp_drop"}

    subjects = {
        str(target)
        for _, target, data in overlay.out_edges("m", data=True)
        if data.get("relation") == "MEASURED_FOR"
    }
    assert subjects == {"s1", "s2"}

    # Representative payload remains strict; no field-wise Measurement merge.
    assert overlay.nodes["m"]["value_numeric"] == 2.4
    assert overlay.nodes["m"]["value_text"] == ""
    assert overlay.nodes["m"]["measurement_result_source_mention_count"] == 2
    assert overlay.nodes["m"]["identity_aware_domain_reconstruction_id"] == (
        IDENTITY_AWARE_DOMAIN_RECONSTRUCTION_ID
    )


def test_domain_reextraction_can_recover_composite_preparation_from_union():
    overlay = build_identity_interpretation_graph(_graph(), [_identity()])
    producer_text = " ".join(
        str(overlay.nodes[source].get("method_label", "")).lower()
        for source, _, data in overlay.in_edges("m", data=True)
        if data.get("relation") == "HAS_MEASUREMENT"
    )
    steps = []
    if "drop-cast" in producer_text:
        steps.append("drop_cast")
    if "mixed" in producer_text:
        steps.append("mixing")
    assert "+".join(steps) == "drop_cast+mixing"
