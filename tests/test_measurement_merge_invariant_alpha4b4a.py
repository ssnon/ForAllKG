from __future__ import annotations

import networkx as nx

from pipeline_core.corpus.measurement_merge_invariants import (
    MEASUREMENT_MERGE_INVARIANT_ID,
    assert_measurement_value_xor,
    measurement_mentions_conflict,
    measurement_value_payload_issues,
)
from scripts.build_paper_graph import merge_chunk_graph


def _measurement(
    *,
    value_numeric="",
    value_text="",
    unit="nm",
    subject_id="sub",
    metric_id="shell_thickness",
):
    return {
        "type": "Measurement",
        "label": "Shell thickness",
        "metric_id": metric_id,
        "metric": "Shell thickness",
        "subject_id": subject_id,
        "source_expression": "reported shell thickness",
        "group_id": "",
        "value_numeric": value_numeric,
        "value_text": value_text,
        "unit": unit,
        "uncertainty": "",
        "qualifier": "",
        "basis": "",
        "conditions_json": "[]",
        "metric_parameters_json": "{}",
        "description": "",
    }


def _entity():
    return {
        "type": "PlasmonicSubstrate",
        "label": "substrate",
        "description": "",
    }


def test_numeric_and_text_mentions_are_semantically_conflicting():
    assert measurement_mentions_conflict(
        _measurement(value_numeric=3.6, value_text=""),
        _measurement(value_numeric="", value_text="3.6–10.0 nm"),
    )


def test_equal_numeric_mentions_can_merge():
    assert not measurement_mentions_conflict(
        _measurement(value_numeric=3.6, value_text=""),
        _measurement(value_numeric="3.600", value_text="", unit="nm"),
    )


def test_different_numeric_results_do_not_merge():
    assert measurement_mentions_conflict(
        _measurement(value_numeric=3.6, value_text=""),
        _measurement(value_numeric=10.0, value_text=""),
    )


def test_merge_chunk_graph_preserves_conflicting_measurement_as_separate_mention():
    merged = nx.MultiDiGraph()
    merged.add_node("sub", **_entity())
    merged.add_node(
        "m",
        **_measurement(value_numeric=3.6, value_text=""),
    )
    merged.add_edge("m", "sub", key="old", relation="MEASURED_FOR")

    chunk = nx.MultiDiGraph()
    chunk.add_node("sub", **_entity())
    chunk.add_node(
        "m",
        **_measurement(
            value_numeric="",
            value_text="3.6–10.0 nm",
            unit="",
        ),
    )
    chunk.add_node(
        "exp",
        type="Experiment",
        label="experiment",
        experiment_type="sers_spectroscopy",
        conditions_json="[]",
    )
    chunk.add_edge("m", "sub", key="1", relation="MEASURED_FOR")
    chunk.add_edge("exp", "m", key="2", relation="HAS_MEASUREMENT")

    collisions = merge_chunk_graph(
        merged,
        chunk,
        chunk_id="paper:chunk:new",
    )

    measurement_ids = sorted(
        str(node_id)
        for node_id, attrs in merged.nodes(data=True)
        if attrs.get("type") == "Measurement"
    )
    assert len(measurement_ids) == 2
    assert "m" in measurement_ids

    preserved = next(
        node_id
        for node_id in measurement_ids
        if node_id != "m"
    )
    assert preserved.startswith("m__mention_measurement_")
    assert merged.nodes["m"]["value_numeric"] == 3.6
    assert merged.nodes["m"]["value_text"] == ""
    assert merged.nodes[preserved]["value_numeric"] == ""
    assert merged.nodes[preserved]["value_text"] == "3.6–10.0 nm"
    assert merged.nodes[preserved]["id_collision_reason"] == (
        "measurement_payload_conflict"
    )
    assert any(
        source == "exp"
        and target == preserved
        and attrs.get("relation") == "HAS_MEASUREMENT"
        for source, target, attrs in merged.edges(data=True)
    )
    assert collisions[0]["collision_reason"] == (
        "measurement_payload_conflict"
    )
    assert_measurement_value_xor(
        merged,
        stage="unit_test_after_split",
    )


def test_identical_numeric_mentions_keep_one_measurement_node():
    merged = nx.MultiDiGraph()
    merged.add_node("sub", **_entity())
    merged.add_node(
        "m",
        **_measurement(value_numeric=3.6, value_text=""),
    )

    chunk = nx.MultiDiGraph()
    chunk.add_node("sub", **_entity())
    chunk.add_node(
        "m",
        **_measurement(value_numeric=3.6, value_text=""),
    )

    collisions = merge_chunk_graph(
        merged,
        chunk,
        chunk_id="paper:chunk:same",
    )
    measurements = [
        node_id
        for node_id, attrs in merged.nodes(data=True)
        if attrs.get("type") == "Measurement"
    ]
    assert measurements == ["m"]
    assert collisions == []


