import json

import networkx as nx

from domains.sers.reproducibility import (
    SERS_AU_AG_REPRODUCIBILITY_ADAPTER,
)


def _g():
    return nx.MultiDiGraph()


def test_rsd_uses_connected_batch_context_without_invention():
    graph = _g()
    graph.add_node(
        "exp",
        type="Experiment",
        label="Batch-to-batch SERS reproducibility",
        description="Signals from three different nanoparticle batches were compared.",
        conditions_json=json.dumps([
            {"name": "number of batches", "value_numeric": 3, "unit": "batches"}
        ]),
    )
    graph.add_node(
        "m",
        type="Measurement",
        metric_id="relative_standard_deviation",
        label="Relative standard deviation",
        value_numeric="14.2",
        value_text="",
        unit="%",
        source_expression="14.2% deviation in batch-to-batch experiments.",
    )
    graph.add_node(
        "group",
        type="MeasurementGroup",
        label="Three-batch reproducibility",
        description="Batch-to-batch signal reproducibility.",
    )
    graph.add_edge("exp", "m", relation="HAS_MEASUREMENT")
    graph.add_edge("m", "group", relation="IN_MEASUREMENT_GROUP")

    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(graph, "P1")
    rsd = [row for row in rows if row.evidence_kind == "relative_standard_deviation"]
    assert len(rsd) == 1
    row = rsd[0]
    assert row.reproducibility_scope == "batch_to_batch"
    assert row.value_numeric == 14.2
    assert row.unit == "%"
    assert row.n_batches == 3
    assert row.n_spots is None
    assert set(row.source_node_ids) == {"exp", "m", "group"}


def test_spatial_average_is_grounded_as_three_positions():
    graph = _g()
    graph.add_node(
        "exp",
        type="Experiment",
        label="SERS measurement",
        description=(
            "An average of three spectra from different substrate positions "
            "was used for reproducibility."
        ),
        conditions_json="[]",
    )
    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(graph, "P1")
    assert len(rows) == 1
    row = rows[0]
    assert row.evidence_kind == "spatial_sampling"
    assert row.reproducibility_scope == "spot_to_spot"
    assert row.n_replicates == 3
    assert row.n_spots == 3


def test_population_distribution_uses_explicit_population_count():
    graph = _g()
    graph.add_node(
        "exp",
        type="Experiment",
        label="Single-particle SERS EF measurements",
        description=(
            "Single-particle SERS measurements were used to determine "
            "enhancement-factor distributions for 110 individual DIPs."
        ),
        conditions_json=json.dumps([
            {"name": "measurement population", "value_numeric": 110, "unit": "particles"}
        ]),
    )
    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(graph, "P1")
    assert len(rows) == 1
    row = rows[0]
    assert row.evidence_kind == "population_sampling"
    assert row.reproducibility_scope == "particle_to_particle"
    assert row.n_particles == 110


def test_qualitative_reproducibility_does_not_invent_counts_or_value():
    graph = _g()
    graph.add_node(
        "m",
        type="Measurement",
        metric_id="signal_retention",
        label="Signal retention",
        value_numeric="",
        value_text="Reproducible Raman signals were obtained.",
        unit="",
        description="Qualitative reproducibility observation.",
    )
    rows = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(graph, "P1")
    assert len(rows) == 1
    row = rows[0]
    assert row.evidence_kind == "repeatability_statement"
    assert row.value_numeric is None
    assert row.value_text == ""
    assert row.reproducibility_scope == "unknown"
    assert row.n_batches is None
    assert row.n_replicates is None


def test_uniform_alone_is_not_reproducibility_evidence():
    graph = _g()
    graph.add_node(
        "exp",
        type="Experiment",
        label="Raman mapping",
        description="A uniform SERS map was observed over the selected region.",
        conditions_json="[]",
    )
    assert SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(graph, "P1") == []
