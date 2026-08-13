from __future__ import annotations

import networkx as nx

from dac_her.comparison_context import dimension_from_values
from dac_her.domains.sers_au_ag_comparison import (
    SERS_AU_AG_COMPARISON_ADAPTER,
    _canonical_concentration,
    _canonical_entity,
    _canonical_power,
    _canonical_raman_peak,
    _canonical_time,
    _canonical_wavelength,
)


def test_alpha4b3b1_physical_scalar_normalization_is_unit_aware():
    assert _canonical_concentration("1e-7 M") == "1e-7 M"
    assert _canonical_concentration("100 nM") == "1e-7 M"
    assert _canonical_concentration("0.1 uM") == "1e-7 M"

    assert _canonical_wavelength("785.0 nm") == "785 nm"
    assert _canonical_power("1000 uW") == "1 mW"
    assert _canonical_time("1 min") == "60 s"
    assert _canonical_raman_peak("1624 cm^-1") == "1624 cm^-1"


def test_alpha4b3b1_entity_aliases_are_narrow_and_conservative():
    assert _canonical_entity("Methylene blue (MB)") == "methylene blue"
    assert _canonical_entity("MB") == "methylene blue"
    assert _canonical_entity("ATP Raman reporter") == "atp"
    assert _canonical_entity("4-aminothiophenol (ATP)") == "4-aminothiophenol"
    # Bare ATP is not expanded to 4-aminothiophenol.
    assert _canonical_entity("ATP") == "atp"


def _contamination_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node("sub", type="PlasmonicSubstrate", label="substrate")
    graph.add_node("exp", type="Experiment", label="SERS experiment")
    graph.add_node("a", type="Analyte", label="Methylene blue")
    graph.add_node(
        "m",
        type="Measurement",
        metric_id="detection_limit",
        metric="Detection limit",
        label="Detection limit",
        value_numeric="1e-9",
        value_text="",
        unit="M",
        source_expression="LOD 10^-9 M",
    )
    graph.add_node(
        "g",
        type="MeasurementGroup",
        label="AgNO3 concentration sweep 50 mM to 300 mM",
    )
    graph.add_edge("sub", "exp", relation="TESTED_IN")
    graph.add_edge("exp", "a", relation="USES_ANALYTE")
    graph.add_edge("exp", "m", relation="HAS_MEASUREMENT")
    graph.add_edge("m", "g", relation="IN_MEASUREMENT_GROUP")
    return graph


def test_alpha4b3b1_synthesis_sweep_is_not_analyte_concentration():
    context = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        _contamination_graph(),
        "P1",
    )[0]
    concentration = context.dimension_map["concentration"]
    assert concentration.status == "unknown"
    assert "300 mm" not in concentration.normalized_value
    assert "50 mm" not in concentration.normalized_value


def test_alpha4b3b1_lod_value_is_not_reused_as_context_concentration():
    context = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        _contamination_graph(),
        "P1",
    )[0]
    assert context.observable_key == "detection_limit"
    assert context.value_numeric == 1e-9
    assert context.dimension_map["concentration"].status == "unknown"


def test_alpha4b3b1_explicit_measurement_concentration_is_canonicalized():
    graph = _contamination_graph()
    graph.nodes["m"]["analyte_concentration"] = "100 nM"
    context = SERS_AU_AG_COMPARISON_ADAPTER.extract_contexts(
        graph,
        "P1",
    )[0]
    concentration = context.dimension_map["concentration"]
    assert concentration.status == "known"
    assert concentration.normalized_value == "1e-7 M"


def test_alpha4b3b1_dimension_normalizer_preserves_raw_provenance():
    dimension = dimension_from_values(
        "concentration",
        [("100 nM", "a"), ("1e-7 M", "b")],
        normalizer=_canonical_concentration,
    )
    assert dimension.status == "known"
    assert dimension.normalized_value == "1e-7 M"
    assert set(dimension.source_values) == {"100 nM", "1e-7 M"}
    assert set(dimension.source_node_ids) == {"a", "b"}
