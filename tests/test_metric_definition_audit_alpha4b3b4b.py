from __future__ import annotations

import networkx as nx

from dac_her.domains.sers_au_ag_metric_definition import (
    SERS_AU_AG_METRIC_DEFINITION_ADAPTER,
)
from dac_her.metric_definition_context import audit_metric_definition_contexts


def test_metric_definition_audit_reports_status_and_formula_counts():
    graph = nx.MultiDiGraph(domain_profile_id="sers_au_ag")
    graph.add_node("sub", type="PlasmonicSubstrate")
    graph.add_node(
        "m",
        type="Measurement",
        label="SERS enhancement factor",
        metric_id="sers_enhancement_factor",
        subject_id="sub",
        source_expression="Enhancement factor was 4.2e6.",
        description="",
        qualifier="",
        value_numeric="4200000",
        value_text="",
        unit="",
        basis="Intensity normalized by estimated molecule number",
        conditions_json="[]",
    )
    graph.add_node(
        "calc",
        type="Calculation",
        label="SERS enhancement factor calculation",
        method_details=(
            "Calculated from SERS and normal Raman intensities normalized "
            "by estimated molecule numbers."
        ),
        conditions_json="[]",
    )
    graph.add_edge("calc", "m", relation="HAS_MEASUREMENT")
    graph.add_edge("m", "sub", relation="MEASURED_FOR")

    contexts = SERS_AU_AG_METRIC_DEFINITION_ADAPTER.extract_contexts(graph, "P1")
    audit = audit_metric_definition_contexts(
        contexts=contexts,
        source_graphs={"P1": graph},
        adapter=SERS_AU_AG_METRIC_DEFINITION_ADAPTER,
    )
    assert audit.structural_gate is True
    assert audit.issues == ()
    assert audit.context_count == 1
    assert audit.known_count == 1
    assert audit.partial_count == 0
    assert audit.unknown_count == 0
    assert audit.explicit_formula_count == 1
