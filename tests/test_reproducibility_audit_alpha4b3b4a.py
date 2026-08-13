import networkx as nx

from dac_her.domains.sers_au_ag_reproducibility import (
    SERS_AU_AG_REPRODUCIBILITY_ADAPTER,
)
from dac_her.reproducibility_evidence import audit_reproducibility_evidence


def test_reproducibility_audit_validates_grounded_source_types():
    graph = nx.MultiDiGraph()
    graph.add_node(
        "m",
        type="Measurement",
        metric_id="relative_standard_deviation",
        value_numeric="5",
        value_text="",
        unit="%",
        source_expression="RSD was 5%.",
    )
    evidence = SERS_AU_AG_REPRODUCIBILITY_ADAPTER.extract_evidence(graph, "P1")
    audit = audit_reproducibility_evidence(
        evidence=evidence,
        source_graphs={"P1": graph},
        adapter=SERS_AU_AG_REPRODUCIBILITY_ADAPTER,
    )
    assert audit.structural_gate is True
    assert audit.evidence_count == 1
    assert audit.quantitative_evidence_count == 1
    assert audit.issues == ()
