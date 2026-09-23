from __future__ import annotations


def test_scientific_claim_evidence_graph_core_module_is_available():
    from pipeline_core.discovery.scientific_claim_evidence_graph import (
        build_scientific_claim_evidence_graph,
    )
    assert callable(build_scientific_claim_evidence_graph)
