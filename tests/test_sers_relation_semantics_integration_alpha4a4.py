from __future__ import annotations

import networkx as nx

from dac_her.domains.graph_registry import get_graph_adapter
from pipeline_core.corpus.graph_semantics import (
    integration_component_diagnostics,
    relation_contract_triage,
    relation_direction_diagnostics,
)
from domains.sers.prompts import (
    SERS_MICRO_REEXTRACT_SYSTEM_PROMPT,
    SERS_PATCH_SYSTEM_PROMPT,
    SERS_PROMPT_VERSION,
    SERS_SYSTEM_PROMPT,
)


def _adapter():
    return get_graph_adapter("sers_au_ag")


def _issues(graph: nx.MultiDiGraph):
    return _adapter().diagnose_relation_contracts(graph)


def test_prompt_version_alpha4a4():
    assert SERS_PROMPT_VERSION == "sers-au-ag-extraction-v1-alpha4a4"


def test_synthesis_method_can_use_optical_condition():
    graph = nx.MultiDiGraph()
    graph.add_node("method", type="SynthesisMethod")
    graph.add_node("light", type="OpticalCondition")
    graph.add_edge(
        "method",
        "light",
        key="e1",
        relation="USES_OPTICAL_CONDITION",
    )
    assert _issues(graph) == []


def test_calculation_can_use_reporter():
    graph = nx.MultiDiGraph()
    graph.add_node("calc", type="Calculation")
    graph.add_node("reporter", type="RamanReporter")
    graph.add_edge(
        "calc",
        "reporter",
        key="e1",
        relation="USES_REPORTER",
    )
    assert _issues(graph) == []


def test_calculation_still_cannot_use_material_relation():
    graph = nx.MultiDiGraph()
    graph.add_node("calc", type="Calculation")
    graph.add_node("water", type="Material")
    graph.add_edge(
        "calc",
        "water",
        key="e1",
        relation="USES_MATERIAL",
    )
    issues = _issues(graph)
    assert len(issues) == 1
    assert issues[0].code == "relation_source_type_mismatch"


def test_reversed_tested_in_is_high_confidence_direction_issue():
    graph = nx.MultiDiGraph()
    graph.add_node("exp", type="Experiment", label="SERS measurement")
    graph.add_node(
        "substrate",
        type="PlasmonicSubstrate",
        label="Au-Ag substrate",
    )
    graph.add_edge(
        "exp",
        "substrate",
        key="e1",
        relation="TESTED_IN",
        evidence_text="The substrate was tested by SERS.",
    )

    triage = relation_contract_triage(graph, graph_adapter=_adapter())
    assert len(triage) == 1
    row = triage[0]
    assert row["category"] == "likely_reversed_relation"
    assert row["confidence"] == "high"
    assert row["suggested_relation"] == "TESTED_IN"
    assert row["suggested_source_id"] == "substrate"
    assert row["suggested_target_id"] == "exp"

    direction_rows = relation_direction_diagnostics(
        graph,
        graph_adapter=_adapter(),
    )
    assert len(direction_rows) == 1


def test_reporter_tested_in_routes_to_uses_reporter_review():
    graph = nx.MultiDiGraph()
    graph.add_node("reporter", type="RamanReporter", label="4-MPy")
    graph.add_node("exp", type="Experiment", label="Raman experiment")
    graph.add_edge(
        "reporter",
        "exp",
        key="e1",
        relation="TESTED_IN",
    )

    row = relation_contract_triage(
        graph,
        graph_adapter=_adapter(),
    )[0]
    assert row["category"] == "wrong_relation_for_role"
    assert row["suggested_relation"] == "USES_REPORTER"
    assert row["suggested_source_id"] == "exp"
    assert row["suggested_target_id"] == "reporter"


def test_calculation_uses_material_is_relation_gap_not_auto_widened():
    graph = nx.MultiDiGraph()
    graph.add_node("calc", type="Calculation", label="Mie calculation")
    graph.add_node("water", type="Material", label="Water")
    graph.add_edge(
        "calc",
        "water",
        key="e1",
        relation="USES_MATERIAL",
    )

    row = relation_contract_triage(
        graph,
        graph_adapter=_adapter(),
    )[0]
    assert row["category"] == "ontology_relation_gap"
    assert row["auto_apply"] is False


def test_biological_material_as_analyte_is_typing_gap():
    graph = nx.MultiDiGraph()
    graph.add_node("exp", type="Experiment", label="cell SERS map")
    graph.add_node("cells", type="Material", label="U87MG cells")
    graph.add_edge(
        "exp",
        "cells",
        key="e1",
        relation="USES_ANALYTE",
    )

    row = relation_contract_triage(
        graph,
        graph_adapter=_adapter(),
    )[0]
    assert row["category"] == "ontology_typing_gap"
    assert row["auto_apply"] is False


def test_motif_component_attachment_requires_owner_review():
    graph = nx.MultiDiGraph()
    graph.add_node("motif", type="StructuralMotif", label="core-shell")
    graph.add_node("au", type="Metal", label="Au")
    graph.add_edge(
        "motif",
        "au",
        key="e1",
        relation="HAS_COMPONENT",
    )

    row = relation_contract_triage(
        graph,
        graph_adapter=_adapter(),
    )[0]
    assert row["category"] == "owner_attachment_required"


def test_integration_diagnostics_generate_review_only_bridge_candidates():
    graph = nx.MultiDiGraph()
    graph.add_node("paper", type="Paper", label="paper")
    graph.add_node(
        "main",
        type="PlasmonicSubstrate",
        label="main substrate",
    )
    graph.add_edge("paper", "main", key="e0", relation="STUDIES")

    graph.add_node(
        "orphan",
        type="Nanostructure",
        label="Au@Ag dimer",
    )
    graph.add_node(
        "calc",
        type="Calculation",
        label="DDA calculation",
    )
    graph.add_edge(
        "orphan",
        "calc",
        key="e1",
        relation="SIMULATED_BY",
    )

    before_nodes = graph.number_of_nodes()
    before_edges = graph.number_of_edges()

    components, candidates = integration_component_diagnostics(
        graph,
        domain_profile_id="sers_au_ag",
    )

    assert len(components) == 1
    assert components[0]["severity"] == "review"
    assert components[0]["contains_primary_subject"] is True
    assert candidates
    assert candidates[0]["suggested_relation"] == "STUDIES"
    assert candidates[0]["auto_apply"] is False

    assert graph.number_of_nodes() == before_nodes
    assert graph.number_of_edges() == before_edges


def test_prompts_encode_alpha4a4_direction_policy():
    main = " ".join(SERS_SYSTEM_PROMPT.split())
    patch = " ".join(SERS_PATCH_SYSTEM_PROMPT.split())
    micro = " ".join(SERS_MICRO_REEXTRACT_SYSTEM_PROMPT.split())

    assert "RELATION DIRECTION AND SCOPE" in main
    assert "subject --TESTED_IN--> Experiment" in main
    assert "subject --SIMULATED_BY--> Calculation" in main
    assert "SynthesisMethod --USES_OPTICAL_CONDITION--> OpticalCondition" in main
    assert "Calculation --USES_REPORTER--> RamanReporter" in main
    assert "RELATION-DIRECTION REPAIR" in patch
    assert "RELATION DIRECTION" in micro
