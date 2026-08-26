from __future__ import annotations

import networkx as nx

from domains.sers.context_compiler import (
    SERSContextCompiler,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisEvidenceStatement,
)


def _statement(
    bridge_id: str,
) -> HypothesisEvidenceStatement:

    return HypothesisEvidenceStatement(
        statement_id="stmt:test",
        text="Grounded bridge-supported statement.",
        epistemic_role="reported",
        claim_kind="association",
        paper_ids=["TEST"],
        scientific_support_node_ids=[
            bridge_id
        ],
        scientific_support_edge_ids=[],
        support_path_ids=[],
        alignment_path_ids=[],
        requires_verification=False,
        eligible_as_premise=True,
        eligible_as_gap=False,
        premise_restrictions=[],
    )


def test_grounded_bridge_follows_direct_expresses_pattern_owner() -> None:
    graph = nx.MultiDiGraph()

    graph.add_node(
        "bridge:test",
        type="BridgeConcept",
        label=(
            "Nanostructure architecture "
            "modulates SERS hotspot properties"
        ),
    )

    graph.add_node(
        "nano:test",
        type="Nanostructure",
        label="Silver microplates",
    )

    graph.add_node(
        "metal:ag",
        type="Metal",
        label="Silver",
    )

    graph.add_node(
        "morphology:platelet",
        type="Morphology",
        label="Platelet morphology",
    )

    graph.add_edge(
        "nano:test",
        "bridge:test",
        relation="EXPRESSES_PATTERN",
    )

    graph.add_edge(
        "nano:test",
        "metal:ag",
        relation="HAS_COMPONENT",
    )

    graph.add_edge(
        "nano:test",
        "morphology:platelet",
        relation="HAS_MORPHOLOGY",
    )

    compiler = SERSContextCompiler(
        graph=graph,
        domain_profile_id="sers-au-ag-v1",
    )

    signature = (
        compiler.compile_grounded_statement(
            _statement(
                "bridge:test"
            )
        )
    )

    values = {
        fact.value
        for fact in signature.facts
        if fact.value is not None
    }

    assert "Silver" in values
    assert "Platelet morphology" in values

    assert any(
        provenance.kind
        == "grounded_structural_edge"
        and "nano:test"
        in provenance.node_ids
        for fact in signature.facts
        for provenance in fact.provenance
    )


def test_grounded_bridge_follows_direct_bridge_pattern_owner_only() -> None:
    graph = nx.MultiDiGraph()

    graph.add_node(
        "bridge:test",
        type="BridgeConcept",
        label=(
            "SERS intensity correlates "
            "with analyte concentration"
        ),
    )

    graph.add_node(
        "substrate:test",
        type="PlasmonicSubstrate",
        label="AuNP filter substrate",
    )

    graph.add_node(
        "metal:au",
        type="Metal",
        label="Gold",
    )

    graph.add_node(
        "pattern:corpus",
        type="CorpusPattern",
        label=(
            "sers intensity correlates "
            "with analyte concentration"
        ),
    )

    graph.add_node(
        "support:unrelated",
        type="Support",
        label="Unrelated corpus-pattern support",
    )

    graph.add_edge(
        "substrate:test",
        "bridge:test",
        relation="GROUNDS_BRIDGE_PATTERN",
    )

    graph.add_edge(
        "substrate:test",
        "metal:au",
        relation="HAS_COMPONENT",
    )

    # Corpus-level pattern linkage must not become experimental context.
    graph.add_edge(
        "pattern:corpus",
        "bridge:test",
        relation="HAS_PATTERN_MENTION",
    )

    graph.add_edge(
        "pattern:corpus",
        "support:unrelated",
        relation="HAS_SUPPORT",
    )

    compiler = SERSContextCompiler(
        graph=graph,
        domain_profile_id="sers-au-ag-v1",
    )

    signature = (
        compiler.compile_grounded_statement(
            _statement(
                "bridge:test"
            )
        )
    )

    values = {
        fact.value
        for fact in signature.facts
        if fact.value is not None
    }

    assert "AuNP filter substrate" in values
    assert "Gold" in values

    assert (
        "Unrelated corpus-pattern support"
        not in values
    )

    assert any(
        provenance.kind
        == "grounded_bridge_owner"
        and "substrate:test"
        in provenance.node_ids
        for fact in signature.facts
        for provenance in fact.provenance
    )
