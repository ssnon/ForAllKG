from __future__ import annotations

import networkx as nx

from domains.sers.context_compiler import (
    SERSContextCompiler,
)
from pipeline_core.discovery.discovery_contracts import (
    DiscoveryInspiration,
)


def _score_breakdown() -> dict:
    return {
        "endpoint_relevance": 0.0,
        "mechanistic_content": 0.0,
        "cross_paper_span": 0.0,
        "community_span": 0.0,
        "relation_rarity": 0.0,
        "exploratory_mode_bonus": 0.0,
        "grounding_redundancy_penalty": 0.0,
        "navigation_burden_penalty": 0.0,
        "reverse_burden_penalty": 0.0,
        "total": 0.0,
    }


def _inspiration() -> DiscoveryInspiration:
    return DiscoveryInspiration.model_validate(
        {
            "inspiration_id":
                "discovery_inspiration:test",

            "source_path_id":
                "path:test",

            "source_corpus_id":
                "corpus:test",

            "source_mode":
                "exploratory",

            "rendered_path":
                "claim entry -> bridge <- claim exit",

            "exploration_score":
                0.5,

            "score_breakdown":
                _score_breakdown(),

            "node_ids": [
                "claim:entry",
                "bridge:test",
                "claim:exit",
            ],

            "candidate_unit_id":
                "candidate_unit:test",

            "candidate_unit_label":
                "Porous hotspot density promotes SERS",

            "candidate_entry_anchor_id":
                "claim:entry",

            "candidate_entry_anchor_label":
                "entry",

            "candidate_exit_anchor_id":
                "claim:exit",

            "candidate_exit_anchor_label":
                "exit",

            "candidate_unit_score":
                0.5,
        }
    )


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    graph.add_node(
        "claim:entry",
        type="ObservationClaim",
        label="SERS performance observation",
    )

    graph.add_node(
        "claim:exit",
        type="MechanismClaim",
        label="Hotspot mechanism claim",
    )

    graph.add_node(
        "bridge:test",
        type="BridgeConcept",
        label="Porous hotspot density promotes SERS",
    )

    graph.add_node(
        "substrate:linked",
        type="PlasmonicSubstrate",
        label="Silver porous microplate film",
    )

    graph.add_node(
        "metal:ag",
        type="Metal",
        label="Silver",
    )

    graph.add_node(
        "morphology:porous",
        type="Morphology",
        label="Porous network",
    )

    # Deliberately same-paper-like but not claim-local.
    graph.add_node(
        "substrate:unrelated",
        type="PlasmonicSubstrate",
        label="Unrelated same-paper substrate",
    )

    graph.add_edge(
        "claim:entry",
        "bridge:test",
        relation="GROUNDS_SEMANTIC_CANDIDATE",
    )

    graph.add_edge(
        "claim:exit",
        "bridge:test",
        relation="GROUNDS_SEMANTIC_CANDIDATE",
    )

    graph.add_edge(
        "claim:entry",
        "substrate:linked",
        relation="APPLIES_TO",
    )

    graph.add_edge(
        "claim:exit",
        "substrate:linked",
        relation="APPLIES_TO",
    )

    graph.add_edge(
        "substrate:linked",
        "metal:ag",
        relation="HAS_COMPONENT",
    )

    graph.add_edge(
        "substrate:linked",
        "morphology:porous",
        relation="HAS_MORPHOLOGY",
    )

    return graph


def test_axis_claim_anchors_follow_direct_applies_to_context() -> None:
    compiler = SERSContextCompiler(
        graph=_graph(),
        domain_profile_id="sers-au-ag-v1",
    )

    signature = (
        compiler.compile_axis_inspiration(
            _inspiration()
        )
    )

    values = {
        fact.value
        for fact in signature.facts
        if fact.value is not None
    }

    assert (
        "Silver porous microplate film"
        in values
    )

    assert "Silver" in values
    assert "Porous network" in values

    assert (
        "Unrelated same-paper substrate"
        not in values
    )


def test_axis_applies_to_target_has_claim_local_provenance() -> None:
    compiler = SERSContextCompiler(
        graph=_graph(),
        domain_profile_id="sers-au-ag-v1",
    )

    signature = (
        compiler.compile_axis_inspiration(
            _inspiration()
        )
    )

    substrate_fact = next(
        fact
        for fact in signature.facts
        if (
            fact.dimension == "substrate"
            and fact.value
            == "Silver porous microplate film"
        )
    )

    assert any(
        provenance.kind
        == "axis_direct_claim"
        for provenance
        in substrate_fact.provenance
    )


def test_axis_structural_context_remains_direct_only() -> None:
    compiler = SERSContextCompiler(
        graph=_graph(),
        domain_profile_id="sers-au-ag-v1",
    )

    signature = (
        compiler.compile_axis_inspiration(
            _inspiration()
        )
    )

    morphology_fact = next(
        fact
        for fact in signature.facts
        if (
            fact.dimension
            == "morphology"
        )
    )

    assert any(
        provenance.kind
        == "axis_structural_edge"
        for provenance
        in morphology_fact.provenance
    )
