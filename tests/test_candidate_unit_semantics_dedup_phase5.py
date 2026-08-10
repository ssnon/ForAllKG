from __future__ import annotations

from dataclasses import replace

import networkx as nx

from dac_her.candidate_unit_selection import (
    CandidateUnitSelectionPolicy,
    CandidateUnitSelector,
)
from dac_her.candidate_units import CandidateUnitBuilder, confirmed_navigation_graph
from dac_her.discovery_semantics import (
    is_alignment_node,
    is_generic_entity_node,
    is_mechanism_edge,
    is_mechanism_node,
    is_scaffold_edge,
    normalized_node_type,
)
from dac_her.domains import get_domain_profile


def test_shared_generic_helper_excludes_alignment_and_mechanism_nodes():
    semantics = get_domain_profile("dac_her").discovery

    assert is_generic_entity_node(
        "metal",
        {"type": "Metal"},
        semantics,
    )
    assert not is_generic_entity_node(
        "hub",
        {
            "type": "Metal",
            "corpus_node_kind": "alignment_hub",
        },
        semantics,
    )
    assert not is_generic_entity_node(
        "paper::mech_custom",
        {"type": "Metal"},
        semantics,
    )


def test_shared_semantics_accept_mapping_attrs_used_by_networkx():
    semantics = get_domain_profile("sers_au_ag").discovery
    assert normalized_node_type({"type": "OpticalCondition"}) == "OPTICALCONDITION"
    assert is_alignment_node({"type": "CorpusAlignment"})
    assert is_mechanism_edge({"relation": "FIELD_COUPLING"}, semantics)
    assert is_scaffold_edge({"relation": "TESTED_IN"}, semantics)


def test_legacy_mechanism_prefix_is_profile_owned_not_hardcoded():
    base = get_domain_profile("sers_au_ag")
    custom_semantics = replace(
        base.discovery,
        mechanism_node_markers=(),
        legacy_mechanism_id_prefixes=("cause_",),
    )
    custom_profile = replace(base, discovery=custom_semantics)

    assert is_mechanism_node(
        "paper::cause_17",
        {"type": "Material"},
        custom_semantics,
    )
    assert not is_mechanism_node(
        "paper::mech_17",
        {"type": "Material"},
        custom_semantics,
    )

    g = nx.DiGraph()
    for node, typ in [
        ("SRC", "Material"),
        ("cause_left", "Material"),
        ("A", "Nanostructure"),
        ("B", "Nanostructure"),
        ("cause_right", "Material"),
        ("TGT", "Material"),
        ("C", "BridgeConcept"),
    ]:
        attrs = {"type": typ, "label": node}
        if node == "C":
            attrs.update(policy_lane="semantic_candidate", requires_verification=True)
        g.add_node(node, **attrs)

    def confirmed(u: str, v: str) -> None:
        g.add_edge(
            u,
            v,
            relation="RELATED_TO",
            edge_class="scientific_confirmed",
            exploration_cost=1.0,
            requires_verification=False,
            reverse_navigation=False,
        )

    def candidate_pair(anchor: str) -> None:
        g.add_edge(
            anchor,
            "C",
            relation="GROUNDS_SEMANTIC_CANDIDATE",
            edge_class="semantic_candidate",
            exploration_cost=2.5,
            requires_verification=True,
            reverse_navigation=False,
            edge_id=f"f:{anchor}:C",
            selected_original_edge_id=f"o:{anchor}:C",
        )
        g.add_edge(
            "C",
            anchor,
            relation="GROUNDS_SEMANTIC_CANDIDATE",
            edge_class="semantic_candidate",
            exploration_cost=3.1,
            requires_verification=True,
            reverse_navigation=True,
            edge_id=f"r:C:{anchor}",
            selected_original_edge_id=f"o:{anchor}:C",
        )

    confirmed("SRC", "cause_left")
    confirmed("cause_left", "A")
    candidate_pair("A")
    candidate_pair("B")
    confirmed("B", "cause_right")
    confirmed("cause_right", "TGT")

    units = CandidateUnitBuilder(g).build(bridge_capable_only=True)
    selector = CandidateUnitSelector(
        g,
        confirmed_navigation_graph(g),
        policy=CandidateUnitSelectionPolicy(max_depth=8, top_k=2),
        unit_relevance={"C": 0.8},
        domain_profile=custom_profile,
    )
    routes = selector.enumerate_routes(
        units,
        [{"node_id": "SRC", "semantic_similarity": 0.8}],
        [{"node_id": "TGT", "semantic_similarity": 0.8}],
    )
    assert routes
    assert routes[0].score.mechanistic_continuity == 1.0


def test_candidate_unit_module_no_longer_defines_private_semantic_classifiers():
    import dac_her.candidate_unit_selection as module

    for name in (
        "_normalized_type",
        "_is_alignment_node",
        "_is_mechanism_node",
        "_is_generic_node",
        "_edge_is_mechanistic",
        "_edge_is_scaffold",
    ):
        assert not hasattr(module, name)
