from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from dac_her.discovery_bundle import (
    DiscoveryBundleBuilder,
    DiscoveryPolicy,
    load_traversal_with_graph,
)
from dac_her.domains import get_domain_profile


def _sers_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node("a", type="Material", label="Au-Ag structure")
    graph.add_node("m1", type="Nanostructure", label="nanogap architecture")
    graph.add_node(
        "hub",
        type="CorpusAlignment",
        graph_layer="corpus_alignment",
        label="alignment hub",
    )
    graph.add_node("m2", type="Nanostructure", label="plasmonic hotspot")
    graph.add_node("b", type="Analyte", label="SERS response")
    graph.add_edge("a", "m1", relation="COUPLES_WITH", edge_class="scientific")
    graph.add_edge("m1", "hub", relation="ALIGNS_TO_REGISTRY_ENTITY", edge_class="registry_alignment")
    graph.add_edge("hub", "m2", relation="ALIGNS_TO_REGISTRY_ENTITY", edge_class="registry_alignment")
    graph.add_edge("m2", "b", relation="FOCUSES_FIELD_AT", edge_class="scientific")
    return graph


def _path_row():
    return {
        "path_id": "path:sers",
        "nodes": ["a", "m1", "hub", "m2", "b"],
        "steps": [
            {"source": "a", "target": "m1", "relation": "COUPLES_WITH", "edge_class": "scientific"},
            {"source": "m1", "target": "hub", "relation": "ALIGNS_TO_REGISTRY_ENTITY", "edge_class": "registry_alignment"},
            {"source": "hub", "target": "m2", "relation": "ALIGNS_TO_REGISTRY_ENTITY", "edge_class": "registry_alignment"},
            {"source": "m2", "target": "b", "relation": "FOCUSES_FIELD_AT", "edge_class": "scientific"},
        ],
        "visited_paper_ids": ["p1", "p2"],
        "path_quality": {
            "path_type": "CROSS_PAPER_MECHANISTIC",
            "mechanistic_content": "medium",
            "mechanistic_edge_density": 0.5,
            "mechanistic_node_density": 0.0,
            "navigation_edge_fraction": 0.5,
            "reverse_fraction": 0.0,
            "endpoint_pair_score": 0.8,
        },
    }


def test_sers_profile_controls_discovery_mechanistic_continuity():
    profile = get_domain_profile("sers_au_ag")
    payload = {
        "corpus_id": "sers_test",
        "domain_profile_id": "sers_au_ag",
        "mode": "exploratory",
        "source_query": "Au-Ag structure",
        "target_query": "SERS response",
        "semantic_stop_query": None,
        "candidate_paths": [_path_row()],
        "paths": [_path_row()],
    }
    bundle = DiscoveryBundleBuilder(
        DiscoveryPolicy(
            top_k=1,
            semantic_diversity_enabled=False,
            min_exploration_score=0.0,
        ),
        domain_profile=profile,
    ).build([("sers.json", payload, _sers_graph())])

    assert bundle.selected_count == 1
    inspiration = bundle.inspirations[0]
    assert inspiration.mechanistic_continuity_band == "high"
    assert inspiration.mechanism_before_alignment is True
    assert inspiration.mechanism_after_alignment is True
    # Nanostructure is a SERS generic entity type; this checks that DAC-HER's
    # hard-coded generic type set is no longer driving the score.
    assert inspiration.generic_entity_fraction > 0.0


def test_discovery_builder_rejects_explicit_domain_mismatch():
    payload = {
        "corpus_id": "sers_test",
        "domain_profile_id": "sers_au_ag",
        "mode": "exploratory",
        "candidate_paths": [_path_row()],
        "paths": [_path_row()],
    }
    with pytest.raises(ValueError, match="does not match traversal"):
        DiscoveryBundleBuilder(
            DiscoveryPolicy(top_k=1, semantic_diversity_enabled=False),
            domain_profile=get_domain_profile("dac_her"),
        ).build([("sers.json", payload, _sers_graph())])


def test_loader_uses_traversal_data_root_instead_of_hardcoded_data_dac(tmp_path: Path):
    data_root = tmp_path / "data_sers"
    graph_dir = data_root / "corpus" / "sers_test" / "mechanism" / "navigation"
    graph_dir.mkdir(parents=True)
    nx.write_graphml(_sers_graph(), graph_dir / "graph.graphml")

    traversal = tmp_path / "traversal.json"
    traversal.write_text(
        json.dumps(
            {
                "corpus_id": "sers_test",
                "domain_profile_id": "sers_au_ag",
                "data_root": str(data_root),
                "mode": "mechanism",
            }
        ),
        encoding="utf-8",
    )

    _, payload, graph = load_traversal_with_graph(traversal)
    assert payload["domain_profile_id"] == "sers_au_ag"
    assert graph.number_of_nodes() == _sers_graph().number_of_nodes()
