from __future__ import annotations

import networkx as nx

from domains.dac_her.profile import DAC_HER_PROFILE


def _dac_builder(policy=None):
    return DiscoveryBundleBuilder(
        policy,
        domain_profile=DAC_HER_PROFILE,
    )

from pipeline_core.discovery.discovery_bundle import DiscoveryBundleBuilder, DiscoveryPolicy


def _graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for node_id, paper, node_type in [
        ("paper::A::m1", "A", "Mechanism"),
        ("paper::A::x1", "A", "CatalystModel"),
        ("paper::B::m2", "B", "Mechanism"),
        ("paper::B::o2", "B", "Observation"),
        ("paper::A::metal", "A", "Metal"),
        ("paper::A::her", "A", "Reaction"),
        ("hub", "", "CorpusAlignment"),
    ]:
        g.add_node(
            node_id,
            label=node_id,
            type=node_type,
            source_paper_id=paper,
            corpus_node_kind="alignment_hub" if node_id == "hub" else "",
        )
    g.add_edge("paper::A::m1", "paper::A::x1", relation="MODULATES", edge_class="scientific", exploration_cost=1.0)
    g.add_edge("paper::A::x1", "hub", relation="ALIGNS_TO_REGISTRY_ENTITY", edge_class="registry_alignment", exploration_cost=1.0)
    g.add_edge("hub", "paper::B::m2", relation="HAS_PAPER_MENTION", edge_class="registry_alignment", exploration_cost=1.0)
    g.add_edge("paper::B::m2", "paper::B::o2", relation="INFLUENCES", edge_class="scientific", exploration_cost=1.0)
    g.add_edge("paper::A::metal", "paper::A::her", relation="CATALYZES", edge_class="scientific", exploration_cost=1.0)
    return g


def _step(source: str, target: str, relation: str, edge_id: str, edge_class: str = "scientific") -> dict:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "edge_class": edge_class,
        "navigation_edge_id": edge_id,
        "selected_original_edge_id": edge_id,
        "requires_verification": False,
    }


def test_discovery_bundle_reserves_cross_paper_mechanistic_path() -> None:
    graph = _graph()
    cross = {
        "path_id": "cross",
        "nodes": ["paper::A::m1", "paper::A::x1", "hub", "paper::B::m2", "paper::B::o2"],
        "steps": [
            _step("paper::A::m1", "paper::A::x1", "MODULATES", "e1"),
            _step("paper::A::x1", "hub", "ALIGNS_TO_REGISTRY_ENTITY", "e2", "registry_alignment"),
            _step("hub", "paper::B::m2", "HAS_PAPER_MENTION", "e3", "registry_alignment"),
            _step("paper::B::m2", "paper::B::o2", "INFLUENCES", "e4"),
        ],
        "visited_paper_ids": ["A", "B"],
        "path_quality": {
            "path_type": "CROSS_PAPER_MECHANISTIC",
            "mechanistic_content": "high",
            "mechanistic_edge_density": 0.5,
            "mechanistic_node_density": 0.5,
            "navigation_edge_fraction": 0.5,
            "reverse_fraction": 0.0,
            "candidate_fraction": 0.0,
            "endpoint_pair_score": 0.70,
        },
    }
    generic = {
        "path_id": "generic",
        "nodes": ["paper::A::metal", "paper::A::her"],
        "steps": [_step("paper::A::metal", "paper::A::her", "CATALYZES", "e5")],
        "visited_paper_ids": ["A"],
        "path_quality": {
            "path_type": "SHARED_ENTITY_BRIDGE",
            "mechanistic_content": "low",
            "mechanistic_edge_density": 0.0,
            "mechanistic_node_density": 0.0,
            "navigation_edge_fraction": 0.0,
            "reverse_fraction": 0.0,
            "candidate_fraction": 0.0,
            "endpoint_pair_score": 0.85,
        },
    }
    payload = {
        "corpus_id": "c1",
        "mode": "mechanism",
        "source_query": "coordination",
        "target_query": "HER",
        "semantic_stop_query": None,
        "paths": [generic],
        "candidate_paths": [generic, cross],
    }
    bundle = _dac_builder(
        DiscoveryPolicy(top_k=1, cross_paper_mechanistic_reserve=1)
    ).build([("traversal.json", payload, graph)])
    assert bundle.selected_count == 1
    assert bundle.inspirations[0].source_path_id == "cross"
    assert bundle.inspirations[0].eligible_as_positive_premise is False
    assert "cross_paper_mechanistic" in bundle.inspirations[0].reason_codes


def test_discovery_bundle_warns_when_candidate_pool_missing() -> None:
    graph = _graph()
    payload = {
        "corpus_id": "c1",
        "mode": "mechanism",
        "source_query": "x",
        "target_query": "y",
        "paths": [],
    }
    bundle = _dac_builder().build([("old.json", payload, graph)])
    assert bundle.used_candidate_pool is False
    assert bundle.warnings


class _FakeSemanticIndex:
    def __init__(self, records, embeddings, model_name="fake-model"):
        import numpy as np

        self.records = records
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.manifest = {"model_name": model_name}

    @property
    def model_name(self):
        return str(self.manifest["model_name"])


def test_alpha2_prefers_mechanistic_continuity_over_one_sided_entity_hop() -> None:
    g = nx.DiGraph()
    nodes = [
        ("paper::A::mech_left", "A", "Mechanism"),
        ("paper::A::model", "A", "CatalystModel"),
        ("hub1", "", "CorpusAlignment"),
        ("paper::B::mech_right", "B", "Mechanism"),
        ("paper::B::obs", "B", "Observation"),
        ("paper::A::motif", "A", "CoordinationMotif"),
        ("paper::A::cat", "A", "Catalyst"),
        ("paper::A::metal", "A", "Metal"),
        ("hub2", "", "CorpusAlignment"),
        ("paper::B::metal", "B", "Metal"),
        ("paper::B::cat", "B", "Catalyst"),
        ("paper::B::mech_tail", "B", "Mechanism"),
    ]
    for node_id, paper, node_type in nodes:
        g.add_node(
            node_id,
            label=node_id,
            type=node_type,
            source_paper_id=paper,
            corpus_node_kind="alignment_hub" if node_id.startswith("hub") else "",
        )

    true_path = {
        "path_id": "true",
        "nodes": [
            "paper::A::mech_left",
            "paper::A::model",
            "hub1",
            "paper::B::mech_right",
            "paper::B::obs",
        ],
        "steps": [
            _step("paper::A::mech_left", "paper::A::model", "MODULATES", "t1"),
            _step("paper::A::model", "hub1", "ALIGNS_TO_REGISTRY_ENTITY", "t2", "registry_alignment"),
            _step("hub1", "paper::B::mech_right", "HAS_PAPER_MENTION", "t3", "registry_alignment"),
            _step("paper::B::mech_right", "paper::B::obs", "INFLUENCES", "t4"),
        ],
        "visited_paper_ids": ["A", "B"],
        "path_quality": {
            "path_type": "CROSS_PAPER_MECHANISTIC",
            "mechanistic_content": "high",
            "mechanistic_edge_density": 0.5,
            "mechanistic_node_density": 0.5,
            "navigation_edge_fraction": 0.5,
            "reverse_fraction": 0.0,
            "candidate_fraction": 0.0,
            "endpoint_pair_score": 0.70,
        },
    }
    weak_path = {
        "path_id": "weak",
        "nodes": [
            "paper::A::motif",
            "paper::A::cat",
            "paper::A::metal",
            "hub2",
            "paper::B::metal",
            "paper::B::cat",
            "paper::B::mech_tail",
        ],
        "steps": [
            _step("paper::A::motif", "paper::A::cat", "HAS_MOTIF", "w1"),
            _step("paper::A::cat", "paper::A::metal", "HAS_METAL", "w2"),
            _step("paper::A::metal", "hub2", "ALIGNS_TO_REGISTRY_ENTITY", "w3", "registry_alignment"),
            _step("hub2", "paper::B::metal", "HAS_PAPER_MENTION", "w4", "registry_alignment"),
            _step("paper::B::metal", "paper::B::cat", "HAS_METAL", "w5"),
            _step("paper::B::cat", "paper::B::mech_tail", "SUPPORTED_MECHANISM_INTERPRETATION", "w6"),
        ],
        "visited_paper_ids": ["A", "B"],
        "path_quality": {
            "path_type": "CROSS_PAPER_MECHANISTIC",
            "mechanistic_content": "high",
            "mechanistic_edge_density": 0.2,
            "mechanistic_node_density": 0.2,
            "navigation_edge_fraction": 0.67,
            "reverse_fraction": 0.0,
            "candidate_fraction": 0.0,
            "endpoint_pair_score": 0.82,
        },
    }
    payload = {
        "corpus_id": "c1",
        "mode": "mechanism",
        "source_query": "coordination",
        "target_query": "HER",
        "semantic_stop_query": None,
        "paths": [],
        "candidate_paths": [weak_path, true_path],
    }
    bundle = _dac_builder(
        DiscoveryPolicy(
            top_k=2,
            cross_paper_mechanistic_reserve=1,
            semantic_diversity_enabled=False,
        )
    ).build([("traversal.json", payload, g)])

    assert bundle.inspirations[0].source_path_id == "true"
    assert bundle.inspirations[0].mechanistic_continuity_band == "high"
    weak = next(item for item in bundle.inspirations if item.source_path_id == "weak")
    assert weak.mechanistic_continuity_band == "medium"
    assert weak.generic_entity_fraction >= 0.75
    assert "generic_entity_hopping" in weak.reason_codes


def test_alpha2_semantic_diversity_removes_near_duplicate_routes() -> None:
    g = nx.DiGraph()
    for node_id in ["a1", "a2", "b1", "b2", "c1", "c2"]:
        g.add_node(node_id, label=node_id, type="Mechanism", source_paper_id=node_id[0].upper())

    def path(path_id: str, left: str, right: str, edge_id: str) -> dict:
        return {
            "path_id": path_id,
            "nodes": [left, right],
            "steps": [_step(left, right, "MODULATES", edge_id)],
            "visited_paper_ids": [left[0].upper()],
            "path_quality": {
                "path_type": "DIRECT_MECHANISTIC",
                "mechanistic_content": "high",
                "mechanistic_edge_density": 1.0,
                "mechanistic_node_density": 1.0,
                "navigation_edge_fraction": 0.0,
                "reverse_fraction": 0.0,
                "candidate_fraction": 0.0,
                "endpoint_pair_score": 0.72,
            },
        }

    p1 = path("p1", "a1", "a2", "e1")
    p2 = path("p2", "b1", "b2", "e2")
    p3 = path("p3", "c1", "c2", "e3")
    payload = {
        "corpus_id": "c1",
        "mode": "mechanism",
        "source_query": "x",
        "target_query": "y",
        "semantic_stop_query": None,
        "paths": [],
        "candidate_paths": [p1, p2, p3],
    }
    # p1 and p2 occupy the same semantic direction; p3 is orthogonal.
    semantic_index = _FakeSemanticIndex(
        records=[{"node_id": x} for x in ["a1", "a2", "b1", "b2", "c1", "c2"]],
        embeddings=[
            [1.0, 0.0], [0.98, 0.02],
            [0.99, 0.01], [0.97, 0.03],
            [0.0, 1.0], [0.02, 0.98],
        ],
    )
    bundle = _dac_builder(
        DiscoveryPolicy(
            top_k=2,
            cross_paper_mechanistic_reserve=0,
            semantic_similarity_threshold=0.88,
            semantic_relaxed_threshold=0.94,
        )
    ).build(
        [("traversal.json", payload, g)],
        semantic_indexes={"traversal.json": semantic_index},
    )
    selected = {item.source_path_id for item in bundle.inspirations}
    assert selected in ({"p1", "p3"}, {"p2", "p3"})
    assert bundle.semantic_diversity_mode == "node_embedding"
    assert max(item.max_semantic_similarity_to_selected for item in bundle.inspirations) < 0.88
