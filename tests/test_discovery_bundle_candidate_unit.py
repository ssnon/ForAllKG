from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from dac_her.discovery_bundle import DiscoveryBundleBuilder, DiscoveryPolicy


@dataclass
class FakeIndex:
    records: list[dict]
    embeddings: np.ndarray
    model_name: str = "test-model"


def _step(u: str, v: str, relation: str, *, candidate: bool = False) -> dict:
    return {
        "source": u,
        "target": v,
        "relation": relation,
        "edge_class": "semantic_candidate" if candidate else "scientific_confirmed",
        "navigation_edge_id": f"n:{u}:{v}",
        "selected_original_edge_id": f"o:{u}:{v}",
        "requires_verification": candidate,
        "reverse_navigation": False,
    }


def test_candidate_unit_core_semantics_can_survive_grounding_overlap_gate() -> None:
    mechanism = nx.DiGraph()
    mechanism.add_node("G1", type="MechanismClaim", label="canonical coordination mechanism")
    mechanism.add_node("G2", type="ObservationClaim", label="canonical HER observation")
    mechanism.add_edge("G1", "G2", relation="INTERPRETED_AS", edge_class="scientific_confirmed")
    grounding_path = {
        "path_id": "ground",
        "nodes": ["G1", "G2"],
        "steps": [_step("G1", "G2", "INTERPRETED_AS")],
        "visited_paper_ids": ["P1"],
        "hop_count": 1,
        "path_quality": {"path_type": "DIRECT_MECHANISTIC", "mechanistic_content": "high", "endpoint_pair_score": 0.8},
    }
    mechanism_payload = {
        "corpus_id": "c",
        "mode": "mechanism",
        "source_query": "nitrogen coordination",
        "target_query": "HER activity",
        "paths": [grounding_path],
        "candidate_paths": [grounding_path],
    }

    exploratory = nx.DiGraph()
    for node, typ in [("G1", "MechanismClaim"), ("A", "CatalystModel"), ("C", "BridgeConcept"), ("B", "MechanismClaim"), ("G2", "ObservationClaim")]:
        exploratory.add_node(node, type=typ, label=node)
    candidate_steps = [
        _step("G1", "A", "APPLIES_TO"),
        _step("A", "C", "GROUNDS_SEMANTIC_CANDIDATE", candidate=True),
        _step("C", "B", "GROUNDS_SEMANTIC_CANDIDATE", candidate=True),
        _step("B", "G2", "INTERPRETED_AS"),
    ]
    for step in candidate_steps:
        exploratory.add_edge(step["source"], step["target"], **step)
    candidate_path = {
        "path_id": "candidate",
        "nodes": ["G1", "A", "C", "B", "G2"],
        "steps": candidate_steps,
        "visited_paper_ids": ["P1", "P2"],
        "hop_count": 4,
        "candidate_edge_count": 2,
        "candidate_unit_count": 1,
        "requires_verification": True,
        "candidate_unit_core_node_ids": ["A", "C", "B"],
        "candidate_unit_semantic_text": "new geometry formation-energy bridge | A | B",
        "candidate_unit": {
            "unit_id": "u1",
            "label": "new geometry formation-energy bridge",
            "entry_anchor_id": "A",
            "entry_anchor_label": "A",
            "exit_anchor_id": "B",
            "exit_anchor_label": "B",
            "proposed_subject": "geometry",
            "proposed_relation": "modulates",
            "proposed_object": "formation energy",
        },
        "candidate_unit_selection": {"total": 0.8, "reaction_switch_penalty": 0.0},
        "path_quality": {
            "path_type": "CANDIDATE_EXPLORATION",
            "mechanistic_content": "high",
            "endpoint_pair_score": 0.8,
            "candidate_fraction": 0.5,
            "navigation_edge_fraction": 0.0,
            "reverse_fraction": 0.0,
        },
    }
    exploratory_payload = {
        "corpus_id": "c",
        "mode": "exploratory",
        "source_query": "nitrogen coordination",
        "target_query": "HER activity",
        "paths": [candidate_path],
        "candidate_paths": [candidate_path],
    }

    # Grounding mean is [1,0]. Candidate full-path mean would still be very
    # close to grounding, while its core contains a distinct [0,1] bridge.
    mechanism_index = FakeIndex(
        records=[{"node_id": "G1"}, {"node_id": "G2"}],
        embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    exploratory_index = FakeIndex(
        records=[{"node_id": x} for x in ["G1", "A", "C", "B", "G2"]],
        embeddings=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    bundle = DiscoveryBundleBuilder(
        DiscoveryPolicy(top_k=2, max_grounding_semantic_similarity=0.95, min_exploration_score=0.0)
    ).build(
        [
            ("mechanism.json", mechanism_payload, mechanism),
            ("candidate.json", exploratory_payload, exploratory),
        ],
        semantic_indexes={"mechanism.json": mechanism_index, "candidate.json": exploratory_index},
    )
    assert bundle.policy_version == "discovery-policy-v3"
    assert bundle.selected_count >= 1
    item = next(item for item in bundle.inspirations if item.candidate_unit_id == "u1")
    assert item.path_type == "CANDIDATE_EXPLORATION"
    assert item.candidate_unit_score == 0.8
    assert item.semantic_similarity_to_grounding < 0.95
