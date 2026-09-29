from __future__ import annotations

import pytest

from pipeline_core.discovery.discovery_contracts import DiscoveryBundle
from scripts.discovery.build_task_conditioned_axis_plan import (
    _a17f_assert_authoritative_replay_equivalence,
)


def _bundle(*, warnings, score=0.5):
    payload = {
        "schema_version": "discovery-bundle-v1",
        "bundle_id": "discovery_bundle:test",
        "corpus_id": "corpus",
        "domain_profile_id": "dac_her",
        "query_signature": "q",
        "inspirations": [
            {
                "inspiration_id": "discovery_inspiration:test",
                "source_path_id": "path:test",
                "source_corpus_id": "corpus",
                "source_mode": "exploratory",
                "path_type": "CANDIDATE_EXPLORATION",
                "paper_ids": ["P1"],
                "node_ids": ["N1", "N2"],
                "edge_ids": ["E1"],
                "relation_sequence": ["REL"],
                "rendered_path": "N1 --REL--> N2",
                "exploration_score": score,
                "score_breakdown": {
                    "endpoint_relevance": 0.5,
                    "mechanistic_content": 0.5,
                    "cross_paper_span": 0.0,
                    "community_span": 0.0,
                    "relation_rarity": 0.0,
                    "exploratory_mode_bonus": 0.0,
                    "mechanistic_continuity": 0.5,
                    "grounding_redundancy_penalty": 0.0,
                    "semantic_grounding_redundancy_penalty": 0.0,
                    "navigation_burden_penalty": 0.0,
                    "reverse_burden_penalty": 0.0,
                    "generic_entity_burden_penalty": 0.0,
                    "registry_hop_penalty": 0.0,
                    "candidate_unit_quality": 0.5,
                    "context_switch_penalty": 0.0,
                    "reaction_domain_switch_penalty": 0.0,
                    "total": score,
                },
                "reason_codes": ["bundle_rank:1"],
                "requires_verification": True,
                "mechanism_before_alignment": True,
                "mechanism_after_alignment": True,
                "mechanistic_continuity_band": "high",
                "generic_entity_fraction": 0.0,
                "max_generic_run_length": 0,
                "registry_hop_fraction": 0.0,
                "semantic_similarity_to_grounding": 0.1,
                "max_semantic_similarity_to_selected": 0.0,
                "semantic_diversity_mode": "node_embedding",
                "candidate_unit_id": "candidate_unit:test",
                "candidate_unit_label": "candidate",
                "candidate_entry_anchor_id": "N1",
                "candidate_entry_anchor_label": "N1",
                "candidate_exit_anchor_id": "N2",
                "candidate_exit_anchor_label": "N2",
                "candidate_proposed_subject": "A",
                "candidate_proposed_relation": "REL",
                "candidate_proposed_object": "B",
                "candidate_unit_score": 0.5,
                "context_switch_penalty": 0.0,
                "reaction_domain_switch_penalty": 0.0,
            }
        ],
        "source_traversal_files": ["a.json", "b.json"],
        "candidate_count": 10,
        "selected_count": 1,
        "used_candidate_pool": True,
        "warnings": list(warnings),
        "semantic_diversity_mode": "node_embedding",
        "semantic_model_name": "model",
        "semantic_similarity_threshold": 0.88,
        "policy_version": "discovery-policy-v3",
        "bundle_sha256": "x" * 64,
    }
    return DiscoveryBundle.model_validate(payload)


def test_a17f_replay_guard_allows_warning_only_metadata_drift():
    expected = _bundle(
        warnings=[
            "diagnostic-a",
            "diagnostic-b",
            "diagnostic-c",
        ]
    )
    replay = _bundle(
        warnings=[
            "diagnostic-a",
            "diagnostic-b",
        ]
    ).model_copy(
        update={"bundle_sha256": "y" * 64}
    )

    removed, added = (
        _a17f_assert_authoritative_replay_equivalence(
            expected_bundle=expected,
            replay_bundle=replay,
        )
    )

    assert removed == ["diagnostic-c"]
    assert added == []


def test_a17f_replay_guard_rejects_scientific_payload_drift():
    expected = _bundle(
        warnings=["diagnostic-a"],
        score=0.5,
    )
    replay = _bundle(
        warnings=["diagnostic-a"],
        score=0.6,
    ).model_copy(
        update={"bundle_sha256": "y" * 64}
    )

    with pytest.raises(
        RuntimeError,
        match="authoritative DiscoveryBundle payload",
    ):
        _a17f_assert_authoritative_replay_equivalence(
            expected_bundle=expected,
            replay_bundle=replay,
        )
