
from __future__ import annotations

from pipeline_core.discovery.fulltext_lower_order_escalation_shadow import escalation_target_claim_ids
from pipeline_core.discovery.residual_novelty_aggregation_shadow import aggregate_residual_novelty


def test_escalation_targets_only_missing_explicit_components():
    report = {
        "topology_residual": {
            "composites": [
                {
                    "topology_state": "EXPLICIT",
                    "full_relation_status": "COMPONENTS_ONLY",
                    "residual_state": "PARTIAL_BASE_SATURATION",
                    "effective_component_claim_ids_shadow": ["a", "b"],
                    "relation_backed_component_claim_ids": ["a"],
                },
                {
                    "topology_state": "NO_COMPONENT_TOPOLOGY",
                    "full_relation_status": "COMPONENTS_ONLY",
                    "residual_state": "NOT_ASSESSABLE_NO_COMPONENT_TOPOLOGY",
                    "effective_component_claim_ids_shadow": [],
                    "relation_backed_component_claim_ids": [],
                },
            ]
        }
    }
    assert escalation_target_claim_ids(report) == ["b"]


def test_fulltext_backing_completes_known_base_without_granting_authority():
    topology = {
        "topology_residual": {
            "composites": [
                {
                    "claim_id": "comp",
                    "hypothesis_id": "h",
                    "full_relation_status": "COMPONENTS_ONLY",
                    "topology_state": "EXPLICIT",
                    "effective_component_claim_ids_shadow": ["a", "b"],
                    "relation_backed_component_claim_ids": ["a"],
                }
            ]
        }
    }
    reviews = {
        "reviews": [
            {
                "claim_id": "a",
                "matches": [
                    {"work_id": "w1", "relationship": "PARTIAL_PRIOR_ART"}
                ],
            },
            {"claim_id": "b", "matches": []},
        ]
    }
    escalation = {
        "targets": [
            {
                "claim_id": "b",
                "status": "FULLTEXT_RELATION_BACKED",
                "fulltext_relation_backed_work_ids": ["w2"],
            }
        ]
    }
    result = aggregate_residual_novelty(
        topology_report=topology,
        claim_reviews=reviews,
        fulltext_escalation=escalation,
    )
    row = result["composites"][0]
    assert row["aggregation_disposition"] == "RESIDUAL_CANDIDATE_SHADOW"
    assert row["aggregated_component_closure"] == "DISTRIBUTED"
    assert result["novelty_authority_created"] is False


def test_no_topology_holds_for_topology():
    topology = {
        "topology_residual": {
            "composites": [
                {
                    "claim_id": "comp",
                    "hypothesis_id": "h",
                    "full_relation_status": "COMPONENTS_ONLY",
                    "topology_state": "NO_COMPONENT_TOPOLOGY",
                    "effective_component_claim_ids_shadow": [],
                    "relation_backed_component_claim_ids": [],
                }
            ]
        }
    }
    result = aggregate_residual_novelty(
        topology_report=topology,
        claim_reviews={"reviews": []},
        fulltext_escalation={"targets": []},
    )
    assert result["composites"][0]["aggregation_disposition"] == "HOLD_FOR_TOPOLOGY"
