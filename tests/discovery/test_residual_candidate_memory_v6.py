
from __future__ import annotations

from pipeline_core.discovery.prior_art_candidate_memory_shadow import (
    bindings_compatible,
    build_candidate_memory,
)
from pipeline_core.discovery.residual_review_stability_shadow import (
    compare_residual_runs,
)
from pipeline_core.discovery.residual_authority_readiness_shadow import (
    build_authority_readiness,
)


def test_candidate_memory_requires_exact_proposition_identity():
    historical = {
        "hypothesis_id": "h",
        "proposition_basis": "A changes B.",
        "relation_endpoint_anchors": ["A", "B"],
        "scope_qualifier_spans": [],
        "directional_qualifier_spans": [],
    }
    current = {
        "hypothesis_id": "h",
        "proposition_basis": "A changes B",
        "relation_endpoint_anchors": ["A", "B"],
        "scope_qualifier_spans": [],
        "directional_qualifier_spans": [],
    }
    assert bindings_compatible(historical, current)

    current["proposition_basis"] = "A may change B under X"
    assert not bindings_compatible(historical, current)


def test_candidate_memory_reexposes_but_does_not_create_evidence():
    hist_bundle = {
        "records": [
            {
                "hypothesis_id": "h",
                "claim_id": "old",
                "claim_local_id": "c1",
                "proposition_basis": "A changes B",
                "relation_endpoint_anchors": ["A", "B"],
                "scope_qualifier_spans": [],
                "directional_qualifier_spans": [],
            }
        ]
    }
    cur_bundle = {
        "records": [
            {
                "hypothesis_id": "h",
                "claim_id": "new",
                "claim_local_id": "c1",
                "proposition_basis": "A changes B",
                "relation_endpoint_anchors": ["A", "B"],
                "scope_qualifier_spans": [],
                "directional_qualifier_spans": [],
            }
        ]
    }
    hist_packet = {
        "works": [
            {
                "work_id": "w1",
                "title": "Work",
                "doi": "10.1/x",
                "retrieval_claim_ids": ["old"],
            }
        ]
    }
    cur_packet = {"works": []}
    hist_reviews = {"reviews": []}
    cur_reviews = {"reviews": [{"claim_id": "new", "matches": []}]}

    report, packet, reviews = build_candidate_memory(
        historical_source_binding=hist_bundle,
        historical_packet=hist_packet,
        historical_reviews=hist_reviews,
        current_source_binding=cur_bundle,
        current_packet=cur_packet,
        current_reviews=cur_reviews,
    )
    assert report["memory_is_evidence"] is False
    assert report["reexposed_claim_work_pair_count"] == 1
    match = reviews["reviews"][0]["matches"][0]
    assert match["relationship"] == "COMPONENT_ONLY"
    assert match["candidate_memory_is_evidence"] is False
    assert packet["works"][0]["doi"] == "10.1/x"


def test_stability_checks_authority_relevant_backing_and_disposition():
    ft = {
        "targets": [
            {
                "claim_id": "a",
                "fulltext_relation_backed_work_ids": ["w"],
                "reviews": [
                    {
                        "work_id": "w",
                        "relationship": "PARTIAL_PRIOR_ART",
                    }
                ],
            }
        ]
    }
    agg = {
        "composites": [
            {
                "claim_id": "comp",
                "hypothesis_id": "h",
                "aggregation_disposition": "RESIDUAL_CANDIDATE_SHADOW",
                "aggregated_residual_state": (
                    "HIGHER_ORDER_RESIDUAL_SAME_WORK_KNOWN_BASE"
                ),
                "aggregated_relation_backed_component_claim_ids": ["a"],
            }
        ]
    }
    result = compare_residual_runs(
        fulltext_a=ft,
        fulltext_b=ft,
        aggregation_a=agg,
        aggregation_b=agg,
    )
    assert result["authority_relevant_stable"] is True


def test_authority_readiness_never_grants_authority():
    agg = {
        "composites": [
            {
                "claim_id": "comp",
                "hypothesis_id": "h",
                "aggregation_disposition": "RESIDUAL_CANDIDATE_SHADOW",
                "topology_state": "EXPLICIT",
                "aggregated_component_claim_ids": ["a"],
                "aggregated_relation_backed_component_claim_ids": ["a"],
                "fulltext_unavailable_component_claim_ids": [],
            }
        ]
    }
    stability = {
        "authority_relevant_stable": True,
        "composites": [
            {"claim_id": "comp", "stable": True}
        ],
    }
    audit = {"pass": True}
    result = build_authority_readiness(
        aggregation_a=agg,
        aggregation_b=agg,
        stability=stability,
        cohort_audit_a=audit,
        cohort_audit_b=audit,
        candidate_memory={
            "reexposed_current_claim_count": 1
        },
    )
    assert result["authority_ready_candidate_count"] == 1
    assert result["novelty_authority_created"] is False
    assert result["n9_authority_created"] is False
