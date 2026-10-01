
from __future__ import annotations

from pipeline_core.discovery.candidate_selection_completeness_shadow import (
    audit_candidate_selection,
)
from pipeline_core.discovery.memory_tiered_fulltext_selection_shadow import (
    build_tiered_selection_plan,
)
from pipeline_core.discovery.residual_authority_readiness_v7_shadow import (
    build_authority_readiness_v7,
)


def _binding(claim_id: str, hypothesis_id: str = "h"):
    return {
        "hypothesis_id": hypothesis_id,
        "claim_id": claim_id,
        "claim_local_id": claim_id,
        "proposition_basis": "Au Al2O3 bilayers change Raman signal intensity",
        "relation_endpoint_anchors": ["Au Al2O3 bilayers", "Raman signal intensity"],
        "scope_qualifier_spans": [],
        "directional_qualifier_spans": [],
    }


def test_tiered_selection_reserves_memory_lanes_without_duplication():
    query_plan = {
        "claims": [
            {
                "claims": [
                    {
                        "claim_id": "new",
                        "text": "Au Al2O3 bilayers change Raman signal intensity",
                        "prior_art_identity_terms": ["Au", "Al2O3"],
                        "relation_nucleus_terms": ["change", "Raman signal intensity"],
                    }
                ]
            }
        ]
    }
    topology = {
        "topology_residual": {
            "composites": [
                {
                    "topology_state": "EXPLICIT",
                    "full_relation_status": "COMPONENTS_ONLY",
                    "residual_state": "UNSATURATED_BASE",
                    "effective_component_claim_ids_shadow": ["new"],
                    "relation_backed_component_claim_ids": [],
                }
            ]
        }
    }
    current_packet = {
        "works": [
            {
                "work_id": "cur",
                "title": "Au Al2O3 Raman",
                "doi": "10.1/cur",
                "abstract": "Au Al2O3 bilayers Raman signal",
                "retrieval_claim_ids": ["new"],
            }
        ]
    }
    memory_packet = {
        "works": [
            *current_packet["works"],
            {
                "work_id": "hist-a",
                "title": "Au Al2O3 bilayer Raman signal",
                "doi": "10.1/a",
                "abstract": "Au Al2O3 bilayers alter Raman signal intensity",
                "retrieval_claim_ids": [],
            },
            {
                "work_id": "hist-c",
                "title": "Au Al2O3 multilayer SERS",
                "doi": "10.1/c",
                "abstract": "Au Al2O3 bilayers and Raman signal intensity",
                "retrieval_claim_ids": [],
            },
        ]
    }
    historical_packet = {
        "works": [
            {
                "work_id": "hist-a",
                "title": "Au Al2O3 bilayer Raman signal",
                "doi": "10.1/a",
                "abstract": "",
                "retrieval_claim_ids": ["old"],
            },
            {
                "work_id": "hist-c",
                "title": "Au Al2O3 multilayer SERS",
                "doi": "10.1/c",
                "abstract": "",
                "retrieval_claim_ids": ["old"],
            },
        ]
    }
    plan = build_tiered_selection_plan(
        query_plan=query_plan,
        topology_report=topology,
        current_source_binding={"records": [_binding("new")]},
        current_packet=current_packet,
        current_reviews={"reviews": [{"claim_id": "new", "matches": []}]},
        memory_packet=memory_packet,
        historical_source_binding={"records": [_binding("old")]},
        historical_packet=historical_packet,
        historical_reviews={"reviews": [{"claim_id": "old", "matches": []}]},
        historical_fulltext={
            "targets": [
                {
                    "claim_id": "old",
                    "reviews": [{"work_id": "hist-a"}],
                }
            ]
        },
        current_slots=1,
        tier_a_slots=1,
        tier_b_slots=0,
        tier_c_slots=1,
    )
    selected = plan["targets"][0]["selected"]
    ids = [row["work_id"] for row in selected]
    assert len(ids) == len(set(ids))
    assert "cur" in ids
    assert "hist-a" in ids
    assert "hist-c" in ids


def test_positive_sentinel_missed_fails_selection_audit():
    plan = {
        "targets": [
            {
                "claim_id": "c",
                "selected": [],
                "total_cap": 4,
                "tier_available_counts": {},
                "tier_selected_counts": {},
            }
        ]
    }
    packet = {
        "works": [
            {"work_id": "w", "doi": "10.1/x"}
        ]
    }
    sentinel = {
        "claim_id": "c",
        "doi": "10.1/x",
        "relationship": "PARTIAL_PRIOR_ART",
        "status": "REVIEWED",
    }
    report = audit_candidate_selection(
        selection_plan=plan,
        packet=packet,
        sentinel=sentinel,
    )
    assert report["pass"] is False
    assert (
        "positive_diagnostic_sentinel_missed_by_selector"
        in report["failures"]
    )


def test_negative_sentinel_does_not_fail_selection_audit():
    report = audit_candidate_selection(
        selection_plan={
            "targets": [
                {
                    "claim_id": "c",
                    "selected": [],
                    "total_cap": 4,
                    "tier_available_counts": {},
                    "tier_selected_counts": {},
                }
            ]
        },
        packet={"works": []},
        sentinel={
            "claim_id": "c",
            "doi": "10.1/x",
            "relationship": "NOT_RELATION_BACKED",
            "status": "REVIEWED",
        },
    )
    assert report["pass"] is True


def test_v7_readiness_holds_base_ready_when_selection_fails():
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
        "composites": [{"claim_id": "comp", "stable": True}],
    }
    audit = {"pass": True}
    report = build_authority_readiness_v7(
        aggregation_a=agg,
        aggregation_b=agg,
        stability=stability,
        cohort_audit_a=audit,
        cohort_audit_b=audit,
        candidate_memory={"reexposed_current_claim_count": 1},
        selection_audit={
            "pass": False,
            "sentinel_relationship": "PARTIAL_PRIOR_ART",
        },
    )
    assert report["base_authority_ready_candidate_count"] == 1
    assert report["authority_ready_candidate_count"] == 0
    assert report["rows"][0]["state"] == "HOLD_SELECTION_COMPLETENESS"
