
from pipeline_core.discovery.sers_closed_loop_decision_consolidation_v2 import (
    build_replacement_map,
    consolidate_post_verification_decisions,
)


def _query(role="NOVELTY_BEARING", importance="core"):
    return {
        "source_portfolio_id": "p",
        "claims": [
            {
                "hypothesis_id": "h",
                "claims": [
                    {
                        "claim_id": "c",
                        "kind": "composite",
                        "importance": importance,
                        "novelty_selection_role": role,
                    }
                ],
            }
        ],
    }


def _portfolio():
    return {
        "portfolio_id": "p",
        "hypotheses": [
            {
                "hypothesis_id": "h",
                "title": "H",
            }
        ],
    }


def _external(depth_state="ALL_GAP_LIKE"):
    return {
        "source_portfolio_id": "p",
        "cards": [
            {
                "hypothesis_id": "h",
                "status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                "novelty_depth_profile": {
                    "novelty_bearing_prior_art_state": depth_state,
                },
            }
        ],
    }


def _generation(decision="CARRIED_FORWARD"):
    return {
        "report_id": "g",
        "records": [
            {
                "source_hypothesis_id": "s",
                "route": "KEEP_RESIDUAL",
                "decision": decision,
                "generated_hypothesis_id": "h",
            }
        ],
    }


def test_carried_candidate_can_be_rejected_by_fresh_prior_art():
    result = consolidate_post_verification_decisions(
        gen1_portfolio=_portfolio(),
        generation_report=_generation(),
        query_plan=_query(),
        external_report=_external("ALL_RELATION_BACKED"),
        aggregation={
            "composites": [
                {
                    "hypothesis_id": "h",
                    "claim_id": "c",
                    "full_relation_status": "DIRECT_PRIOR_ART",
                    "topology_state": "EXPLICIT",
                    "aggregation_disposition": "NO_RESIDUAL",
                    "aggregated_residual_state": (
                        "NO_RESIDUAL_FULL_RELATION_BACKED"
                    ),
                    "aggregated_component_claim_ids": ["a"],
                    "aggregated_relation_backed_component_claim_ids": ["a"],
                }
            ]
        },
        cohort_audit={"pass": True},
    )
    row = result["rows"][0]
    assert row["post_verification_decision"] == "REJECT_KNOWN_AXIS_REPEAT"
    assert row["advance_to_effective_gen1"] is False
    assert row["carried_forward_has_no_epistemic_immunity"] is True


def test_core_topology_hold_blocks_hypothesis():
    result = consolidate_post_verification_decisions(
        gen1_portfolio=_portfolio(),
        generation_report=_generation("ACCEPTED_GENERATION_SHADOW"),
        query_plan=_query(),
        external_report=_external(),
        aggregation={
            "composites": [
                {
                    "hypothesis_id": "h",
                    "claim_id": "c",
                    "full_relation_status": "COMPONENTS_ONLY",
                    "topology_state": "NO_COMPONENT_TOPOLOGY",
                    "aggregation_disposition": "HOLD_FOR_TOPOLOGY",
                    "aggregated_residual_state": (
                        "NOT_ASSESSABLE_NO_COMPONENT_TOPOLOGY"
                    ),
                    "aggregated_component_claim_ids": [],
                    "aggregated_relation_backed_component_claim_ids": [],
                }
            ]
        },
        cohort_audit={"pass": True},
    )
    assert (
        result["rows"][0]["post_verification_decision"]
        == "HOLD_CORE_TOPOLOGY_UNRESOLVED"
    )


def test_supporting_hold_does_not_block_clean_novelty_bearing_residual():
    query = {
        "source_portfolio_id": "p",
        "claims": [
            {
                "hypothesis_id": "h",
                "claims": [
                    {
                        "claim_id": "novel",
                        "kind": "composite",
                        "importance": "core",
                        "novelty_selection_role": "NOVELTY_BEARING",
                    },
                    {
                        "claim_id": "support",
                        "kind": "composite",
                        "importance": "supporting",
                        "novelty_selection_role": "AUXILIARY",
                    },
                ],
            }
        ],
    }
    aggregation = {
        "composites": [
            {
                "hypothesis_id": "h",
                "claim_id": "novel",
                "full_relation_status": "COMPONENTS_ONLY",
                "topology_state": "EXPLICIT",
                "aggregation_disposition": "RESIDUAL_CANDIDATE_SHADOW",
                "aggregated_residual_state": (
                    "HIGHER_ORDER_RESIDUAL_DISTRIBUTED_KNOWN_BASE"
                ),
                "aggregated_component_claim_ids": ["a", "b"],
                "aggregated_relation_backed_component_claim_ids": ["a", "b"],
            },
            {
                "hypothesis_id": "h",
                "claim_id": "support",
                "full_relation_status": "COMPONENTS_ONLY",
                "topology_state": "NO_COMPONENT_TOPOLOGY",
                "aggregation_disposition": "HOLD_FOR_TOPOLOGY",
                "aggregated_residual_state": (
                    "NOT_ASSESSABLE_NO_COMPONENT_TOPOLOGY"
                ),
                "aggregated_component_claim_ids": [],
                "aggregated_relation_backed_component_claim_ids": [],
            },
        ]
    }
    result = consolidate_post_verification_decisions(
        gen1_portfolio=_portfolio(),
        generation_report=_generation("ACCEPTED_GENERATION_SHADOW"),
        query_plan=query,
        external_report=_external(),
        aggregation=aggregation,
        cohort_audit={"pass": True},
    )
    row = result["rows"][0]
    assert (
        row["post_verification_decision"]
        == "ADVANCE_RESIDUAL_CANDIDATE_SHADOW"
    )
    assert "nonblocking_composite_remains_unresolved" in row["warnings"]


def test_no_composite_fails_closed():
    result = consolidate_post_verification_decisions(
        gen1_portfolio=_portfolio(),
        generation_report=_generation(),
        query_plan={
            "source_portfolio_id": "p",
            "claims": [],
        },
        external_report=_external(),
        aggregation={"composites": []},
        cohort_audit={"pass": True},
    )
    assert (
        result["rows"][0]["post_verification_decision"]
        == "HOLD_REVALIDATION_INCOMPLETE_NO_COMPOSITE"
    )


def test_replacement_map_uses_fresh_post_decision_for_carried_candidate():
    replacement = build_replacement_map(
        generation_report=_generation(),
        decisions={
            "rows": [
                {
                    "hypothesis_id": "h",
                    "post_verification_decision": "REJECT_KNOWN_AXIS_REPEAT",
                    "advance_to_effective_gen1": False,
                }
            ]
        },
    )
    assert (
        replacement["rows"][0]["final_transition"]
        == "REJECT_AFTER_FRESH_PRIOR_ART"
    )
