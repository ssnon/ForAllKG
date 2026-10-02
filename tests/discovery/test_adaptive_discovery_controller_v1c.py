from pipeline_core.discovery.sers_closed_loop_decision_consolidation_v2 import (
    consolidate_post_verification_decisions,
)


def test_insufficient_search_evidence_cannot_advance_residual():
    result = consolidate_post_verification_decisions(
        gen1_portfolio={
            "portfolio_id": "p",
            "hypotheses": [
                {
                    "hypothesis_id": "h",
                    "title": "H",
                }
            ],
        },
        generation_report={
            "records": [
                {
                    "source_hypothesis_id": "source",
                    "generated_hypothesis_id": "h",
                    "route": "AXIS_MUTATION",
                    "decision": "ACCEPTED_GENERATION_SHADOW",
                }
            ],
        },
        query_plan={
            "source_portfolio_id": "p",
            "claims": [
                {
                    "hypothesis_id": "h",
                    "claims": [
                        {
                            "claim_id": "c",
                            "importance": "core",
                            "novelty_selection_role": (
                                "NOVELTY_BEARING"
                            ),
                        }
                    ],
                }
            ],
        },
        external_report={
            "source_portfolio_id": "p",
            "cards": [
                {
                    "hypothesis_id": "h",
                    "status": "INSUFFICIENT_SEARCH_EVIDENCE",
                    "novelty_depth_profile": {
                        "novelty_bearing_prior_art_state": "MIXED",
                    },
                }
            ],
        },
        aggregation={
            "composites": [
                {
                    "hypothesis_id": "h",
                    "claim_id": "c",
                    "aggregation_disposition": (
                        "RESIDUAL_CANDIDATE_SHADOW"
                    ),
                    "aggregated_residual_state": (
                        "HIGHER_ORDER_RESIDUAL_DISTRIBUTED_KNOWN_BASE"
                    ),
                    "full_relation_status": "COMPONENTS_ONLY",
                    "topology_state": "EXPLICIT",
                    "evidence_depth": "METADATA_ONLY",
                    "aggregated_component_claim_ids": ["a", "b"],
                    "aggregated_relation_backed_component_claim_ids": [
                        "a",
                        "b",
                    ],
                }
            ],
        },
        cohort_audit={"pass": True},
    )

    row = result["rows"][0]

    assert (
        row["post_verification_decision"]
        == "HOLD_CORE_EVIDENCE_UNRESOLVED"
    )
    assert row["advance_to_effective_gen1"] is False
    assert (
        "insufficient_external_search_evidence_for_residual_authority"
        in row["reason_codes"]
    )
