
from pipeline_core.discovery.scientific_portfolio_closed_loop import (
    compile_residual_epistemic_state,
)


def _base():
    return (
        {
            "portfolio_id": "p",
            "hypotheses": [
                {"hypothesis_id": "h", "title": "H"}
            ],
        },
        {
            "source_portfolio_id": "p",
            "claims": [
                {
                    "hypothesis_id": "h",
                    "claims": [
                        {
                            "claim_id": "c",
                            "importance": "core",
                            "novelty_selection_role": "NOVELTY_BEARING",
                        }
                    ],
                }
            ],
        },
        {
            "source_portfolio_id": "p",
            "cards": [
                {
                    "hypothesis_id": "h",
                    "status": "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                }
            ],
        },
    )


def test_clean_residual_becomes_feedback_keep_candidate():
    portfolio, query, external = _base()
    result = compile_residual_epistemic_state(
        portfolio=portfolio,
        query_plan=query,
        external_report=external,
        aggregation={
            "composites": [
                {
                    "hypothesis_id": "h",
                    "claim_id": "c",
                    "aggregation_disposition": "RESIDUAL_CANDIDATE_SHADOW",
                }
            ]
        },
    )
    assert (
        result["hypotheses"][0]["final_epistemic_state"]
        == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW"
    )


def test_known_core_relation_routes_to_prior_art_state():
    portfolio, query, external = _base()
    result = compile_residual_epistemic_state(
        portfolio=portfolio,
        query_plan=query,
        external_report=external,
        aggregation={
            "composites": [
                {
                    "hypothesis_id": "h",
                    "claim_id": "c",
                    "aggregation_disposition": "NO_RESIDUAL",
                }
            ]
        },
    )
    assert (
        result["hypotheses"][0]["final_epistemic_state"]
        == "PRIOR_ART_BACKED_OR_NO_RESIDUAL"
    )


def test_supporting_topology_hold_does_not_block_clean_core_residual():
    portfolio, query, external = _base()
    query["claims"][0]["claims"].append(
        {
            "claim_id": "s",
            "importance": "supporting",
            "novelty_selection_role": "TESTING_PREDICTION",
        }
    )
    result = compile_residual_epistemic_state(
        portfolio=portfolio,
        query_plan=query,
        external_report=external,
        aggregation={
            "composites": [
                {
                    "hypothesis_id": "h",
                    "claim_id": "c",
                    "aggregation_disposition": "RESIDUAL_CANDIDATE_SHADOW",
                },
                {
                    "hypothesis_id": "h",
                    "claim_id": "s",
                    "aggregation_disposition": "HOLD_FOR_TOPOLOGY",
                },
            ]
        },
    )
    row = result["hypotheses"][0]
    assert row["final_epistemic_state"] == "RESIDUAL_AUTHORITY_CANDIDATE_SHADOW"
    assert row["nonblocking_hold_count"] == 1


def test_insufficient_search_evidence_blocks_residual_authority():
    portfolio, query, external = _base()

    external["cards"][0]["status"] = (
        "INSUFFICIENT_SEARCH_EVIDENCE"
    )

    result = compile_residual_epistemic_state(
        portfolio=portfolio,
        query_plan=query,
        external_report=external,
        aggregation={
            "composites": [
                {
                    "hypothesis_id": "h",
                    "claim_id": "c",
                    "aggregation_disposition": (
                        "RESIDUAL_CANDIDATE_SHADOW"
                    ),
                }
            ]
        },
    )

    row = result["hypotheses"][0]

    assert (
        row["final_epistemic_state"]
        == "UNRESOLVED_EVIDENCE_GAP"
    )
    assert (
        row["state_reason"]
        == "insufficient_external_search_evidence_for_residual_authority"
    )
    assert row["authority_ready_candidate_shadow"] is False
