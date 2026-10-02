from scripts.discovery.run_adaptive_discovery_controller_shadow import (
    _append_history,
    _audit_failed_hypothesis_ids,
    _controller_summary_status,
    _mark_audit_failure_states,
)
from scripts.discovery.run_residual_novelty_cohort_audit_shadow import (
    build_audit_report,
)


def test_noncomposite_query_plan_makes_topology_not_applicable():
    report = build_audit_report(
        aggregation={"composites": []},
        completion={"records": []},
        bundle={"claim_count": 2},
        topology={
            "topology_residual": {
                "composites": [],
            }
        },
        query_plan={
            "claims": [
                {
                    "hypothesis_id": "h1",
                    "claims": [
                        {
                            "claim_id": "c1",
                            "kind": "moderator_interaction",
                        },
                        {
                            "claim_id": "c2",
                            "kind": "distinctive_prediction",
                        },
                    ],
                }
            ]
        },
    )

    assert report["pass"] is True
    assert report["expected_composite_claim_count"] == 0
    assert report["topology_composite_count"] == 0
    assert (
        report["topology_applicability"]
        == "NOT_APPLICABLE_NO_PLANNED_COMPOSITES"
    )
    assert (
        "topology_not_applicable_no_composite_claims"
        in report["warnings"]
    )
    assert "topology_composite_population_empty" not in report["failures"]


def test_planned_composite_missing_from_topology_fails_closed():
    report = build_audit_report(
        aggregation={"composites": []},
        completion={"records": []},
        bundle={"claim_count": 1},
        topology={
            "topology_residual": {
                "composites": [],
            }
        },
        query_plan={
            "claims": [
                {
                    "hypothesis_id": "h1",
                    "claims": [
                        {
                            "claim_id": "composite-1",
                            "kind": "composite",
                        }
                    ],
                }
            ]
        },
    )

    assert report["pass"] is False
    assert report["topology_applicability"] == "APPLICABLE"
    assert "topology_composite_population_empty" in report["failures"]
    assert any(
        row.startswith(
            "topology_missing_expected_composites:"
        )
        for row in report["failures"]
    )


def test_audit_failure_is_promoted_to_terminal_epistemic_hold():
    decisions = {
        "rows": [
            {
                "hypothesis_id": "h2",
                "post_verification_decision": (
                    "HOLD_COHORT_AUDIT_FAILED"
                ),
            }
        ]
    }

    failed = _audit_failed_hypothesis_ids(decisions)

    assert failed == {"h2"}

    state = {
        "state_counts": {
            "UNRESOLVED_TOPOLOGY_GAP": 1,
        },
        "hypotheses": [
            {
                "hypothesis_id": "h2",
                "final_epistemic_state": (
                    "UNRESOLVED_TOPOLOGY_GAP"
                ),
                "state_reason": "x",
                "authority_ready_candidate_shadow": False,
                "authority_readiness_state": (
                    "UNRESOLVED_TOPOLOGY_GAP"
                ),
            }
        ],
    }

    marked = _mark_audit_failure_states(
        state,
        failed,
    )

    row = marked["hypotheses"][0]

    assert row["pre_audit_epistemic_state"] == (
        "UNRESOLVED_TOPOLOGY_GAP"
    )
    assert row["final_epistemic_state"] == "AUDIT_FAILURE_HOLD"
    assert row["authority_readiness_state"] == "AUDIT_FAILURE_HOLD"
    assert marked["state_counts"] == {
        "AUDIT_FAILURE_HOLD": 1,
    }


def test_terminal_generation_record_is_not_duplicated_in_history():
    history = {}

    _append_history(
        history=history,
        root_by_source={"h1": "root"},
        plan={
            "decisions": [
                {
                    "current_hypothesis_id": "h1",
                    "root_hypothesis_id": "root",
                    "current_epistemic_state": (
                        "UNRESOLVED_TOPOLOGY_GAP"
                    ),
                    "action": "REQUEST_GRAPH_RETRAVERSAL",
                }
            ]
        },
        generation={
            "records": [
                {
                    "source_hypothesis_id": "h1",
                    "generated_hypothesis_id": None,
                    "route": "REQUEST_GRAPH_RETRAVERSAL",
                    "decision": "HELD_FOR_GRAPH_RETRAVERSAL",
                }
            ]
        },
        decisions={"rows": []},
        state={
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "final_epistemic_state": (
                        "UNRESOLVED_TOPOLOGY_GAP"
                    ),
                }
            ]
        },
        round_index=4,
    )

    assert len(history["root"]) == 1
    assert (
        history["root"][0]["generation_decision"]
        == "HELD_FOR_GRAPH_RETRAVERSAL"
    )


def test_controller_summary_statuses_are_specific():
    assert _controller_summary_status(
        audit_failures=[{"round_index": 2}],
        graph_handoffs=[],
        final_effective_count=0,
        terminal_stops=[],
    ) == "COMPLETE_FAIL_CLOSED_AUDIT"

    assert _controller_summary_status(
        audit_failures=[],
        graph_handoffs=[{"round_index": 4}],
        final_effective_count=0,
        terminal_stops=[],
    ) == "COMPLETE_WITH_GRAPH_RETRAVERSAL_HANDOFF"

    assert _controller_summary_status(
        audit_failures=[],
        graph_handoffs=[],
        final_effective_count=2,
        terminal_stops=[],
    ) == "COMPLETE_WITH_EFFECTIVE_CANDIDATES"

    assert _controller_summary_status(
        audit_failures=[],
        graph_handoffs=[],
        final_effective_count=0,
        terminal_stops=[{"reason": "bounded"}],
    ) == "COMPLETE_SEARCH_EXHAUSTED"
