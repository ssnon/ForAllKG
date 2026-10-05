from scripts.discovery.run_adaptive_discovery_controller_v2_shadow import (
    _latest_materialized_history_head,
)


def test_latest_materialized_history_head_skips_rejected_generated_tail():
    rows = [
        {
            "round_index": 1,
            "current_hypothesis_id": "h_seed",
            "action": "FRESH_CONTEXT_REAXIS",
            "epistemic_state": "PRIOR_ART_BACKED_OR_NO_RESIDUAL",
        },
        {
            "round_index": 2,
            "current_hypothesis_id": "h_valid",
            "action": "SAME_PREMISE_SHARPEN",
            "epistemic_state": "PRIOR_ART_BACKED_OR_NO_RESIDUAL",
        },
        {
            "round_index": 3,
            "current_hypothesis_id": "h_rejected_axis",
            "action": "AXIS_MUTATION",
            "epistemic_state": None,
        },
    ]

    current, index, stale = _latest_materialized_history_head(
        rows=rows,
        materialized_hypothesis_ids={"h_seed", "h_valid"},
    )

    assert current == "h_valid"
    assert index == 1
    assert stale == ["h_rejected_axis"]


def test_latest_materialized_history_head_uses_latest_valid_candidate():
    rows = [
        {"current_hypothesis_id": "h1", "epistemic_state": "UNRESOLVED_EVIDENCE_GAP"},
        {"current_hypothesis_id": "h2", "epistemic_state": "UNRESOLVED_EVIDENCE_GAP"},
        {"current_hypothesis_id": "h3", "epistemic_state": "UNRESOLVED_EVIDENCE_GAP"},
    ]

    current, index, stale = _latest_materialized_history_head(
        rows=rows,
        materialized_hypothesis_ids={"h1", "h2", "h3"},
    )

    assert current == "h3"
    assert index == 2
    assert stale == []


def test_latest_materialized_history_head_fails_closed_when_none_materialized():
    rows = [
        {"current_hypothesis_id": "h_rejected_1", "epistemic_state": None},
        {"current_hypothesis_id": "h_rejected_2", "epistemic_state": None},
    ]

    current, index, stale = _latest_materialized_history_head(
        rows=rows,
        materialized_hypothesis_ids=set(),
    )

    assert current is None
    assert index is None
    assert stale == ["h_rejected_1", "h_rejected_2"]
