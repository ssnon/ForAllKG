from __future__ import annotations

from pipeline_core.discovery.reframing.atomic_synthesis_n10_smoke import (
    ready_for_closure_count,
    summarize_n9_states,
)


def test_n9_state_summary_counts_atomic_claims():
    payload = {
        "hypotheses": [
            {
                "claims": [
                    {"shadow_state": "READY_FOR_CLOSURE"},
                    {"shadow_state": "NEEDS_REFINEMENT"},
                ]
            },
            {
                "claims": [
                    {"shadow_state": "READY_FOR_CLOSURE"},
                ]
            },
        ]
    }
    assert summarize_n9_states(payload) == {
        "NEEDS_REFINEMENT": 1,
        "READY_FOR_CLOSURE": 2,
    }
    assert ready_for_closure_count(payload) == 2


def test_ready_count_is_zero_when_all_claims_need_refinement():
    payload = {
        "hypotheses": [
            {
                "claims": [
                    {"shadow_state": "NEEDS_REFINEMENT"},
                    {"shadow_state": "NEEDS_REFINEMENT"},
                ]
            }
        ]
    }
    assert ready_for_closure_count(payload) == 0
