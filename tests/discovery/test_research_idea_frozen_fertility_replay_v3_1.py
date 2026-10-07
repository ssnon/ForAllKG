from __future__ import annotations

from copy import deepcopy

from pipeline_core.discovery.research_idea_adaptive_fertility import (
    build_adaptive_fertility_report,
)
from pipeline_core.discovery.research_idea_frozen_fertility_replay import (
    compare_frozen_fertility_audits,
    replay_frozen_fertility,
)


def _snapshot(*, usable_b: int = 0, exhausted_b: bool = True):
    parallel = {
        "report_id": "parallel:1",
        "members": [
            {
                "member_id": "m:a",
                "idea_id": "idea:a",
                "epistemic_realization_id": "e:a",
                "epistemic_maturity": "STRICT_GROUNDED",
            },
            {
                "member_id": "m:b",
                "idea_id": "idea:b",
                "epistemic_realization_id": "e:b",
                "epistemic_maturity": "EVIDENCE_SEEKING",
            },
        ],
        "epistemic_debts": [
            {
                "debt_id": "d:b",
                "idea_id": "idea:b",
                "requirement_kind": "EVIDENCE_COVERAGE_GAP",
                "disposition": "SEARCH_DEBT",
            }
        ],
        "evolution_handoffs": [
            {
                "handoff_id": "h:b",
                "idea_id": "idea:b",
                "recommended_channels": ["TRANSFORM", "EXPLORE"],
                "nonbinding_operator_hints": ["PROXY_CHALLENGE"],
                "reason_codes": ["TEST_HANDOFF"],
            }
        ],
        "idea_states": [
            {
                "idea_id": "idea:a",
                "generation_index": 5,
                "active_member_ids": ["m:a"],
                "remains_in_search_population": True,
            },
            {
                "idea_id": "idea:b",
                "generation_index": 5,
                "active_member_ids": ["m:b"],
                "remains_in_search_population": True,
            },
        ],
    }
    lifecycle = {
        "report_id": "life:1",
        "generation_index": 5,
        "local_states": [
            {
                "idea_id": "idea:a",
                "realization_count": 1,
                "usable_grounded_realization_count": 1,
                "failed_or_abstained_count": 0,
                "not_operationalizable_count": 0,
                "local_search_budget": 2,
                "local_search_budget_used": 1,
                "local_search_exhausted": False,
                "rescued_within_same_idea": False,
            },
            {
                "idea_id": "idea:b",
                "realization_count": 2 if exhausted_b else 1,
                "usable_grounded_realization_count": usable_b,
                "failed_or_abstained_count": 2 if usable_b == 0 else 1,
                "not_operationalizable_count": 0,
                "local_search_budget": 2,
                "local_search_budget_used": 2 if exhausted_b else 1,
                "local_search_exhausted": exhausted_b and usable_b == 0,
                "rescued_within_same_idea": False,
            },
        ],
    }
    return parallel, lifecycle


def test_replay_exactly_reproduces_recorded_v31_report():
    parallel, lifecycle = _snapshot()
    recorded = build_adaptive_fertility_report(
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    audit = replay_frozen_fertility(
        snapshot_label="recorded",
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=5,
        recorded_fertility_report=recorded,
    )
    assert audit.replay_repeat_exact_match is True
    assert audit.recorded_report_exact_match is True
    assert audit.recorded_decision_match_count == 2
    assert audit.recorded_decision_mismatch_count == 0


def test_replay_is_policy_only_and_zero_call():
    parallel, lifecycle = _snapshot()
    audit = replay_frozen_fertility(
        snapshot_label="zero-call",
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    assert audit.llm_call_count == 0
    assert audit.external_search_executed is False
    assert audit.evidence_acquisition_executed is False
    assert audit.canonical_graph_mutated is False
    assert audit.policy_only_replay is True


def test_policy_input_fingerprint_is_stable_under_irrelevant_diagnostic_fields():
    parallel, lifecycle = _snapshot()
    left = replay_frozen_fertility(
        snapshot_label="left",
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    mutated_parallel = deepcopy(parallel)
    mutated_parallel["diagnostic_only_field"] = {"anything": [1, 2, 3]}
    mutated_lifecycle = deepcopy(lifecycle)
    mutated_lifecycle["unrelated_summary"] = "ignored"
    right = replay_frozen_fertility(
        snapshot_label="right",
        parallel_report=mutated_parallel,
        lifecycle=mutated_lifecycle,
        cycle_generation_index=5,
    )
    assert left.policy_input_sha256 == right.policy_input_sha256
    assert left.replay_fertility_report.model_dump(mode="json") == right.replay_fertility_report.model_dump(mode="json")


def test_policy_input_fingerprint_changes_when_fertility_relevant_state_changes():
    parallel, lifecycle = _snapshot()
    left = replay_frozen_fertility(
        snapshot_label="unresolved",
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    _, grounded_lifecycle = _snapshot(usable_b=1, exhausted_b=False)
    right = replay_frozen_fertility(
        snapshot_label="grounded",
        parallel_report=parallel,
        lifecycle=grounded_lifecycle,
        cycle_generation_index=5,
    )
    assert left.policy_input_sha256 != right.policy_input_sha256
    assert {row.idea_id: row.disposition for row in left.replay_fertility_report.decisions}["idea:b"] == "EVOLVE_CHILD"
    assert {row.idea_id: row.disposition for row in right.replay_fertility_report.decisions}["idea:b"] == "HOLD_STABLE"


def test_comparison_separates_same_policy_from_changed_frozen_state():
    parallel, lifecycle = _snapshot()
    left = replay_frozen_fertility(
        snapshot_label="old-state",
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    _, grounded_lifecycle = _snapshot(usable_b=1, exhausted_b=False)
    right = replay_frozen_fertility(
        snapshot_label="new-state",
        parallel_report=parallel,
        lifecycle=grounded_lifecycle,
        cycle_generation_index=5,
    )
    comparison = compare_frozen_fertility_audits(left, right)
    assert comparison["same_policy_input"] is False
    assert comparison["changed_decision_count"] == 1
    assert comparison["changed_idea_ids"] == ["idea:b"]
    assert comparison["causal_superiority_asserted"] is False


def test_missing_recorded_report_is_valid_for_v30_counterfactual_replay():
    parallel, lifecycle = _snapshot()
    audit = replay_frozen_fertility(
        snapshot_label="v3.0-counterfactual",
        parallel_report=parallel,
        lifecycle=lifecycle,
        cycle_generation_index=5,
        recorded_fertility_report=None,
    )
    assert audit.recorded_fertility_report_id is None
    assert audit.recorded_report_exact_match is None
    assert audit.recorded_decision_match_count == 0
    assert audit.recorded_decision_mismatch_count == 0
