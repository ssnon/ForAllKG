from __future__ import annotations

from dac_her.alpha4c5f_reserve import (
    EXPECTED_RESERVE_SET,
    Alpha4c5fExecutionPolicy,
    Alpha4c5fTraversalPolicy,
)


def test_v3_reserve_has_exactly_14_unique_papers():
    assert len(EXPECTED_RESERVE_SET) == 14


def test_execution_policy_is_single_shot_and_count_free():
    policy = Alpha4c5fExecutionPolicy()
    assert policy.exact_reserve_set_required is True
    assert policy.paper_override_allowed is False
    assert policy.count_thresholds_used_for_acceptance is False
    assert policy.zero_trend_yield_is_execution_failure is False
    assert policy.zero_hypotheses_is_evaluation_failure is False
    assert (
        policy.reserve_consumed_before_first_scientific_transformation
        is True
    )
    assert policy.rerun_after_consumption_allowed is False
    assert policy.automatic_scientific_output_rollback is False


def test_execution_is_evidence_mode_without_bridge():
    policy = Alpha4c5fExecutionPolicy()
    assert policy.evidence_mode_only is True
    assert policy.bridge_required is False
    assert policy.new_extraction_llm_allowed is False


def test_traversal_defaults_are_frozen_conservative_top_n():
    traversal = Alpha4c5fTraversalPolicy(
        source_query="nanostructure design",
        target_query="SERS performance",
        node_index_model="test-model",
    )
    assert traversal.mode == "evidence"
    assert traversal.algorithm == "top_n"
    assert traversal.max_depth == 8
    assert traversal.top_k == 8
    assert traversal.node_map_k == 20
    assert traversal.endpoint_pair_k == 12
    assert traversal.include_alignment_hubs_in_index is False
