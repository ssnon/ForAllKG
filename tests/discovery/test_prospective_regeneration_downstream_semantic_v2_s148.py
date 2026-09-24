from pipeline_core.discovery.prospective_regeneration_downstream_v2 import ProspectiveRegenerationDownstreamV2Policy


def test_downstream_semantic_budget_is_single_invocation():
    policy = ProspectiveRegenerationDownstreamV2Policy()
    assert policy.budget.deterministic_hypothesis_benchmark_max == 1
    assert policy.budget.semantic_critic_stage_max == 1


def test_downstream_semantic_failure_has_no_retry():
    policy = ProspectiveRegenerationDownstreamV2Policy()
    assert policy.downstream_failure_retry_allowed is False
    assert policy.downstream_failure_triggers_second_regeneration is False


def test_downstream_semantic_cannot_mutate_regenerated_portfolio():
    policy = ProspectiveRegenerationDownstreamV2Policy()
    assert policy.regenerated_portfolio_mutation_allowed is False
    assert policy.regenerated_portfolio_id_must_remain_stable is True
