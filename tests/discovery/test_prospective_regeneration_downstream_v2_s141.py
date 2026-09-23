from __future__ import annotations

import pytest

from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    ProspectiveRegenerationDownstreamV2Policy,
    RegenerationDownstreamStageBudgetV2,
    build_regeneration_downstream_v2_freeze,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)


def _unit_freeze():
    return build_regeneration_unit_v2_freeze(
        repository_head_sha="a" * 40,
        repository_tracked_worktree_dirty=False,
    )


def test_budget_is_stage_invocation_not_raw_provider_call_count() -> None:
    budget = RegenerationDownstreamStageBudgetV2()
    assert budget.budget_unit == "semantic_stage_invocation"
    assert budget.provider_parse_retries_max == 3


def test_downstream_forbids_full_e2e_and_upstream_reruns() -> None:
    policy = ProspectiveRegenerationDownstreamV2Policy()
    assert policy.full_e2e_rerun_allowed is False
    assert policy.upstream_graph_exploration_rerun_allowed is False
    assert policy.upstream_context_construction_rerun_allowed is False
    assert policy.upstream_discovery_axis_generation_rerun_allowed is False
    assert policy.initial_hypothesis_generation_rerun_allowed is False


def test_downstream_preserves_regenerated_portfolio_identity() -> None:
    policy = ProspectiveRegenerationDownstreamV2Policy()
    assert policy.regenerated_portfolio_mutation_allowed is False
    assert policy.regenerated_portfolio_id_must_remain_stable is True
    assert policy.frozen_hypothesis_context_id_must_remain_stable is True


def test_downstream_stage_sequence_is_bounded() -> None:
    budget = RegenerationDownstreamStageBudgetV2()
    assert budget.semantic_critic_stage_max == 1
    assert budget.external_novelty_stage_max == 1
    assert budget.n9_shadow_intake_stage_max == 1
    assert budget.n9_full_closure_stage_max == 1
    assert budget.n10_certification_stage_max == 1
    assert budget.binding_plan_build_max == 1
    assert budget.preverifier_gate_v2_max == 1


def test_continuation_refinement_and_second_regeneration_are_zero() -> None:
    budget = RegenerationDownstreamStageBudgetV2()
    assert budget.targeted_novelty_continuation_max == 0
    assert budget.novelty_refinement_max == 0
    assert budget.same_lineage_hypothesis_regeneration_max == 0
    assert budget.semantic_repair_stage_max == 0


def test_downstream_failure_is_terminal_for_regenerated_lineage() -> None:
    policy = ProspectiveRegenerationDownstreamV2Policy()
    assert policy.downstream_failure_retry_allowed is False
    assert policy.downstream_failure_triggers_second_regeneration is False


def test_freeze_records_protocol_fix_not_scientific_tuning() -> None:
    frozen = build_regeneration_downstream_v2_freeze(
        regeneration_unit_freeze=_unit_freeze(),
        regeneration_unit_freeze_file_sha256="b" * 64,
        repository_head_sha="c" * 40,
        repository_tracked_worktree_dirty=False,
    )
    assert frozen.frozen_before_new_cohort_source_tasks is True
    assert frozen.frozen_before_new_cohort_generation is True
    assert (
        frozen.s135_protocol_failure_used_to_separate_regeneration_and_downstream
        is True
    )
    assert (
        frozen.s135_scientific_outputs_used_to_tune_downstream_content
        is False
    )


def test_freeze_requires_clean_tracked_worktree() -> None:
    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_regeneration_downstream_v2_freeze(
            regeneration_unit_freeze=_unit_freeze(),
            regeneration_unit_freeze_file_sha256="b" * 64,
            repository_head_sha="c" * 40,
            repository_tracked_worktree_dirty=True,
        )
