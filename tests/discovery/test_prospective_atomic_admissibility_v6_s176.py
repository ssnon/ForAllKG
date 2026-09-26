from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    build_p29_p33_spec_from_v4_infrastructure,
    build_prospective_atomic_admissibility_freeze_v5,
)
from pipeline_core.discovery.prospective_atomic_admissibility_v6 import (
    build_p34_p38_spec_from_v5_infrastructure,
    build_prospective_atomic_admissibility_comparison_collector_freeze_v6,
    build_prospective_atomic_admissibility_execution_plan_v6,
    build_prospective_atomic_admissibility_freeze_v6,
    default_p34_p38_tasks,
    materialize_downstream_argv_v6,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from tests.discovery.test_prospective_atomic_admissibility_campaign_freeze_v5_s174f import (
    _v4,
)


def _v5(tmp_path: Path):
    source, unit = _v4(tmp_path)
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v5"),
    )
    freeze = build_prospective_atomic_admissibility_freeze_v5(
        spec=spec,
        source_spec_sha256="4" * 64,
        infrastructure_freeze=source,
        infrastructure_freeze_file_sha256="5" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="3" * 64,
        repository_head_sha="a" * 40,
        repository_tracked_worktree_dirty=False,
    )
    return freeze, unit


def _v6(tmp_path: Path):
    v5, unit = _v5(tmp_path)
    spec = build_p34_p38_spec_from_v5_infrastructure(
        infrastructure_freeze=v5,
        campaign_root=str(tmp_path / "v6"),
    )
    freeze = build_prospective_atomic_admissibility_freeze_v6(
        spec=spec,
        source_spec_sha256="6" * 64,
        infrastructure_freeze=v5,
        infrastructure_freeze_file_sha256="7" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="3" * 64,
        repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
    )
    return freeze, unit


def _plan(tmp_path: Path):
    freeze, unit = _v6(tmp_path)
    unit_path = tmp_path / "regen.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")
    return build_prospective_atomic_admissibility_execution_plan_v6(
        campaign_freeze=freeze,
        campaign_freeze_file_sha256="8" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_path=unit_path,
        regeneration_unit_freeze_file_sha256="3" * 64,
        execution_plan_repository_head_sha="c" * 40,
        repository_tracked_worktree_dirty=False,
        settings=ProspectiveAuthorityExecutionSettingsV3(
            base_url="https://example.invalid/v1",
            api_key_env="FIXTURE_KEY",
        ),
    )


def test_v6_tasks_are_exactly_p34_p38_and_unique() -> None:
    tasks = default_p34_p38_tasks()
    assert [row.case_id for row in tasks] == [
        "P34", "P35", "P36", "P37", "P38"
    ]
    assert len({row.relation_family for row in tasks}) == 5


def test_v6_spec_uses_no_prior_outcomes(tmp_path: Path) -> None:
    v5, _ = _v5(tmp_path)
    spec = build_p34_p38_spec_from_v5_infrastructure(
        infrastructure_freeze=v5,
        campaign_root=str(tmp_path / "v6"),
    )
    assert spec.prior_outcome_artifacts_consumed_by_builder is False
    assert spec.prior_task_definitions_used_only_for_nonoverlap_guard is True
    assert spec.infrastructure_and_model_policy_only_inherited_from_v5 is True


def test_v6_freeze_preserves_fixed_case_denominator(tmp_path: Path) -> None:
    freeze, _ = _v6(tmp_path)
    assert freeze.case_ids == ["P34", "P35", "P36", "P37", "P38"]
    assert freeze.p34_p38_outputs_observed_before_freeze is False
    assert freeze.later_case_adaptation_allowed is False
    assert freeze.failed_or_terminal_case_replacement_allowed is False


def test_v6_execution_plan_is_fixed_order_and_non_authoritative(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    assert plan.case_ids == ["P34", "P35", "P36", "P37", "P38"]
    assert plan.case_order_fixed_p34_to_p38 is True
    assert plan.execution_authority_granted_by_this_plan is False
    assert plan.comparison_collector_contract_required_before_execution is True


def test_v6_execution_plan_supports_zero_hypothesis_downstream_path(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    for case in plan.cases:
        assert "--semantic-run" in case.downstream_campaign_argv_base
        assert "--semantic-review" not in case.downstream_campaign_argv_base
        assert "--save-prompts" in case.downstream_campaign_argv_base


def test_v6_semantic_review_materialization_is_file_existence_only(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    case = plan.cases[0]
    without = materialize_downstream_argv_v6(case)
    assert "--semantic-review" not in without

    review = Path(case.initial_semantic_review_path)
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text('{"fixture":true}\n', encoding="utf-8")
    with_review = materialize_downstream_argv_v6(case)
    assert "--semantic-review" in with_review


def test_v6_collector_freeze_is_before_execution(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    frozen = build_prospective_atomic_admissibility_comparison_collector_freeze_v6(
        execution_plan=plan,
        execution_plan_file_sha256="9" * 64,
        collector_repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
        comparison_output_path=tmp_path / "comparison.json",
    )
    assert frozen.case_ids == ["P34", "P35", "P36", "P37", "P38"]
    assert frozen.collector_frozen_before_p34_p38_execution is True
    assert frozen.p34_p38_outputs_observed_before_collector_freeze is False
    assert frozen.terminal_before_decomposition_retained_in_case_denominator is True
    assert frozen.result_conditioned_artifact_selection_allowed is False
