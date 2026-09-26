from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_v7 import (
    build_p39_p43_spec_from_v6_infrastructure,
    build_prospective_atomic_admissibility_comparison_collector_freeze_v7,
    build_prospective_atomic_admissibility_execution_plan_v7,
    build_prospective_atomic_admissibility_freeze_v7,
    default_p39_p43_tasks,
    materialize_downstream_argv_v7,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from tests.discovery.test_prospective_atomic_admissibility_v6_s176 import (
    _v6,
)


def _v7(tmp_path: Path):
    v6, unit = _v6(tmp_path)
    spec = build_p39_p43_spec_from_v6_infrastructure(
        infrastructure_freeze=v6,
        campaign_root=str(tmp_path / "v7"),
    )
    freeze = build_prospective_atomic_admissibility_freeze_v7(
        spec=spec,
        source_spec_sha256="a" * 64,
        infrastructure_freeze=v6,
        infrastructure_freeze_file_sha256="b" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="3" * 64,
        repository_head_sha="c" * 40,
        repository_worktree_dirty=False,
    )
    return freeze, unit


def _plan(tmp_path: Path):
    freeze, unit = _v7(tmp_path)
    unit_path = tmp_path / "regen.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")
    return build_prospective_atomic_admissibility_execution_plan_v7(
        campaign_freeze=freeze,
        campaign_freeze_file_sha256="d" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_path=unit_path,
        regeneration_unit_freeze_file_sha256="3" * 64,
        execution_plan_repository_head_sha="e" * 40,
        repository_worktree_dirty=False,
        settings=ProspectiveAuthorityExecutionSettingsV3(
            base_url="https://example.invalid/v1",
            api_key_env="FIXTURE_KEY",
        ),
    )


def test_v7_tasks_are_fresh_p39_p43() -> None:
    tasks = default_p39_p43_tasks()
    assert [row.case_id for row in tasks] == [
        "P39", "P40", "P41", "P42", "P43"
    ]
    assert len({row.design_axis for row in tasks}) == 5
    assert len({row.relation_family for row in tasks}) == 5


def test_v7_spec_uses_v6_only_as_infrastructure(tmp_path: Path) -> None:
    v6, _ = _v6(tmp_path)
    spec = build_p39_p43_spec_from_v6_infrastructure(
        infrastructure_freeze=v6,
        campaign_root=str(tmp_path / "v7"),
    )
    assert spec.prior_outcome_artifacts_consumed_by_builder is False
    assert spec.prior_task_definitions_used_only_for_nonoverlap_guard is True
    assert spec.infrastructure_and_model_policy_only_inherited_from_v6 is True


def test_v7_freeze_requires_full_clean_worktree(tmp_path: Path) -> None:
    v6, unit = _v6(tmp_path)
    spec = build_p39_p43_spec_from_v6_infrastructure(
        infrastructure_freeze=v6,
        campaign_root=str(tmp_path / "v7"),
    )
    try:
        build_prospective_atomic_admissibility_freeze_v7(
            spec=spec,
            source_spec_sha256="a" * 64,
            infrastructure_freeze=v6,
            infrastructure_freeze_file_sha256="b" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_file_sha256="3" * 64,
            repository_head_sha="c" * 40,
            repository_worktree_dirty=True,
        )
    except ValueError as exc:
        assert "clean worktree" in str(exc)
    else:
        raise AssertionError("dirty full worktree must fail v7 freeze")


def test_v7_execution_is_fixed_order(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert plan.case_ids == ["P39", "P40", "P41", "P42", "P43"]
    assert plan.case_order_fixed_p39_to_p43 is True
    assert plan.execution_authority_granted_by_this_plan is False
    assert plan.repository_worktree_dirty is False


def test_v7_downstream_runtime_condition_is_only_semantic_review(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    case = plan.cases[0]
    base = list(case.downstream_campaign_argv_base)
    assert materialize_downstream_argv_v7(case) == base

    review = Path(case.initial_semantic_review_path)
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text('{"fixture":true}\n', encoding="utf-8")

    observed = materialize_downstream_argv_v7(case)
    assert observed[:-2] == base
    assert observed[-2] == "--semantic-review"
    assert observed[-1] == str(review.resolve())


def test_v7_collector_freeze_uses_full_clean_contract(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    frozen = build_prospective_atomic_admissibility_comparison_collector_freeze_v7(
        execution_plan=plan,
        execution_plan_file_sha256="f" * 64,
        collector_repository_head_sha="1" * 40,
        repository_worktree_dirty=False,
        comparison_output_path=tmp_path / "comparison.json",
    )
    assert frozen.case_ids == ["P39", "P40", "P41", "P42", "P43"]
    assert frozen.collector_frozen_before_p39_p43_execution is True
    assert frozen.p39_p43_outputs_observed_before_collector_freeze is False
    assert frozen.repository_worktree_dirty is False
