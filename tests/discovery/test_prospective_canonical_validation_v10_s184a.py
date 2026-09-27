from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from pipeline_core.discovery.prospective_canonical_validation_v10 import (
    build_p54_p58_spec_from_v9_infrastructure,
    build_prospective_canonical_validation_collector_freeze_v10,
    build_prospective_canonical_validation_execution_plan_v10,
    ProspectiveCanonicalValidationCaseResultV10,
    build_prospective_canonical_validation_freeze_v10,
    default_p54_p58_tasks,
    materialize_downstream_argv_v10,
)
from tests.discovery.test_prospective_canonical_validation_v9_s183a import (
    _freeze as _v9_freeze,
)


def _freeze(tmp_path: Path):
    v8, unit = _v9_freeze(tmp_path)
    spec = build_p54_p58_spec_from_v9_infrastructure(
        infrastructure_freeze=v8,
        campaign_root=str(tmp_path / "v10"),
    )
    frozen = build_prospective_canonical_validation_freeze_v10(
        spec=spec,
        source_spec_sha256="a" * 64,
        infrastructure_freeze=v8,
        infrastructure_freeze_file_sha256="b" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="3" * 64,
        repository_head_sha="c" * 40,
        repository_worktree_dirty=False,
    )
    return frozen, unit


def _plan(tmp_path: Path):
    frozen, unit = _freeze(tmp_path)
    unit_path = tmp_path / "regen.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")
    return build_prospective_canonical_validation_execution_plan_v10(
        campaign_freeze=frozen,
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


def test_v10_tasks_are_fresh_p54_p58() -> None:
    tasks = default_p54_p58_tasks()
    assert [row.case_id for row in tasks] == [
        "P54", "P55", "P56", "P57", "P58"
    ]
    assert len({row.design_axis for row in tasks}) == 5
    assert len({row.relation_family for row in tasks}) == 5


def test_v10_spec_uses_v9_freeze_as_infrastructure_only(
    tmp_path: Path,
) -> None:
    v8, _ = _v9_freeze(tmp_path)
    spec = build_p54_p58_spec_from_v9_infrastructure(
        infrastructure_freeze=v8,
        campaign_root=str(tmp_path / "v10"),
    )
    assert spec.prospective_goal == (
        "CANONICAL_STABLE_ID_END_TO_END_INTEGRITY"
    )
    assert spec.prior_outcome_artifacts_consumed_by_builder is False
    assert spec.prior_v9_execution_outputs_consumed is False
    assert spec.prior_task_definitions_used_only_for_nonoverlap_guard is True
    assert spec.v9_infrastructure_and_model_policy_only is True


def test_v10_freeze_requires_full_clean_worktree(tmp_path: Path) -> None:
    v8, unit = _v9_freeze(tmp_path)
    spec = build_p54_p58_spec_from_v9_infrastructure(
        infrastructure_freeze=v8,
        campaign_root=str(tmp_path / "v10"),
    )
    with pytest.raises(ValueError, match="clean worktree"):
        build_prospective_canonical_validation_freeze_v10(
            spec=spec,
            source_spec_sha256="a" * 64,
            infrastructure_freeze=v8,
            infrastructure_freeze_file_sha256="b" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_file_sha256="3" * 64,
            repository_head_sha="c" * 40,
            repository_worktree_dirty=True,
        )


def test_v10_execution_is_fixed_order_and_uses_strict_campaign(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    assert plan.case_ids == ["P54", "P55", "P56", "P57", "P58"]
    assert plan.case_order_fixed_p54_to_p58 is True
    assert plan.execution_authority_granted_by_this_plan is False
    for case in plan.cases:
        assert "--stop-after-initial-semantic" in case.initial_e2e_argv
        assert case.downstream_campaign_argv_base[:3] == [
            "python",
            "-m",
            "scripts.discovery.run_pre_n10_prospective_campaign_v1",
        ]
        assert case.downstream_campaign_output_root.endswith(
            "prospective_canonical_validation_v10"
        )


def test_v10_downstream_runtime_condition_is_only_semantic_review(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    case = plan.cases[0]
    base = list(case.downstream_campaign_argv_base)
    assert materialize_downstream_argv_v10(case) == base

    review = Path(case.initial_semantic_review_path)
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text('{"fixture":true}\n', encoding="utf-8")

    observed = materialize_downstream_argv_v10(case)
    assert observed[:-2] == base
    assert observed[-2] == "--semantic-review"
    assert observed[-1] == str(review.resolve())


def test_v10_collector_freezes_canonical_artifact_paths_before_execution(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    frozen = build_prospective_canonical_validation_collector_freeze_v10(
        execution_plan=plan,
        execution_plan_file_sha256="f" * 64,
        collector_repository_head_sha="1" * 40,
        repository_worktree_dirty=False,
        validation_output_path=tmp_path / "canonical.validation.json",
    )
    assert frozen.case_ids == ["P54", "P55", "P56", "P57", "P58"]
    assert frozen.collector_frozen_before_p54_p58_execution is True
    assert frozen.p54_p58_outputs_observed_before_collector_freeze is False
    first = frozen.cases[0]
    assert first.initial_contract_v2_path.endswith(
        "01_initial_vpre/contract_v2.report.json"
    )
    assert first.post_primary_contract_v2_path.endswith(
        "02_primary_router/contract_v2.after_primary_router.json"
    )
    assert first.relational_binding_bridge_path.endswith(
        "07_relational_binding_bridge/relational_binding_bridge.report.json"
    )
    assert first.vpost_plan_path.endswith(
        "08_vpost_shadow/vpost_execution.plan.json"
    )


def test_v10_result_tracks_factor_projection_guard_metrics() -> None:
    row = ProspectiveCanonicalValidationCaseResultV10(
        case_id="P54",
        campaign_final_status="fixture",
        accounting_status="TERMINAL_BEFORE_INITIAL_VPRE",
        initial_v2_verified=False,
        primary_v2_verified=False,
        regenerated_v2_lineage_count=0,
        handoff_lineage_count=0,
        handoff_stable_id_lineage_count=0,
        handoff_legacy_authority_lineage_count=0,
        bridge_reached=False,
        vpost_reached=False,
        vpost_legacy_flag_count=0,
        vpost_completed_count=0,
        fresh_legacy_authority_violation=False,
    )
    assert row.bridge_binding_ready_lineage_count == 0
    assert row.bridge_not_binding_ready_lineage_count == 0
    assert row.bridge_binding_reason_counts == {}
    assert row.canonical_identity_endpoint_overlap_claim_count == 0
    assert row.vpost_execution_required_lineage_count == 0
    assert row.vpost_skipped_not_binding_ready_lineage_count == 0
