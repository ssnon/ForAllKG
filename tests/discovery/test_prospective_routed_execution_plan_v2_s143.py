from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    build_regeneration_downstream_v2_freeze,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRoutedCampaignSpec,
    ProspectiveRoutedSourceTaskDefinition,
    build_prospective_routed_campaign_freeze,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze_v2 import (
    build_p16_p20_spec_from_infrastructure_freeze,
    build_prospective_routed_campaign_freeze_v2,
)
from pipeline_core.discovery.prospective_routed_execution_plan_v2 import (
    ProspectiveRoutedCaseExecutionPlanV2,
    RoutedExecutionSettingsV2,
    build_prospective_routed_execution_plan_v2,
    default_route_protocols_v2,
)


def _old_freeze():
    tasks = [
        ProspectiveRoutedSourceTaskDefinition(
            case_id=case_id,
            relation_family=f"family_{idx}",
            source=f"source_{idx}",
            target=f"target_{idx}",
            question=f"question_{idx}",
        )
        for idx, case_id in enumerate(
            ["P11", "P12", "P13", "P14", "P15"],
            start=1,
        )
    ]
    spec = ProspectiveRoutedCampaignSpec(
        campaign_name="old",
        campaign_root="/tmp/old",
        domain_profile_id="sers_au_ag",
        corpus_id="sers500_final_v2",
        data_root="/tmp/data",
        semantic_roots=["/tmp/semantic"],
        generation_model="openai/gpt-5.6-luna",
        critic_model="openai/gpt-5.6-luna",
        tasks=tasks,
    )
    return build_prospective_routed_campaign_freeze(
        spec=spec,
        source_spec_sha256="1" * 64,
        repository_head_sha="a" * 40,
        repository_tracked_worktree_dirty=False,
    )


def _unit():
    return build_regeneration_unit_v2_freeze(
        repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
    )


def _downstream(unit):
    return build_regeneration_downstream_v2_freeze(
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="2" * 64,
        repository_head_sha="c" * 40,
        repository_tracked_worktree_dirty=False,
    )


def _campaign(unit, downstream):
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=_old_freeze(),
        campaign_root="/tmp/new",
    )
    return build_prospective_routed_campaign_freeze_v2(
        spec=spec,
        source_spec_sha256="3" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="4" * 64,
        regeneration_downstream_freeze=downstream,
        regeneration_downstream_freeze_file_sha256="5" * 64,
        repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
    )


def _plan():
    unit = _unit()
    downstream = _downstream(unit)
    campaign = _campaign(unit, downstream)
    return build_prospective_routed_execution_plan_v2(
        source_freeze=campaign,
        source_freeze_file_sha256="6" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="7" * 64,
        regeneration_downstream_freeze=downstream,
        regeneration_downstream_freeze_file_sha256="8" * 64,
        execution_plan_repository_head_sha="e" * 40,
        repository_tracked_worktree_dirty=False,
    )


def test_plan_is_exactly_p16_p20() -> None:
    plan = _plan()
    assert plan.case_ids == ["P16", "P17", "P18", "P19", "P20"]
    assert [row.case_id for row in plan.cases] == plan.case_ids


def test_regeneration_routes_use_unit_v2_not_full_e2e() -> None:
    protocols = default_route_protocols_v2()
    regen = [
        row
        for row in protocols
        if "THEN_REGENERATE_IF_UNAVAILABLE" in row.route_action
    ]
    assert len(regen) == 2
    for row in regen:
        assert row.fallback_operation == "PROSPECTIVE_REGENERATION_UNIT_V2"
        assert row.fallback_max_attempts == 1
        assert row.fallback_structured_generation_calls_per_hypothesis_max == 1
        assert row.fallback_repair_calls_per_hypothesis_max == 0
        assert row.fallback_full_e2e_argv_allowed is False
        assert (
            row.fallback_downstream_operation
            == "PROSPECTIVE_REGENERATION_DOWNSTREAM_V2"
        )


def test_case_plan_contains_context_and_no_regeneration_argv() -> None:
    plan = _plan()
    case = plan.cases[0]
    assert case.hypothesis_context_path.endswith("hypothesis.context.json")
    assert case.regeneration_argv is None
    assert case.regeneration_full_e2e_argv_present is False
    assert (
        "{final_hypothesis_id_slug}"
        in case.regeneration_unit_result_path_template
    )


def test_initial_main_e2e_is_explicitly_not_regeneration() -> None:
    plan = _plan()
    for case in plan.cases:
        assert case.initial_main_e2e_is_regeneration is False
        assert "scripts.discovery.run_dac_discovery_e2e" in case.initial_main_e2e_argv


def test_plan_references_same_frozen_unit_and_downstream() -> None:
    plan = _plan()
    assert plan.source_regeneration_unit_freeze_id.startswith(
        "prospective_regeneration_unit_v2_freeze:"
    )
    assert plan.source_regeneration_downstream_freeze_id.startswith(
        "prospective_regeneration_downstream_v2_freeze:"
    )
    assert plan.regeneration_unit_contract_frozen_before_first_case_execution
    assert plan.regeneration_downstream_contract_frozen_before_first_case_execution


def test_settings_forbid_regeneration_e2e_and_outcome_inputs() -> None:
    settings = RoutedExecutionSettingsV2()
    assert settings.regeneration_full_e2e_argv_allowed is False
    assert settings.regeneration_previous_hypothesis_text_allowed is False
    assert settings.regeneration_novelty_or_verifier_outcome_allowed is False
    assert settings.regeneration_downstream_full_e2e_rerun_allowed is False


def test_plan_is_frozen_before_any_case_outcome() -> None:
    plan = _plan()
    assert plan.gate_v2_results_observed_before_plan_freeze is False
    assert plan.repair_results_observed_before_plan_freeze is False
    assert plan.regeneration_results_observed_before_plan_freeze is False
    assert plan.endpoint_results_observed_before_plan_freeze is False
    assert plan.verifier_results_observed_before_plan_freeze is False
    assert plan.post_case_adaptation_allowed is False
    assert plan.case_replacement_allowed is False


def test_execution_plan_requires_clean_worktree() -> None:
    unit = _unit()
    downstream = _downstream(unit)
    campaign = _campaign(unit, downstream)
    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_routed_execution_plan_v2(
            source_freeze=campaign,
            source_freeze_file_sha256="6" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_file_sha256="7" * 64,
            regeneration_downstream_freeze=downstream,
            regeneration_downstream_freeze_file_sha256="8" * 64,
            execution_plan_repository_head_sha="e" * 40,
            repository_tracked_worktree_dirty=True,
        )
