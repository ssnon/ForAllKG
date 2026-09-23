from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRoutedCampaignSpec,
    ProspectiveRoutedSourceTaskDefinition,
    build_prospective_routed_campaign_freeze,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    RoutedExecutionSettings,
    RoutedStageProtocol,
    build_prospective_routed_execution_plan,
    default_route_protocols,
)


def _source_freeze():
    task_values = [
        ("P11", "f11", "shape", "polarization"),
        ("P12", "f12", "roughness", "hotspot density"),
        ("P13", "f13", "aggregation", "reproducibility"),
        ("P14", "f14", "detuning", "enhancement"),
        ("P15", "f15", "ligand coverage", "selectivity"),
    ]
    spec = ProspectiveRoutedCampaignSpec(
        campaign_name="P11_P15",
        campaign_root="/tmp/P11_P15",
        domain_profile_id="sers_au_ag",
        corpus_id="sers500_final_v2",
        data_root="/tmp/data",
        semantic_roots=["/tmp/semantic"],
        generation_model="openai/gpt-5.6-luna",
        critic_model="openai/gpt-5.6-luna",
        tasks=[
            ProspectiveRoutedSourceTaskDefinition(
                case_id=case_id,
                relation_family=family,
                source=source,
                target=target,
                question=f"How does {source} relate to {target}?",
            )
            for case_id, family, source, target in task_values
        ],
    )
    return build_prospective_routed_campaign_freeze(
        spec=spec,
        source_spec_sha256="a" * 64,
        repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
    )


def test_route_protocols_are_canonical_and_bounded() -> None:
    protocols = default_route_protocols()
    assert [row.route_action for row in protocols] == [
        "PROCEED_TO_LITERAL_ENDPOINT_BINDING",
        "ZERO_DELTA_SPECIFICATION_REPAIR",
        "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE",
        "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE",
    ]
    assert all(row.primary_max_attempts == 1 for row in protocols)
    assert all(row.second_primary_attempt_allowed is False for row in protocols)


def test_source_alignment_route_has_one_regeneration_fallback() -> None:
    protocol = default_route_protocols()[2]
    assert protocol.primary_llm_calls_per_unit_max == 1
    assert protocol.primary_audit_calls_per_unit_max == 1
    assert protocol.fallback_operation == "FRESH_REGENERATION"
    assert protocol.fallback_max_attempts == 1
    assert protocol.fallback_llm_calls_per_hypothesis_max == 1


def test_decomposition_route_has_one_regeneration_fallback() -> None:
    protocol = default_route_protocols()[3]
    assert protocol.primary_operation == "SOURCE_SUPPORTED_ATOMIC_DECOMPOSITION"
    assert protocol.fallback_operation == "FRESH_REGENERATION"
    assert protocol.fallback_max_attempts == 1


def test_settings_require_projection_preflight_before_verifier() -> None:
    settings = RoutedExecutionSettings()
    assert settings.projection_preflight_required_before_verifier is True
    assert settings.verifier_requires_at_least_one_projected_claim is True
    assert settings.endpoint_binding_requires_post_route_gate_ready is True


def test_source_alignment_cannot_generate_new_text() -> None:
    settings = RoutedExecutionSettings()
    assert settings.source_alignment_new_text_allowed is False
    assert settings.source_alignment_generation_mode == (
        "select_existing_hypothesis_card_surfaces_only"
    )


def test_plan_is_frozen_before_any_routed_outcome() -> None:
    plan = build_prospective_routed_execution_plan(
        source_freeze=_source_freeze(),
        source_freeze_file_sha256="c" * 64,
        execution_plan_repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
    )
    assert plan.case_ids == ["P11", "P12", "P13", "P14", "P15"]
    assert plan.gate_v2_results_observed_before_plan_freeze is False
    assert plan.repair_results_observed_before_plan_freeze is False
    assert plan.regeneration_results_observed_before_plan_freeze is False
    assert plan.endpoint_results_observed_before_plan_freeze is False
    assert plan.verifier_results_observed_before_plan_freeze is False
    assert plan.post_case_adaptation_allowed is False


def test_plan_requires_clean_tracked_worktree() -> None:
    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_routed_execution_plan(
            source_freeze=_source_freeze(),
            source_freeze_file_sha256="c" * 64,
            execution_plan_repository_head_sha="d" * 40,
            repository_tracked_worktree_dirty=True,
        )


def test_proceed_protocol_cannot_have_llm_or_fallback() -> None:
    with pytest.raises(ValidationError):
        RoutedStageProtocol(
            route_action="PROCEED_TO_LITERAL_ENDPOINT_BINDING",
            primary_operation="PASSTHROUGH_NO_MUTATION",
            primary_llm_calls_per_unit_max=1,
            primary_audit_calls_per_unit_max=0,
        )
