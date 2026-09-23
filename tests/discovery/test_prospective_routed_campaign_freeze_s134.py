from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRepairRegenerationRouterPolicy,
    ProspectiveRoutedCampaignSpec,
    ProspectiveRoutedSourceTaskDefinition,
    build_prospective_routed_campaign_freeze,
)


def _tasks() -> list[ProspectiveRoutedSourceTaskDefinition]:
    values = [
        (
            "P11",
            "shape_polarization_coupling",
            "particle shape anisotropy",
            "polarization dependence",
        ),
        (
            "P12",
            "roughness_hotspot_coupling",
            "nanoscale surface roughness",
            "hotspot density",
        ),
        (
            "P13",
            "aggregation_reproducibility",
            "particle aggregation state",
            "spectral reproducibility",
        ),
        (
            "P14",
            "excitation_detuning_response",
            "excitation wavelength detuning",
            "SERS enhancement intensity",
        ),
        (
            "P15",
            "ligand_adsorption_selectivity",
            "surface ligand coverage",
            "analyte adsorption selectivity",
        ),
    ]
    return [
        ProspectiveRoutedSourceTaskDefinition(
            case_id=case_id,
            relation_family=family,
            source=source,
            target=target,
            question=f"How does {source} relate to {target} in SERS?",
        )
        for case_id, family, source, target in values
    ]


def _spec() -> ProspectiveRoutedCampaignSpec:
    return ProspectiveRoutedCampaignSpec(
        campaign_name="P11_P15",
        campaign_root="/tmp/P11_P15",
        domain_profile_id="sers_au_ag",
        corpus_id="sers500_final_v2",
        data_root="/tmp/data",
        semantic_roots=["/tmp/semantic"],
        generation_model="openai/gpt-5.6-luna",
        critic_model="openai/gpt-5.6-luna",
        tasks=_tasks(),
    )


def test_router_policy_is_bounded_and_nonadaptive() -> None:
    policy = ProspectiveRepairRegenerationRouterPolicy()
    assert policy.max_specification_repair_attempts_per_claim == 1
    assert policy.max_source_alignment_attempts_per_claim == 1
    assert policy.max_atomic_decomposition_attempts_per_hypothesis == 1
    assert policy.max_regeneration_attempts_per_hypothesis == 1
    assert policy.second_repair_after_failed_reentry_allowed is False
    assert policy.later_case_settings_may_adapt_to_earlier_case_outcomes is False


def test_router_mapping_is_explicit() -> None:
    policy = ProspectiveRepairRegenerationRouterPolicy()
    assert policy.route_for_hint(
        "PROCEED_TO_LITERAL_ENDPOINT_BINDING"
    ) == "PROCEED_TO_LITERAL_ENDPOINT_BINDING"
    assert policy.route_for_hint(
        "SPECIFICATION_REPAIR_REVIEW"
    ) == "ZERO_DELTA_SPECIFICATION_REPAIR"
    assert policy.route_for_hint(
        "SOURCE_CONTRACT_ALIGNMENT_OR_REGENERATE_REVIEW"
    ) == "ZERO_DELTA_SOURCE_ALIGNMENT_THEN_REGENERATE_IF_UNAVAILABLE"
    assert policy.route_for_hint(
        "DECOMPOSE_OR_REGENERATE_REVIEW"
    ) == "ATOMIC_DECOMPOSITION_THEN_REGENERATE_IF_UNAVAILABLE"


def test_router_rejects_unknown_hint() -> None:
    policy = ProspectiveRepairRegenerationRouterPolicy()
    with pytest.raises(ValueError, match="unsupported gate-v2"):
        policy.route_for_hint("UNKNOWN")


def test_source_alignment_is_zero_delta_and_surface_bounded() -> None:
    policy = ProspectiveRepairRegenerationRouterPolicy()
    assert policy.source_alignment_may_only_reuse_existing_hypothesis_card_surfaces
    assert policy.source_alignment_requires_zero_scientific_delta
    assert policy.specification_repair_requires_zero_scientific_delta


def test_campaign_requires_exact_fresh_p11_p15_namespace() -> None:
    tasks = _tasks()
    payload = _spec().model_dump(mode="json")
    payload["tasks"] = [
        row.model_dump(mode="json")
        for row in reversed(tasks)
    ]
    with pytest.raises(ValidationError, match="exactly P11-P15"):
        ProspectiveRoutedCampaignSpec.model_validate(payload)


def test_freeze_preserves_no_outcome_observation_invariants() -> None:
    frozen = build_prospective_routed_campaign_freeze(
        spec=_spec(),
        source_spec_sha256="a" * 64,
        repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
    )
    assert frozen.case_ids == ["P11", "P12", "P13", "P14", "P15"]
    assert frozen.router_policy_frozen_before_generation is True
    assert frozen.gate_v2_outcomes_observed_before_freeze is False
    assert frozen.repair_outputs_observed_before_freeze is False
    assert frozen.regeneration_outputs_observed_before_freeze is False
    assert frozen.failed_or_abstained_case_replacement_allowed is False


def test_freeze_requires_clean_tracked_worktree() -> None:
    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_routed_campaign_freeze(
            spec=_spec(),
            source_spec_sha256="a" * 64,
            repository_head_sha="b" * 40,
            repository_tracked_worktree_dirty=True,
        )
