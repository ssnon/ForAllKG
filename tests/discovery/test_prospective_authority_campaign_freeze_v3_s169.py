from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_authority_campaign_freeze_v3 import (
    ProspectiveAuthorityCampaignSpecV3,
    build_p21_p25_spec_from_v2_infrastructure,
    build_prospective_authority_campaign_freeze_v3,
    default_p21_p25_tasks,
)
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


def _p11_p15_freeze():
    tasks = [
        ProspectiveRoutedSourceTaskDefinition(
            case_id=case_id,
            relation_family="old_" + case_id.lower(),
            source="source " + case_id,
            target="target " + case_id,
            question="question " + case_id,
        )
        for case_id in ["P11", "P12", "P13", "P14", "P15"]
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


def _p16_p20_freeze():
    old = _p11_p15_freeze()
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=old,
        campaign_root="/tmp/v2",
    )
    unit = _unit()
    downstream = build_regeneration_downstream_v2_freeze(
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="2" * 64,
        repository_head_sha="c" * 40,
        repository_tracked_worktree_dirty=False,
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


def test_default_v3_tasks_are_exactly_p21_p25_on_five_axes() -> None:
    tasks = default_p21_p25_tasks()
    assert [row.case_id for row in tasks] == [
        "P21", "P22", "P23", "P24", "P25"
    ]
    assert [row.design_axis for row in tasks] == [
        "ensemble_geometry",
        "gap_orientation",
        "surface_access",
        "mesoscale_morphology",
        "hybrid_interface",
    ]
    assert len({row.relation_family for row in tasks}) == 5


def test_v3_spec_inherits_only_uniform_infrastructure_and_model_roles() -> None:
    source = _p16_p20_freeze()
    spec = build_p21_p25_spec_from_v2_infrastructure(
        infrastructure_freeze=source,
        campaign_root="/tmp/v3",
    )
    assert spec.domain_profile_id == "sers_au_ag"
    assert spec.corpus_id == "sers500_final_v2"
    assert spec.model_policy.initial_generation_model == "openai/gpt-5.6-luna"
    assert spec.model_policy.initial_critic_model == "openai/gpt-5.6-luna"
    assert spec.model_policy.regeneration_model == "openai/gpt-5.6-luna"
    assert spec.tasks_defined_without_prior_scientific_outcomes is True
    assert spec.prior_v2_task_definitions_used_only_for_nonoverlap_guard is True


def test_v3_relation_families_are_disjoint_from_p16_p20() -> None:
    source = _p16_p20_freeze()
    spec = build_p21_p25_spec_from_v2_infrastructure(
        infrastructure_freeze=source,
        campaign_root="/tmp/v3",
    )
    old = {row.relation_family for row in source.tasks}
    new = {row.relation_family for row in spec.tasks}
    assert old.isdisjoint(new)


def test_v3_spec_rejects_wrong_case_or_axis_order() -> None:
    source = _p16_p20_freeze()
    spec = build_p21_p25_spec_from_v2_infrastructure(
        infrastructure_freeze=source,
        campaign_root="/tmp/v3",
    )
    payload = spec.model_dump(mode="json")
    payload["tasks"][0], payload["tasks"][1] = (
        payload["tasks"][1],
        payload["tasks"][0],
    )
    with pytest.raises(ValidationError, match="exactly P21-P25"):
        ProspectiveAuthorityCampaignSpecV3.model_validate(payload)


def test_v3_freeze_reuses_same_frozen_regeneration_unit_as_infrastructure() -> None:
    source = _p16_p20_freeze()
    spec = build_p21_p25_spec_from_v2_infrastructure(
        infrastructure_freeze=source,
        campaign_root="/tmp/v3",
    )
    unit = _unit()
    frozen = build_prospective_authority_campaign_freeze_v3(
        spec=spec,
        source_spec_sha256="6" * 64,
        infrastructure_freeze=source,
        infrastructure_freeze_file_sha256="7" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="8" * 64,
        repository_head_sha="e" * 40,
        repository_tracked_worktree_dirty=False,
    )
    assert frozen.case_ids == ["P21", "P22", "P23", "P24", "P25"]
    assert frozen.source_regeneration_unit_freeze_id == unit.freeze_id
    assert frozen.initial_e2e_cutpoint_required is True
    assert frozen.initial_external_novelty_allowed is False
    assert frozen.initial_n10_allowed is False
    assert frozen.second_regeneration_allowed is False
    assert frozen.result_conditioned_route_changes_allowed is False
    assert frozen.prior_scientific_outputs_used_to_define_tasks is False


def test_v3_freeze_rejects_different_regeneration_unit() -> None:
    source = _p16_p20_freeze()
    spec = build_p21_p25_spec_from_v2_infrastructure(
        infrastructure_freeze=source,
        campaign_root="/tmp/v3",
    )
    other = build_regeneration_unit_v2_freeze(
        repository_head_sha="f" * 40,
        repository_tracked_worktree_dirty=False,
    )
    with pytest.raises(ValueError, match="regeneration-unit freeze ID mismatch"):
        build_prospective_authority_campaign_freeze_v3(
            spec=spec,
            source_spec_sha256="6" * 64,
            infrastructure_freeze=source,
            infrastructure_freeze_file_sha256="7" * 64,
            regeneration_unit_freeze=other,
            regeneration_unit_freeze_file_sha256="8" * 64,
            repository_head_sha="e" * 40,
            repository_tracked_worktree_dirty=False,
        )


def test_v3_freeze_requires_clean_tracked_worktree() -> None:
    source = _p16_p20_freeze()
    spec = build_p21_p25_spec_from_v2_infrastructure(
        infrastructure_freeze=source,
        campaign_root="/tmp/v3",
    )
    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_authority_campaign_freeze_v3(
            spec=spec,
            source_spec_sha256="6" * 64,
            infrastructure_freeze=source,
            infrastructure_freeze_file_sha256="7" * 64,
            regeneration_unit_freeze=_unit(),
            regeneration_unit_freeze_file_sha256="8" * 64,
            repository_head_sha="e" * 40,
            repository_tracked_worktree_dirty=True,
        )
