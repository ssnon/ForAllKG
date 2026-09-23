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
    ProspectiveRoutedCampaignSpecV2,
    build_p16_p20_spec_from_infrastructure_freeze,
    build_prospective_routed_campaign_freeze_v2,
    default_p16_p20_tasks,
)


def _old_freeze():
    tasks = []
    for index, case_id in enumerate(
        ["P11", "P12", "P13", "P14", "P15"],
        start=1,
    ):
        tasks.append(
            ProspectiveRoutedSourceTaskDefinition(
                case_id=case_id,
                relation_family=f"family_{index}",
                source=f"source_{index}",
                target=f"target_{index}",
                question=f"question_{index}",
            )
        )
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


def _unit_freeze():
    return build_regeneration_unit_v2_freeze(
        repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
    )


def _downstream_freeze(unit):
    return build_regeneration_downstream_v2_freeze(
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="2" * 64,
        repository_head_sha="c" * 40,
        repository_tracked_worktree_dirty=False,
    )


def test_default_tasks_are_exactly_p16_p20_and_distinct() -> None:
    tasks = default_p16_p20_tasks()
    assert [row.case_id for row in tasks] == [
        "P16", "P17", "P18", "P19", "P20"
    ]
    assert len({row.relation_family for row in tasks}) == 5
    assert len({(row.source, row.target) for row in tasks}) == 5


def test_spec_inherits_only_uniform_infrastructure() -> None:
    old = _old_freeze()
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=old,
        campaign_root="/tmp/new",
    )
    assert spec.domain_profile_id == "sers_au_ag"
    assert spec.corpus_id == "sers500_final_v2"
    assert spec.generation_model == "openai/gpt-5.6-luna"
    assert spec.tasks_defined_without_prior_scientific_outcomes is True
    assert spec.infrastructure_only_inherited_from_prior_campaign is True


def test_spec_rejects_wrong_case_order() -> None:
    old = _old_freeze()
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=old,
        campaign_root="/tmp/new",
    )
    payload = spec.model_dump(mode="json")
    payload["tasks"] = list(reversed(payload["tasks"]))
    with pytest.raises(ValidationError, match="exactly P16-P20"):
        ProspectiveRoutedCampaignSpecV2.model_validate(payload)


def test_freeze_references_both_pre_frozen_regeneration_contracts() -> None:
    old = _old_freeze()
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=old,
        campaign_root="/tmp/new",
    )
    unit = _unit_freeze()
    downstream = _downstream_freeze(unit)
    frozen = build_prospective_routed_campaign_freeze_v2(
        spec=spec,
        source_spec_sha256="3" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="4" * 64,
        regeneration_downstream_freeze=downstream,
        regeneration_downstream_freeze_file_sha256="5" * 64,
        repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
    )
    assert frozen.case_ids == ["P16", "P17", "P18", "P19", "P20"]
    assert frozen.source_regeneration_unit_freeze_id == unit.freeze_id
    assert (
        frozen.source_regeneration_downstream_freeze_id
        == downstream.freeze_id
    )
    assert frozen.regeneration_unit_v2_frozen_before_source_tasks is True
    assert frozen.regeneration_downstream_v2_frozen_before_source_tasks is True


def test_freeze_forbids_outcome_informed_task_definition() -> None:
    old = _old_freeze()
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=old,
        campaign_root="/tmp/new",
    )
    unit = _unit_freeze()
    downstream = _downstream_freeze(unit)
    frozen = build_prospective_routed_campaign_freeze_v2(
        spec=spec,
        source_spec_sha256="3" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="4" * 64,
        regeneration_downstream_freeze=downstream,
        regeneration_downstream_freeze_file_sha256="5" * 64,
        repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
    )
    assert frozen.prior_scientific_outputs_used_to_define_tasks is False
    assert frozen.prior_campaign_used_for_infrastructure_only is True
    assert frozen.later_case_adaptation_allowed is False
    assert frozen.failed_or_abstained_case_replacement_allowed is False


def test_downstream_must_reference_same_regeneration_unit() -> None:
    old = _old_freeze()
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=old,
        campaign_root="/tmp/new",
    )
    unit = _unit_freeze()
    other_unit = build_regeneration_unit_v2_freeze(
        repository_head_sha="e" * 40,
        repository_tracked_worktree_dirty=False,
    )
    downstream = _downstream_freeze(other_unit)
    with pytest.raises(ValueError, match="freeze ID mismatch"):
        build_prospective_routed_campaign_freeze_v2(
            spec=spec,
            source_spec_sha256="3" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_file_sha256="4" * 64,
            regeneration_downstream_freeze=downstream,
            regeneration_downstream_freeze_file_sha256="5" * 64,
            repository_head_sha="d" * 40,
            repository_tracked_worktree_dirty=False,
        )


def test_freeze_requires_clean_tracked_worktree() -> None:
    old = _old_freeze()
    spec = build_p16_p20_spec_from_infrastructure_freeze(
        infrastructure_freeze=old,
        campaign_root="/tmp/new",
    )
    unit = _unit_freeze()
    downstream = _downstream_freeze(unit)
    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_routed_campaign_freeze_v2(
            spec=spec,
            source_spec_sha256="3" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_file_sha256="4" * 64,
            regeneration_downstream_freeze=downstream,
            regeneration_downstream_freeze_file_sha256="5" * 64,
            repository_head_sha="d" * 40,
            repository_tracked_worktree_dirty=True,
        )
