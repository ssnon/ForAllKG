from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    ProspectiveAtomicAdmissibilitySpecV5,
    build_p29_p33_spec_from_v4_infrastructure,
    build_prospective_atomic_admissibility_freeze_v5,
    default_p29_p33_tasks,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    build_p26_p28_spec_from_v3_infrastructure,
    build_prospective_decomposition_provenance_freeze_v4,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from tests.discovery.test_prospective_authority_execution_plan_v3_s170 import (
    _campaign as _v3_campaign,
)


def _v4(tmp_path):
    v3, unit = _v3_campaign(tmp_path)
    spec = build_p26_p28_spec_from_v3_infrastructure(
        infrastructure_freeze=v3,
        campaign_root=str(tmp_path / "v4"),
    )
    freeze = build_prospective_decomposition_provenance_freeze_v4(
        spec=spec,
        source_spec_sha256="1" * 64,
        infrastructure_freeze=v3,
        infrastructure_freeze_file_sha256="2" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="3" * 64,
        repository_head_sha="a" * 40,
        repository_tracked_worktree_dirty=False,
    )
    return freeze, unit


def test_default_v5_tasks_are_exactly_p29_p33() -> None:
    tasks = default_p29_p33_tasks()
    assert [row.case_id for row in tasks] == [
        "P29", "P30", "P31", "P32", "P33"
    ]
    assert [row.design_axis for row in tasks] == [
        "particle_shape",
        "alloy_composition",
        "surface_coverage",
        "excitation_power",
        "crystal_surface",
    ]
    assert len({row.relation_family for row in tasks}) == 5


def test_v5_spec_uses_v4_only_as_infrastructure_and_nonoverlap_guard(
    tmp_path,
) -> None:
    source, _ = _v4(tmp_path)
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v5"),
    )

    assert spec.prospective_goal == (
        "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10"
    )
    assert spec.prior_v4_outcome_artifacts_consumed_by_builder is False
    assert (
        spec.prior_v4_task_definitions_used_only_for_nonoverlap_guard
        is True
    )
    assert (
        spec.infrastructure_and_model_policy_only_inherited_from_v4
        is True
    )
    assert spec.expected_legacy_outcomes_predeclared is False
    assert spec.expected_neutral_outcomes_predeclared is False
    assert spec.model_policy == source.tasks[0].model_policy


def test_v5_relation_families_and_surfaces_are_disjoint_from_v4(
    tmp_path,
) -> None:
    source, _ = _v4(tmp_path)
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v5"),
    )
    old_families = {row.relation_family for row in source.tasks}
    new_families = {row.relation_family for row in spec.tasks}
    assert old_families.isdisjoint(new_families)

    old_surfaces = {
        (row.source, row.stop, row.target, row.question)
        for row in source.tasks
    }
    new_surfaces = {
        (row.source, row.stop, row.target, row.question)
        for row in spec.tasks
    }
    assert old_surfaces.isdisjoint(new_surfaces)


def test_v5_spec_rejects_wrong_case_order(tmp_path) -> None:
    source, _ = _v4(tmp_path)
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v5"),
    )
    payload = spec.model_dump(mode="json")
    payload["tasks"][0], payload["tasks"][1] = (
        payload["tasks"][1],
        payload["tasks"][0],
    )
    with pytest.raises(ValidationError, match="exactly P29-P33"):
        ProspectiveAtomicAdmissibilitySpecV5.model_validate(payload)


def test_v5_freeze_locks_all_preoutcome_boundaries(tmp_path) -> None:
    source, unit = _v4(tmp_path)
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v5"),
    )
    frozen = build_prospective_atomic_admissibility_freeze_v5(
        spec=spec,
        source_spec_sha256="4" * 64,
        infrastructure_freeze=source,
        infrastructure_freeze_file_sha256="5" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="3" * 64,
        repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
    )

    assert frozen.case_ids == ["P29", "P30", "P31", "P32", "P33"]
    assert frozen.p29_p33_initial_outputs_observed_before_freeze is False
    assert frozen.p29_p33_legacy_vpre_outputs_observed_before_freeze is False
    assert (
        frozen.p29_p33_neutral_pre_n10_outputs_observed_before_freeze
        is False
    )
    assert (
        frozen.p29_p33_source_reference_outcomes_observed_before_freeze
        is False
    )
    assert (
        frozen.p29_p33_semantic_fidelity_outcomes_observed_before_freeze
        is False
    )
    assert frozen.later_case_adaptation_allowed is False
    assert frozen.failed_or_terminal_case_replacement_allowed is False
    assert frozen.result_conditioned_route_changes_allowed is False
    assert frozen.second_regeneration_allowed is False
    assert frozen.production_selection_authority is False


def test_v5_freeze_rejects_different_regeneration_unit(tmp_path) -> None:
    source, _ = _v4(tmp_path)
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v5"),
    )
    other = build_regeneration_unit_v2_freeze(
        repository_head_sha="f" * 40,
        repository_tracked_worktree_dirty=False,
    )
    with pytest.raises(ValueError, match="regeneration-unit freeze ID mismatch"):
        build_prospective_atomic_admissibility_freeze_v5(
            spec=spec,
            source_spec_sha256="4" * 64,
            infrastructure_freeze=source,
            infrastructure_freeze_file_sha256="5" * 64,
            regeneration_unit_freeze=other,
            regeneration_unit_freeze_file_sha256="6" * 64,
            repository_head_sha="b" * 40,
            repository_tracked_worktree_dirty=False,
        )


def test_v5_freeze_requires_clean_tracked_worktree(tmp_path) -> None:
    source, unit = _v4(tmp_path)
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v5"),
    )
    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_atomic_admissibility_freeze_v5(
            spec=spec,
            source_spec_sha256="4" * 64,
            infrastructure_freeze=source,
            infrastructure_freeze_file_sha256="5" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_file_sha256="3" * 64,
            repository_head_sha="b" * 40,
            repository_tracked_worktree_dirty=True,
        )
