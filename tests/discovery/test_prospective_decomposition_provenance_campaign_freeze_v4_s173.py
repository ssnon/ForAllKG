from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    ProspectiveDecompositionProvenanceSpecV4,
    build_p26_p28_spec_from_v3_infrastructure,
    build_prospective_decomposition_provenance_freeze_v4,
    default_p26_p28_tasks,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from tests.discovery.test_prospective_authority_execution_plan_v3_s170 import (
    _campaign,
)


def _v3(tmp_path):
    freeze, unit = _campaign(tmp_path)
    return freeze, unit


def test_default_v4_diagnostic_tasks_are_exactly_p26_p28() -> None:
    tasks = default_p26_p28_tasks()
    assert [row.case_id for row in tasks] == ["P26", "P27", "P28"]
    assert [row.design_axis for row in tasks] == [
        "gap_coupling",
        "resonance_detuning",
        "aggregation_topology",
    ]
    assert len({row.relation_family for row in tasks}) == 3


def test_v4_spec_is_explicitly_engineering_diagnostic(tmp_path) -> None:
    source, _ = _v3(tmp_path)
    spec = build_p26_p28_spec_from_v3_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v4"),
    )
    assert spec.scientific_validation_authority is False
    assert spec.diagnostic_goal_motivated_by_prior_v3_outcomes is True
    assert spec.prior_v3_outcome_artifacts_consumed_by_builder is False
    assert spec.source_tasks_predeclared_before_v4_execution is True
    assert spec.expected_scientific_outcomes_predeclared is False
    assert spec.model_policy == source.tasks[0].model_policy


def test_v4_relation_families_are_disjoint_from_v3(tmp_path) -> None:
    source, _ = _v3(tmp_path)
    spec = build_p26_p28_spec_from_v3_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v4"),
    )
    old = {row.relation_family for row in source.tasks}
    new = {row.relation_family for row in spec.tasks}
    assert old.isdisjoint(new)


def test_v4_spec_rejects_wrong_case_order(tmp_path) -> None:
    source, _ = _v3(tmp_path)
    spec = build_p26_p28_spec_from_v3_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v4"),
    )
    payload = spec.model_dump(mode="json")
    payload["tasks"][0], payload["tasks"][1] = (
        payload["tasks"][1],
        payload["tasks"][0],
    )
    with pytest.raises(ValidationError, match="exactly P26-P28"):
        ProspectiveDecompositionProvenanceSpecV4.model_validate(payload)


def test_v4_freeze_locks_observability_without_scientific_authority(
    tmp_path,
) -> None:
    source, unit = _v3(tmp_path)
    spec = build_p26_p28_spec_from_v3_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v4"),
    )
    frozen = build_prospective_decomposition_provenance_freeze_v4(
        spec=spec,
        source_spec_sha256="1" * 64,
        infrastructure_freeze=source,
        infrastructure_freeze_file_sha256="2" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="3" * 64,
        repository_head_sha="a" * 40,
        repository_tracked_worktree_dirty=False,
    )
    assert frozen.case_ids == ["P26", "P27", "P28"]
    assert frozen.engineering_diagnostic_only is True
    assert frozen.scientific_validation_authority is False
    assert (
        frozen.required_regeneration_decomposition_sidecar_filename
        == "claim_decomposition.sanitization_audit.json"
    )
    assert frozen.sidecar_required_if_regeneration_decomposition_executes is True
    assert frozen.sidecar_missing_is_diagnostic_failure is True
    assert frozen.sidecar_has_scientific_authority is False
    assert frozen.later_case_adaptation_allowed is False
    assert frozen.failed_or_terminal_case_replacement_allowed is False
    assert frozen.second_regeneration_allowed is False
    assert frozen.result_conditioned_route_changes_allowed is False


def test_v4_freeze_rejects_different_regeneration_unit(tmp_path) -> None:
    source, _ = _v3(tmp_path)
    spec = build_p26_p28_spec_from_v3_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v4"),
    )
    other = build_regeneration_unit_v2_freeze(
        repository_head_sha="f" * 40,
        repository_tracked_worktree_dirty=False,
    )
    with pytest.raises(ValueError, match="regeneration-unit freeze ID mismatch"):
        build_prospective_decomposition_provenance_freeze_v4(
            spec=spec,
            source_spec_sha256="1" * 64,
            infrastructure_freeze=source,
            infrastructure_freeze_file_sha256="2" * 64,
            regeneration_unit_freeze=other,
            regeneration_unit_freeze_file_sha256="3" * 64,
            repository_head_sha="a" * 40,
            repository_tracked_worktree_dirty=False,
        )


def test_v4_freeze_requires_clean_tracked_worktree(tmp_path) -> None:
    source, unit = _v3(tmp_path)
    spec = build_p26_p28_spec_from_v3_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v4"),
    )
    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_decomposition_provenance_freeze_v4(
            spec=spec,
            source_spec_sha256="1" * 64,
            infrastructure_freeze=source,
            infrastructure_freeze_file_sha256="2" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_file_sha256="3" * 64,
            repository_head_sha="a" * 40,
            repository_tracked_worktree_dirty=True,
        )
