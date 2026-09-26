from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from pipeline_core.discovery.prospective_decomposition_provenance_campaign_freeze_v4 import (
    build_p26_p28_spec_from_v3_infrastructure,
    build_prospective_decomposition_provenance_freeze_v4,
)
from pipeline_core.discovery.prospective_decomposition_provenance_execution_plan_v4 import (
    build_prospective_decomposition_provenance_execution_plan_v4,
    materialize_downstream_argv_v4,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from tests.discovery.test_prospective_authority_execution_plan_v3_s170 import (
    _campaign as _v3_campaign,
)


def _campaign(tmp_path: Path):
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


def _plan(tmp_path: Path):
    campaign, unit = _campaign(tmp_path)
    unit_path = tmp_path / "regeneration.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")
    return build_prospective_decomposition_provenance_execution_plan_v4(
        campaign_freeze=campaign,
        campaign_freeze_file_sha256="4" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_path=unit_path,
        regeneration_unit_freeze_file_sha256="3" * 64,
        execution_plan_repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
        settings=ProspectiveAuthorityExecutionSettingsV3(
            base_url="https://example.invalid/v1",
            api_key_env="FIXTURE_KEY",
        ),
    )


def test_v4_execution_plan_freezes_exactly_p26_p28_in_order(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    assert plan.case_ids == ["P26", "P27", "P28"]
    assert [row.case_id for row in plan.cases] == plan.case_ids
    assert plan.engineering_diagnostic_only is True
    assert plan.scientific_validation_authority is False
    assert plan.source_id_coverage_observed_before_plan_freeze is False
    assert plan.exact_source_rebind_recovery_observed_before_plan_freeze is False
    assert plan.later_case_adaptation_allowed is False
    assert plan.failed_or_terminal_case_replacement_allowed is False


def test_v4_initial_argv_is_cut_before_external_novelty(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    for case in plan.cases:
        argv = case.initial_e2e_argv
        assert argv[:3] == [
            "python",
            "-m",
            "scripts.discovery.run_dac_discovery_e2e",
        ]
        assert "--stop-after-initial-semantic" in argv
        assert "--nonobviousness-original-fallback-enforce" not in argv
        assert "--nonobviousness-post-generation-enforce" not in argv
        assert "--overwrite-run" not in argv
        assert argv[argv.index("--providers") + 1] == "auto"
        assert argv[argv.index("--results-per-query") + 1] == "12"


def test_v4_downstream_argv_freezes_models_and_regeneration_unit(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    case = plan.cases[0]
    argv = case.downstream_campaign_argv_base

    assert argv[:3] == [
        "python",
        "-m",
        "scripts.discovery.run_pre_n10_prospective_campaign_v1",
    ]
    assert "--semantic-review" not in argv
    assert "--save-prompts" in argv
    assert "--allow-dirty-worktree" not in argv
    assert argv[argv.index("--api-key-env") + 1] == "FIXTURE_KEY"
    assert argv[argv.index("--base-url") + 1] == "https://example.invalid/v1"
    assert (
        argv[argv.index("--regeneration-unit-freeze") + 1]
        == str((tmp_path / "regeneration.freeze.json").resolve())
    )
    assert (
        Path(case.downstream_campaign_output_root).name
        == "prospective_decomposition_provenance_v4"
    )


def test_v4_plan_freezes_regeneration_sidecar_discovery_contract(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    for case in plan.cases:
        reentry = Path(case.regeneration_reentry_root)
        lineage = Path(case.regeneration_lineage_root)
        downstream = Path(case.downstream_campaign_output_root)

        assert reentry == downstream / "04_regeneration_reentry"
        assert lineage == reentry / "lineage"
        assert case.regeneration_decomposition_sidecar_filename == (
            "claim_decomposition.sanitization_audit.json"
        )
        assert case.regeneration_decomposition_sidecar_relative_glob == (
            "lineage/*/claim_decomposition.sanitization_audit.json"
        )
        assert case.regeneration_decomposition_sidecar_schema == (
            "pre-n10-regeneration-decomposition-sanitization-audit-v1"
        )
        assert (
            case.sidecar_required_if_regeneration_decomposition_executes
            is True
        )
        assert case.sidecar_missing_is_diagnostic_failure is True
        assert case.sidecar_has_scientific_authority is False


def test_v4_semantic_review_argument_is_materialized_only_if_present(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    case = plan.cases[0]
    review = Path(case.initial_semantic_review_path)
    review.parent.mkdir(parents=True, exist_ok=True)

    without = materialize_downstream_argv_v4(case)
    assert "--semantic-review" not in without

    review.write_text('{"fixture":true}\n', encoding="utf-8")
    with_review = materialize_downstream_argv_v4(case)
    assert "--semantic-review" in with_review
    assert with_review[with_review.index("--semantic-review") + 1] == str(
        review.resolve()
    )


def test_v4_execution_plan_rejects_different_regeneration_unit(
    tmp_path: Path,
) -> None:
    campaign, _ = _campaign(tmp_path)
    other = build_regeneration_unit_v2_freeze(
        repository_head_sha="f" * 40,
        repository_tracked_worktree_dirty=False,
    )
    unit_path = tmp_path / "other.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="freeze ID mismatch"):
        build_prospective_decomposition_provenance_execution_plan_v4(
            campaign_freeze=campaign,
            campaign_freeze_file_sha256="4" * 64,
            regeneration_unit_freeze=other,
            regeneration_unit_freeze_path=unit_path,
            regeneration_unit_freeze_file_sha256="3" * 64,
            execution_plan_repository_head_sha="b" * 40,
            repository_tracked_worktree_dirty=False,
        )


def test_v4_execution_plan_requires_clean_tracked_worktree(
    tmp_path: Path,
) -> None:
    campaign, unit = _campaign(tmp_path)
    unit_path = tmp_path / "regeneration.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_decomposition_provenance_execution_plan_v4(
            campaign_freeze=campaign,
            campaign_freeze_file_sha256="4" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_path=unit_path,
            regeneration_unit_freeze_file_sha256="3" * 64,
            execution_plan_repository_head_sha="b" * 40,
            repository_tracked_worktree_dirty=True,
        )
