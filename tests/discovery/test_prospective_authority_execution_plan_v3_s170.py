from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.prospective_authority_campaign_freeze_v3 import (
    build_p21_p25_spec_from_v2_infrastructure,
    build_prospective_authority_campaign_freeze_v3,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
    build_prospective_authority_execution_plan_v3,
    materialize_downstream_argv_v3,
)
from tests.discovery.test_prospective_authority_campaign_freeze_v3_s169 import (
    _p16_p20_freeze,
    _unit,
)


def _campaign(tmp_path: Path):
    infrastructure = _p16_p20_freeze()
    spec = build_p21_p25_spec_from_v2_infrastructure(
        infrastructure_freeze=infrastructure,
        campaign_root=str(tmp_path / "v3"),
    )
    unit = _unit()
    freeze = build_prospective_authority_campaign_freeze_v3(
        spec=spec,
        source_spec_sha256="6" * 64,
        infrastructure_freeze=infrastructure,
        infrastructure_freeze_file_sha256="7" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="8" * 64,
        repository_head_sha="e" * 40,
        repository_tracked_worktree_dirty=False,
    )
    return freeze, unit


def _plan(tmp_path: Path):
    campaign, unit = _campaign(tmp_path)
    unit_path = tmp_path / "regeneration.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")
    return build_prospective_authority_execution_plan_v3(
        campaign_freeze=campaign,
        campaign_freeze_file_sha256="9" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_path=unit_path,
        regeneration_unit_freeze_file_sha256="8" * 64,
        execution_plan_repository_head_sha="f" * 40,
        repository_tracked_worktree_dirty=False,
        settings=ProspectiveAuthorityExecutionSettingsV3(
            base_url="https://example.invalid/v1",
            api_key_env="FIXTURE_KEY",
        ),
    )


def test_v3_execution_plan_freezes_exactly_p21_p25_in_order(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    assert plan.case_ids == ["P21", "P22", "P23", "P24", "P25"]
    assert [row.case_id for row in plan.cases] == plan.case_ids
    assert plan.initial_hypothesis_outputs_observed_before_plan_freeze is False
    assert plan.semantic_outputs_observed_before_plan_freeze is False
    assert plan.later_case_adaptation_allowed is False
    assert plan.failed_or_abstained_case_replacement_allowed is False


def test_initial_argv_is_cut_before_external_novelty_and_has_no_legacy_n10_flags(
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


def test_downstream_argv_freezes_models_provider_and_regeneration_unit(
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
    assert argv[argv.index("--decomposition-model") + 1] == (
        _campaign(tmp_path)[0].tasks[0].model_policy.decomposition_model
    )
    assert argv[argv.index("--vpost-model") + 1] == (
        _campaign(tmp_path)[0].tasks[0].model_policy.vpost_model
    )


def test_semantic_review_argument_is_materialized_only_if_file_exists(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    case = plan.cases[0]
    review = Path(case.initial_semantic_review_path)
    review.parent.mkdir(parents=True, exist_ok=True)

    without = materialize_downstream_argv_v3(case)
    assert "--semantic-review" not in without

    review.write_text('{"fixture":true}\n', encoding="utf-8")
    with_review = materialize_downstream_argv_v3(case)
    assert "--semantic-review" in with_review
    assert with_review[with_review.index("--semantic-review") + 1] == str(
        review.resolve()
    )


def test_execution_plan_rejects_different_regeneration_unit(
    tmp_path: Path,
) -> None:
    campaign, _ = _campaign(tmp_path)
    # repository_head_sha participates in the freeze digest, so build a truly
    # different valid unit instead of mutating an existing frozen object.
    from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
        build_regeneration_unit_v2_freeze,
    )

    other = build_regeneration_unit_v2_freeze(
        repository_head_sha="1" * 40,
        repository_tracked_worktree_dirty=False,
    )
    unit_path = tmp_path / "other.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="freeze ID mismatch"):
        build_prospective_authority_execution_plan_v3(
            campaign_freeze=campaign,
            campaign_freeze_file_sha256="9" * 64,
            regeneration_unit_freeze=other,
            regeneration_unit_freeze_path=unit_path,
            regeneration_unit_freeze_file_sha256="8" * 64,
            execution_plan_repository_head_sha="f" * 40,
            repository_tracked_worktree_dirty=False,
        )


def test_execution_plan_requires_clean_tracked_worktree(
    tmp_path: Path,
) -> None:
    campaign, unit = _campaign(tmp_path)
    unit_path = tmp_path / "regeneration.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_authority_execution_plan_v3(
            campaign_freeze=campaign,
            campaign_freeze_file_sha256="9" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_path=unit_path,
            regeneration_unit_freeze_file_sha256="8" * 64,
            execution_plan_repository_head_sha="f" * 40,
            repository_tracked_worktree_dirty=True,
        )
