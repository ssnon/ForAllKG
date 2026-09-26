from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_core.discovery.prospective_atomic_admissibility_campaign_freeze_v5 import (
    build_p29_p33_spec_from_v4_infrastructure,
    build_prospective_atomic_admissibility_freeze_v5,
)
from pipeline_core.discovery.prospective_atomic_admissibility_execution_plan_v5 import (
    build_prospective_atomic_admissibility_execution_plan_v5,
    materialize_downstream_argv_v5,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)
from tests.discovery.test_prospective_atomic_admissibility_campaign_freeze_v5_s174f import (
    _v4,
)


def _campaign(tmp_path: Path):
    source, unit = _v4(tmp_path)
    spec = build_p29_p33_spec_from_v4_infrastructure(
        infrastructure_freeze=source,
        campaign_root=str(tmp_path / "v5"),
    )
    freeze = build_prospective_atomic_admissibility_freeze_v5(
        spec=spec,
        source_spec_sha256="4" * 64,
        infrastructure_freeze=source,
        infrastructure_freeze_file_sha256="5" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_file_sha256="3" * 64,
        repository_head_sha="b" * 40,
        repository_tracked_worktree_dirty=False,
    )
    return freeze, unit


def _plan(tmp_path: Path):
    campaign, unit = _campaign(tmp_path)
    unit_path = tmp_path / "regeneration.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")
    return build_prospective_atomic_admissibility_execution_plan_v5(
        campaign_freeze=campaign,
        campaign_freeze_file_sha256="6" * 64,
        regeneration_unit_freeze=unit,
        regeneration_unit_freeze_path=unit_path,
        regeneration_unit_freeze_file_sha256="3" * 64,
        execution_plan_repository_head_sha="c" * 40,
        repository_tracked_worktree_dirty=False,
        settings=ProspectiveAuthorityExecutionSettingsV3(
            base_url="https://example.invalid/v1",
            api_key_env="FIXTURE_KEY",
        ),
    )


def test_v5_execution_plan_freezes_exactly_p29_p33(tmp_path) -> None:
    plan = _plan(tmp_path)

    assert plan.case_ids == ["P29", "P30", "P31", "P32", "P33"]
    assert [row.case_id for row in plan.cases] == plan.case_ids
    assert plan.prospective_goal == (
        "LEGACY_VPRE_VS_NEUTRAL_ATOMIC_PRE_N10"
    )
    assert plan.comparison_collector_contract_required_before_execution is True
    assert plan.comparison_collector_contract_frozen_by_this_plan is False
    assert plan.execution_authority_granted_by_this_plan is False


def test_v5_initial_argv_is_cut_before_external_novelty(tmp_path) -> None:
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


def test_v5_downstream_argv_freezes_models_and_regeneration_unit(
    tmp_path,
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
        Path(case.downstream_campaign_output_root).name
        == "prospective_atomic_admissibility_v5"
    )


def test_v5_semantic_review_policy_is_only_file_existence_branch(
    tmp_path,
) -> None:
    plan = _plan(tmp_path)
    case = plan.cases[0]
    review = Path(case.initial_semantic_review_path)
    review.parent.mkdir(parents=True, exist_ok=True)

    without = materialize_downstream_argv_v5(case)
    assert "--semantic-review" not in without

    review.write_text('{"fixture":true}\n', encoding="utf-8")
    with_review = materialize_downstream_argv_v5(case)
    assert "--semantic-review" in with_review
    assert with_review[with_review.index("--semantic-review") + 1] == str(
        review.resolve()
    )


def test_v5_plan_predeclares_no_outcomes(tmp_path) -> None:
    plan = _plan(tmp_path)

    assert plan.initial_hypothesis_outputs_observed_before_plan_freeze is False
    assert plan.semantic_outputs_observed_before_plan_freeze is False
    assert plan.legacy_vpre_outputs_observed_before_plan_freeze is False
    assert plan.neutral_pre_n10_outputs_observed_before_plan_freeze is False
    assert plan.source_reference_outcomes_observed_before_plan_freeze is False
    assert plan.semantic_fidelity_outcomes_observed_before_plan_freeze is False
    assert plan.comparison_outputs_observed_before_plan_freeze is False
    assert plan.external_novelty_outputs_observed_before_plan_freeze is False
    assert plan.n10_outputs_observed_before_plan_freeze is False


def test_v5_execution_plan_rejects_different_regeneration_unit(
    tmp_path,
) -> None:
    campaign, _ = _campaign(tmp_path)
    other = build_regeneration_unit_v2_freeze(
        repository_head_sha="f" * 40,
        repository_tracked_worktree_dirty=False,
    )
    unit_path = tmp_path / "other.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="freeze ID mismatch"):
        build_prospective_atomic_admissibility_execution_plan_v5(
            campaign_freeze=campaign,
            campaign_freeze_file_sha256="6" * 64,
            regeneration_unit_freeze=other,
            regeneration_unit_freeze_path=unit_path,
            regeneration_unit_freeze_file_sha256="7" * 64,
            execution_plan_repository_head_sha="c" * 40,
            repository_tracked_worktree_dirty=False,
        )


def test_v5_execution_plan_requires_clean_tracked_worktree(
    tmp_path,
) -> None:
    campaign, unit = _campaign(tmp_path)
    unit_path = tmp_path / "regeneration.freeze.json"
    unit_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean tracked worktree"):
        build_prospective_atomic_admissibility_execution_plan_v5(
            campaign_freeze=campaign,
            campaign_freeze_file_sha256="6" * 64,
            regeneration_unit_freeze=unit,
            regeneration_unit_freeze_path=unit_path,
            regeneration_unit_freeze_file_sha256="3" * 64,
            execution_plan_repository_head_sha="c" * 40,
            repository_tracked_worktree_dirty=True,
        )
