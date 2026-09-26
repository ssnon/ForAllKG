from __future__ import annotations

from pathlib import Path

from pipeline_core.discovery.prospective_atomic_admissibility_comparison_collector_v5 import (
    build_prospective_atomic_admissibility_comparison_collector_freeze_v5,
    classify_pair,
    classify_reason_transition,
)
from pipeline_core.discovery.prospective_atomic_admissibility_execution_plan_v5 import (
    build_prospective_atomic_admissibility_execution_plan_v5,
)
from pipeline_core.discovery.prospective_authority_execution_plan_v3 import (
    ProspectiveAuthorityExecutionSettingsV3,
)
from tests.discovery.test_prospective_atomic_admissibility_execution_plan_v5_s174f import (
    _campaign,
)


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


def _freeze(tmp_path: Path):
    plan = _plan(tmp_path)
    return build_prospective_atomic_admissibility_comparison_collector_freeze_v5(
        execution_plan=plan,
        execution_plan_file_sha256="7" * 64,
        collector_repository_head_sha="d" * 40,
        repository_tracked_worktree_dirty=False,
        comparison_output_path=tmp_path / "comparison.report.json",
    )


def test_collector_freeze_fixes_p29_p33_case_denominator(tmp_path) -> None:
    frozen = _freeze(tmp_path)

    assert frozen.case_ids == ["P29", "P30", "P31", "P32", "P33"]
    assert [row.case_id for row in frozen.cases] == frozen.case_ids
    assert frozen.case_count == 5
    assert frozen.case_denominator_fixed_before_execution is True
    assert (
        frozen.terminal_before_decomposition_retained_in_case_denominator
        is True
    )
    assert frozen.terminal_case_claim_rows_synthesized is False


def test_collector_freeze_fixes_initial_and_reentry_artifact_rules(
    tmp_path,
) -> None:
    frozen = _freeze(tmp_path)
    case = frozen.cases[0]
    campaign = Path(case.campaign_root)

    assert Path(case.campaign_report_path) == campaign / "campaign.report.json"
    assert Path(case.initial_vpre_query_path) == (
        campaign / "01_initial_vpre" / "claims_queries.json"
    )
    assert Path(case.initial_vpre_contract_path) == (
        campaign / "01_initial_vpre" / "contract.report.json"
    )
    assert Path(case.initial_vpre_audit_path) == (
        campaign
        / "01_initial_vpre"
        / "claim_decomposition.sanitization_audit.json"
    )
    assert Path(case.regeneration_reentry_report_path) == (
        campaign / "04_regeneration_reentry" / "reentry_v2.report.json"
    )
    assert Path(case.regeneration_lineage_root) == (
        campaign / "04_regeneration_reentry" / "lineage"
    )


def test_collector_freeze_disallows_result_conditioned_selection(
    tmp_path,
) -> None:
    frozen = _freeze(tmp_path)

    assert frozen.audit_record_subset_selection_allowed is False
    assert frozen.result_conditioned_artifact_selection_allowed is False
    assert frozen.unreported_extra_regeneration_audit_is_failure is True
    assert frozen.missing_required_initial_artifact_is_failure is True
    assert frozen.missing_required_reentry_artifact_is_failure is True
    assert frozen.llm_calls_allowed is False
    assert frozen.artifact_mutation_allowed is False


def test_collector_freeze_is_execution_precondition_not_authority(
    tmp_path,
) -> None:
    frozen = _freeze(tmp_path)

    assert frozen.collector_frozen_before_p29_p33_execution is True
    assert frozen.p29_p33_outputs_observed_before_collector_freeze is False
    assert frozen.campaign_execution_may_begin_after_this_freeze is True
    assert frozen.engineering_diagnostic_only is True
    assert frozen.scientific_validation_authority is False
    assert frozen.production_selection_authority is False


def test_four_way_pair_taxonomy_is_total() -> None:
    assert classify_pair(
        legacy_ready=True,
        neutral_ready=True,
    ) == "LEGACY_READY__NEUTRAL_READY"
    assert classify_pair(
        legacy_ready=True,
        neutral_ready=False,
    ) == "LEGACY_READY__NEUTRAL_NOT_READY"
    assert classify_pair(
        legacy_ready=False,
        neutral_ready=True,
    ) == "LEGACY_NOT_READY__NEUTRAL_READY"
    assert classify_pair(
        legacy_ready=False,
        neutral_ready=False,
    ) == "LEGACY_NOT_READY__NEUTRAL_NOT_READY"


def test_reason_transition_distinguishes_representation_from_semantics() -> None:
    exact = ["prediction_exact_source_binding_cardinality:0"]

    assert classify_reason_transition(
        legacy_source_reasons=exact,
        neutral_source_reference_status="READY",
        neutral_semantic_fidelity_status="PASS",
        neutral_blocking_dimensions=[],
        neutral_ready=True,
    ) == "REPRESENTATION_LOSS_ONLY_CANDIDATE"

    assert classify_reason_transition(
        legacy_source_reasons=exact,
        neutral_source_reference_status="READY",
        neutral_semantic_fidelity_status="INVALID",
        neutral_blocking_dimensions=["SEMANTIC_FIDELITY"],
        neutral_ready=False,
    ) == "EXACT_TEXT_FAILURE_RECLASSIFIED_WITH_SEMANTIC_BLOCKER"


def test_collector_claim_population_is_all_audited_claims(tmp_path) -> None:
    frozen = _freeze(tmp_path)

    assert frozen.claim_population_policy == (
        "ALL_AUDITED_CLAIMS_FROM_INITIAL_AND_REENTRY"
    )
    assert frozen.frozen_legacy_contract_is_legacy_authority is True
    assert frozen.neutral_replay_uses_frozen_semantic_binding is True
    assert frozen.semantic_fidelity_replay_must_match_frozen_audit is True
