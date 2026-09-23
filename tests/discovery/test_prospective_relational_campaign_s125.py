from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.prospective_relational_campaign import (
    ProspectiveCampaignStageRecord,
    ProspectiveRelationalCampaignLaunch,
    build_campaign_launch,
    build_campaign_result,
    build_case_result,
)


def _launch():
    return build_campaign_launch(
        execution_plan_id="execution:1",
        execution_plan_sha256="a" * 64,
        execution_plan_file_sha256="b" * 64,
        launcher_repository_head_sha="c" * 40,
        scientific_repository_head_sha="d" * 40,
        pinned_worktree_path="/tmp/frozen",
    )


def _case(case_id: str, disposition: str):
    return build_case_result(
        case_id=case_id,
        source_task_id="task:" + case_id,
        execution_plan_id="execution:1",
        scientific_repository_head_sha="d" * 40,
        disposition=disposition,
        stage_records=[
            ProspectiveCampaignStageRecord(
                stage_name="stage",
                argv=["python", "-m", "example"],
                return_code=0,
                log_path="/tmp/log",
            ).model_dump(mode="json")
        ],
        main_e2e_manifest_status="complete",
        selected_final_hypothesis_id=(
            "final:" + case_id
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
        selected_candidate_hypothesis_id=(
            "candidate:" + case_id
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
        selected_original_hypothesis_id=(
            "original:" + case_id
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
        endpoint_selected_claim_count=(
            1 if disposition == "VERIFIER_COMPLETE" else None
        ),
        endpoint_bound_claim_count=(
            1 if disposition == "VERIFIER_COMPLETE" else None
        ),
        endpoint_abstained_claim_count=(
            0 if disposition == "VERIFIER_COMPLETE" else None
        ),
        endpoint_novelty_bearing_bound_claim_count=(
            1 if disposition == "VERIFIER_COMPLETE" else None
        ),
        verifier_manifest_id=(
            "manifest:" + case_id
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
        certification_decision=(
            "UNRESOLVED"
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
        bounded_closure_state=(
            "BOUNDED_REVIEW_CLOSED"
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
        bounded_external_distinctness_state=(
            "INSUFFICIENT_EXTERNAL_SEARCH_EVIDENCE"
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
        positive_nonobviousness_authority_state=(
            "NOT_AUTHORIZED"
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
        fatal_blocker_state=(
            "NONE"
            if disposition == "VERIFIER_COMPLETE"
            else None
        ),
    )


def test_campaign_launch_binds_scientific_head_and_all_cases() -> None:
    launch = _launch()
    assert launch.scientific_repository_head_sha == "d" * 40
    assert launch.case_ids == ["P06", "P07", "P08", "P09", "P10"]
    assert launch.frozen_head_used_for_all_scientific_stages is True
    assert launch.post_case_adaptation_allowed is False
    assert launch.case_replacement_allowed is False


def test_case_result_preserves_failure_without_replacement() -> None:
    row = _case("P06", "MAIN_E2E_STAGE_FAILED")
    assert row.disposition == "MAIN_E2E_STAGE_FAILED"
    assert row.case_replaced is False
    assert row.settings_adapted_after_previous_case is False


def test_sparse_case_result_hash_includes_optional_defaults() -> None:
    row = build_case_result(
        case_id="P06",
        source_task_id="task:P06",
        execution_plan_id="execution:1",
        scientific_repository_head_sha="d" * 40,
        disposition="ENDPOINT_BINDING_ABSTAINED",
        stage_records=[],
        main_e2e_manifest_status="complete",
        selected_final_hypothesis_id="final:P06",
        selected_candidate_hypothesis_id="candidate:P06",
        selected_original_hypothesis_id="original:P06",
        endpoint_selected_claim_count=1,
        endpoint_bound_claim_count=0,
        endpoint_abstained_claim_count=1,
        endpoint_novelty_bearing_bound_claim_count=0,
    )
    assert row.disposition == "ENDPOINT_BINDING_ABSTAINED"
    assert row.verifier_manifest_id is None
    assert row.certification_decision is None


def test_sparse_early_failure_case_result_hash_is_valid() -> None:
    row = build_case_result(
        case_id="P07",
        source_task_id="task:P07",
        execution_plan_id="execution:1",
        scientific_repository_head_sha="d" * 40,
        disposition="MAIN_E2E_STAGE_FAILED",
        stage_records=[],
        main_e2e_manifest_status="failed",
    )
    assert row.disposition == "MAIN_E2E_STAGE_FAILED"
    assert row.selected_final_hypothesis_id is None
    assert row.endpoint_selected_claim_count is None


def test_verifier_complete_requires_complete_outcome() -> None:
    with pytest.raises(
        ValidationError,
        match="VERIFIER_COMPLETE requires complete verifier outcome",
    ):
        build_case_result(
            case_id="P06",
            source_task_id="task:P06",
            execution_plan_id="execution:1",
            scientific_repository_head_sha="d" * 40,
            disposition="VERIFIER_COMPLETE",
            stage_records=[],
        )


def test_campaign_result_accounts_for_all_five_cases() -> None:
    launch = _launch()
    rows = [
        _case("P06", "VERIFIER_COMPLETE"),
        _case("P07", "NO_BINDING_READY_HYPOTHESIS"),
        _case("P08", "ENDPOINT_BINDING_ABSTAINED"),
        _case("P09", "MAIN_E2E_STAGE_FAILED"),
        _case("P10", "VERIFIER_COMPLETE"),
    ]
    result = build_campaign_result(
        launch=launch,
        execution_plan_id="execution:1",
        scientific_repository_head_sha="d" * 40,
        case_results=rows,
    )

    assert result.all_cases_accounted_for is True
    assert result.case_replacement_performed is False
    assert result.post_case_adaptation_performed is False
    assert result.disposition_counts == {
        "ENDPOINT_BINDING_ABSTAINED": 1,
        "MAIN_E2E_STAGE_FAILED": 1,
        "NO_BINDING_READY_HYPOTHESIS": 1,
        "VERIFIER_COMPLETE": 2,
    }


def test_campaign_result_rejects_missing_or_reordered_case() -> None:
    launch = _launch()
    rows = [
        _case("P07", "NO_BINDING_READY_HYPOTHESIS"),
        _case("P06", "VERIFIER_COMPLETE"),
        _case("P08", "ENDPOINT_BINDING_ABSTAINED"),
        _case("P09", "MAIN_E2E_STAGE_FAILED"),
        _case("P10", "VERIFIER_COMPLETE"),
    ]
    with pytest.raises(ValueError, match="ordered exactly P06-P10"):
        build_campaign_result(
            launch=launch,
            execution_plan_id="execution:1",
            scientific_repository_head_sha="d" * 40,
            case_results=rows,
        )


def test_launch_hash_detects_mutation() -> None:
    launch = _launch()
    payload = launch.model_dump(mode="json")
    payload["pinned_worktree_path"] = "/tmp/mutated"
    with pytest.raises(
        ValidationError,
        match="prospective campaign launch SHA mismatch",
    ):
        ProspectiveRelationalCampaignLaunch.model_validate(payload)
