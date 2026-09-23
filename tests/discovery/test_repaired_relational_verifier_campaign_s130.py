from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline_core.discovery.repaired_relational_verifier_campaign import (
    RepairedVerifierCaseResult,
    RepairedRelationalVerifierCampaignReport,
    build_repaired_verifier_campaign_report,
)


def _row(
    case_id: str,
    *,
    status: str,
    decision: str | None = None,
) -> RepairedVerifierCaseResult:
    complete = status == "VERIFIER_COMPLETE_AFTER_R1"
    return RepairedVerifierCaseResult(
        case_id=case_id,
        status=status,
        reentry_status=(
            "VERIFIER_READY_AFTER_R1"
            if status != "NOT_VERIFIER_READY_AFTER_R1"
            else "ENDPOINT_BINDING_ABSTAINED_AFTER_R1"
        ),
        final_hypothesis_id=(
            "final:" + case_id
            if status != "NOT_VERIFIER_READY_AFTER_R1"
            else None
        ),
        candidate_hypothesis_id=(
            "candidate:" + case_id
            if complete
            else None
        ),
        verifier_manifest_id=(
            "manifest:" + case_id
            if complete
            else None
        ),
        verifier_manifest_sha256=(
            "a" * 64
            if complete
            else None
        ),
        certification_report_id=(
            "cert:" + case_id
            if complete
            else None
        ),
        certification_decision=decision,
        bounded_closure_state=(
            "BOUNDED_REVIEW_CLOSED"
            if complete
            else None
        ),
        bounded_external_distinctness_state=(
            "INSUFFICIENT_EXTERNAL_SEARCH_EVIDENCE"
            if complete
            else None
        ),
        positive_nonobviousness_authority_state=(
            "NOT_AUTHORIZED"
            if complete
            else None
        ),
        fatal_blocker_state=(
            "NONE"
            if complete
            else None
        ),
        frozen_scientific_head_sha="b" * 40,
        verifier_output_dir=(
            "/tmp/" + case_id
            if status != "NOT_VERIFIER_READY_AFTER_R1"
            else None
        ),
        verifier_log_path=(
            "/tmp/" + case_id + ".log"
            if status != "NOT_VERIFIER_READY_AFTER_R1"
            else None
        ),
    )


def test_report_accounts_for_only_ready_verifier_attempts() -> None:
    cases = [
        _row(
            "P06",
            status="VERIFIER_COMPLETE_AFTER_R1",
            decision="UNRESOLVED",
        ),
        _row(
            "P07",
            status="VERIFIER_COMPLETE_AFTER_R1",
            decision="REJECTED",
        ),
        _row("P08", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P09", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P10", status="NOT_VERIFIER_READY_AFTER_R1"),
    ]
    report = build_repaired_verifier_campaign_report(
        source_reentry_report_id="reentry:1",
        source_reentry_report_sha256="c" * 64,
        source_reentry_report_file_sha256="d" * 64,
        source_execution_plan_id="execution:1",
        source_campaign_launch_id="launch:1",
        frozen_scientific_head_sha="b" * 40,
        cases=cases,
    )
    assert report.verifier_attempted_case_ids == ["P06", "P07"]
    assert report.verifier_complete_case_ids == ["P06", "P07"]
    assert report.certification_decision_counts == {
        "REJECTED": 1,
        "UNRESOLVED": 1,
    }


def test_failed_ready_case_is_attempted_but_not_complete() -> None:
    cases = [
        _row("P06", status="VERIFIER_STAGE_FAILED_AFTER_R1"),
        _row("P07", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P08", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P09", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P10", status="NOT_VERIFIER_READY_AFTER_R1"),
    ]
    report = build_repaired_verifier_campaign_report(
        source_reentry_report_id="reentry:1",
        source_reentry_report_sha256="c" * 64,
        source_reentry_report_file_sha256="d" * 64,
        source_execution_plan_id="execution:1",
        source_campaign_launch_id="launch:1",
        frozen_scientific_head_sha="b" * 40,
        cases=cases,
    )
    assert report.verifier_attempted_case_ids == ["P06"]
    assert report.verifier_complete_case_ids == []


def test_complete_case_requires_full_c_d_n_x_outcome() -> None:
    with pytest.raises(
        ValidationError,
        match="requires full outcome",
    ):
        RepairedVerifierCaseResult(
            case_id="P06",
            status="VERIFIER_COMPLETE_AFTER_R1",
            reentry_status="VERIFIER_READY_AFTER_R1",
            final_hypothesis_id="final:P06",
            candidate_hypothesis_id="candidate:P06",
            verifier_manifest_id="manifest:P06",
            verifier_manifest_sha256="a" * 64,
            certification_report_id="cert:P06",
            certification_decision="UNRESOLVED",
            frozen_scientific_head_sha="b" * 40,
        )


def test_report_rejects_reordered_cases() -> None:
    cases = [
        _row("P07", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P06", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P08", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P09", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P10", status="NOT_VERIFIER_READY_AFTER_R1"),
    ]
    with pytest.raises(ValueError, match="ordered exactly P06-P10"):
        build_repaired_verifier_campaign_report(
            source_reentry_report_id="reentry:1",
            source_reentry_report_sha256="c" * 64,
            source_reentry_report_file_sha256="d" * 64,
            source_execution_plan_id="execution:1",
            source_campaign_launch_id="launch:1",
            frozen_scientific_head_sha="b" * 40,
            cases=cases,
        )


def test_report_hash_detects_mutation() -> None:
    cases = [
        _row("P06", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P07", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P08", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P09", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P10", status="NOT_VERIFIER_READY_AFTER_R1"),
    ]
    report = build_repaired_verifier_campaign_report(
        source_reentry_report_id="reentry:1",
        source_reentry_report_sha256="c" * 64,
        source_reentry_report_file_sha256="d" * 64,
        source_execution_plan_id="execution:1",
        source_campaign_launch_id="launch:1",
        frozen_scientific_head_sha="b" * 40,
        cases=cases,
    )
    payload = report.model_dump(mode="json")
    payload["frozen_scientific_head_sha"] = "e" * 40
    with pytest.raises(
        ValidationError,
        match="repaired verifier campaign SHA mismatch",
    ):
        RepairedRelationalVerifierCampaignReport.model_validate(payload)


def test_nonready_case_cannot_accidentally_count_as_attempted() -> None:
    cases = [
        _row("P06", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P07", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P08", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P09", status="NOT_VERIFIER_READY_AFTER_R1"),
        _row("P10", status="NOT_VERIFIER_READY_AFTER_R1"),
    ]
    report = build_repaired_verifier_campaign_report(
        source_reentry_report_id="reentry:1",
        source_reentry_report_sha256="c" * 64,
        source_reentry_report_file_sha256="d" * 64,
        source_execution_plan_id="execution:1",
        source_campaign_launch_id="launch:1",
        frozen_scientific_head_sha="b" * 40,
        cases=cases,
    )
    assert report.verifier_attempted_case_ids == []
    assert report.certification_decision_counts == {}
