from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


VerifierCaseStatus = Literal[
    "NOT_VERIFIER_READY_AFTER_R1",
    "VERIFIER_STAGE_FAILED_AFTER_R1",
    "VERIFIER_COMPLETE_AFTER_R1",
]


class RepairedVerifierCaseResult(StrictModel):
    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    status: VerifierCaseStatus
    reentry_status: str
    final_hypothesis_id: str | None = None
    candidate_hypothesis_id: str | None = None

    verifier_manifest_id: str | None = None
    verifier_manifest_sha256: str | None = None
    certification_report_id: str | None = None
    certification_decision: str | None = None
    bounded_closure_state: str | None = None
    bounded_external_distinctness_state: str | None = None
    positive_nonobviousness_authority_state: str | None = None
    fatal_blocker_state: str | None = None

    frozen_scientific_head_sha: str
    verifier_output_dir: str | None = None
    verifier_log_path: str | None = None

    second_repair_attempt_performed: Literal[False] = False
    original_prospective_result_preserved: Literal[True] = True
    repaired_specification_result_preserved: Literal[True] = True
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_case(self) -> "RepairedVerifierCaseResult":
        complete_fields = (
            self.verifier_manifest_id,
            self.verifier_manifest_sha256,
            self.certification_report_id,
            self.certification_decision,
            self.bounded_closure_state,
            self.bounded_external_distinctness_state,
            self.positive_nonobviousness_authority_state,
            self.fatal_blocker_state,
        )
        if self.status == "VERIFIER_COMPLETE_AFTER_R1":
            if any(value is None for value in complete_fields):
                raise ValueError(
                    "completed repaired verifier case requires full outcome"
                )
            if self.final_hypothesis_id is None:
                raise ValueError(
                    "completed repaired verifier case requires final hypothesis"
                )
            if self.candidate_hypothesis_id is None:
                raise ValueError(
                    "completed repaired verifier case requires candidate hypothesis"
                )
        return self


class RepairedRelationalVerifierCampaignReport(StrictModel):
    schema_version: Literal[
        "repaired-relational-verifier-campaign-report-v1"
    ] = "repaired-relational-verifier-campaign-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_reentry_report_id: str
    source_reentry_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_reentry_report_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_execution_plan_id: str
    source_campaign_launch_id: str
    frozen_scientific_head_sha: str

    cases: list[RepairedVerifierCaseResult]
    case_ids: list[str]
    case_count: Literal[5] = 5
    status_counts: dict[str, int]
    verifier_attempted_case_ids: list[str]
    verifier_complete_case_ids: list[str]
    certification_decision_counts: dict[str, int]

    verifier_executed_only_for_reentry_ready_cases: Literal[True] = True
    verifier_executed_with_frozen_scientific_code: Literal[True] = True
    verifier_settings_reused_from_frozen_execution_plan: Literal[True] = True
    second_repair_attempt_performed: Literal[False] = False
    original_prospective_results_preserved: Literal[True] = True
    repaired_specification_results_preserved: Literal[True] = True
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "RepairedRelationalVerifierCampaignReport":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        if self.case_ids != expected:
            raise ValueError("repaired verifier cases must be P06-P10")
        if [row.case_id for row in self.cases] != expected:
            raise ValueError(
                "repaired verifier case rows must be ordered P06-P10"
            )
        counts = Counter(row.status for row in self.cases)
        if dict(sorted(counts.items())) != dict(
            sorted(self.status_counts.items())
        ):
            raise ValueError("status_counts mismatch")

        attempted = [
            row.case_id
            for row in self.cases
            if row.status != "NOT_VERIFIER_READY_AFTER_R1"
        ]
        if attempted != self.verifier_attempted_case_ids:
            raise ValueError("verifier_attempted_case_ids mismatch")

        completed = [
            row.case_id
            for row in self.cases
            if row.status == "VERIFIER_COMPLETE_AFTER_R1"
        ]
        if completed != self.verifier_complete_case_ids:
            raise ValueError("verifier_complete_case_ids mismatch")

        decisions = Counter(
            row.certification_decision
            for row in self.cases
            if row.certification_decision is not None
        )
        if dict(sorted(decisions.items())) != dict(
            sorted(self.certification_decision_counts.items())
        ):
            raise ValueError("certification_decision_counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("repaired verifier campaign SHA mismatch")
        if observed_id != (
            "repaired_relational_verifier_campaign_report:"
            + expected_sha[:20]
        ):
            raise ValueError("repaired verifier campaign ID mismatch")
        return self


def build_repaired_verifier_campaign_report(
    *,
    source_reentry_report_id: str,
    source_reentry_report_sha256: str,
    source_reentry_report_file_sha256: str,
    source_execution_plan_id: str,
    source_campaign_launch_id: str,
    frozen_scientific_head_sha: str,
    cases: list[RepairedVerifierCaseResult],
) -> RepairedRelationalVerifierCampaignReport:
    if [row.case_id for row in cases] != [
        "P06", "P07", "P08", "P09", "P10"
    ]:
        raise ValueError("case rows must be ordered exactly P06-P10")

    status_counts = Counter(row.status for row in cases)
    attempted = [
        row.case_id
        for row in cases
        if row.status != "NOT_VERIFIER_READY_AFTER_R1"
    ]
    complete = [
        row.case_id
        for row in cases
        if row.status == "VERIFIER_COMPLETE_AFTER_R1"
    ]
    decisions = Counter(
        row.certification_decision
        for row in cases
        if row.certification_decision is not None
    )

    body = {
        "schema_version":
            "repaired-relational-verifier-campaign-report-v1",
        "source_reentry_report_id": source_reentry_report_id,
        "source_reentry_report_sha256": source_reentry_report_sha256,
        "source_reentry_report_file_sha256":
            source_reentry_report_file_sha256,
        "source_execution_plan_id": source_execution_plan_id,
        "source_campaign_launch_id": source_campaign_launch_id,
        "frozen_scientific_head_sha": frozen_scientific_head_sha,
        "cases": [row.model_dump(mode="json") for row in cases],
        "case_ids": ["P06", "P07", "P08", "P09", "P10"],
        "case_count": 5,
        "status_counts": dict(sorted(status_counts.items())),
        "verifier_attempted_case_ids": attempted,
        "verifier_complete_case_ids": complete,
        "certification_decision_counts": dict(sorted(decisions.items())),
        "verifier_executed_only_for_reentry_ready_cases": True,
        "verifier_executed_with_frozen_scientific_code": True,
        "verifier_settings_reused_from_frozen_execution_plan": True,
        "second_repair_attempt_performed": False,
        "original_prospective_results_preserved": True,
        "repaired_specification_results_preserved": True,
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return RepairedRelationalVerifierCampaignReport(
        **body,
        report_id=(
            "repaired_relational_verifier_campaign_report:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "RepairedRelationalVerifierCampaignReport",
    "RepairedVerifierCaseResult",
    "VerifierCaseStatus",
    "build_repaired_verifier_campaign_report",
]
