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


CaseDisposition = Literal[
    "MAIN_E2E_STAGE_FAILED",
    "UPSTREAM_NO_RELATIONAL_BINDING_INPUTS",
    "BINDING_PLAN_STAGE_FAILED",
    "NO_BINDING_READY_HYPOTHESIS",
    "STRUCTURAL_SELECTION_STAGE_FAILED",
    "ENDPOINT_BINDING_STAGE_FAILED",
    "ENDPOINT_BINDING_ABSTAINED",
    "VERIFIER_STAGE_FAILED",
    "VERIFIER_COMPLETE",
]


class ProspectiveCampaignStageRecord(StrictModel):
    stage_name: str
    argv: list[str]
    return_code: int
    log_path: str


class ProspectiveRelationalCaseResult(StrictModel):
    schema_version: Literal[
        "prospective-relational-case-result-v1"
    ] = "prospective-relational-case-result-v1"

    result_id: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    case_id: Literal["P06", "P07", "P08", "P09", "P10"]
    source_task_id: str
    execution_plan_id: str
    scientific_repository_head_sha: str

    disposition: CaseDisposition
    stage_records: list[ProspectiveCampaignStageRecord]

    main_e2e_manifest_status: str | None = None
    selected_final_hypothesis_id: str | None = None
    selected_candidate_hypothesis_id: str | None = None
    selected_original_hypothesis_id: str | None = None

    endpoint_selected_claim_count: int | None = Field(default=None, ge=0)
    endpoint_bound_claim_count: int | None = Field(default=None, ge=0)
    endpoint_abstained_claim_count: int | None = Field(default=None, ge=0)
    endpoint_novelty_bearing_bound_claim_count: int | None = Field(
        default=None,
        ge=0,
    )

    verifier_manifest_id: str | None = None
    certification_decision: str | None = None
    bounded_closure_state: str | None = None
    bounded_external_distinctness_state: str | None = None
    positive_nonobviousness_authority_state: str | None = None
    fatal_blocker_state: str | None = None

    case_replaced: Literal[False] = False
    settings_adapted_after_previous_case: Literal[False] = False
    endpoint_outcome_used_for_hypothesis_selection: Literal[False] = False
    verifier_outcome_used_for_hypothesis_selection: Literal[False] = False
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> "ProspectiveRelationalCaseResult":
        if self.disposition == "VERIFIER_COMPLETE":
            required = (
                self.verifier_manifest_id,
                self.certification_decision,
                self.bounded_closure_state,
                self.bounded_external_distinctness_state,
                self.positive_nonobviousness_authority_state,
                self.fatal_blocker_state,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "VERIFIER_COMPLETE requires complete verifier outcome"
                )

        body = self.model_dump(mode="json")
        observed_id = body.pop("result_id")
        observed_sha = body.pop("result_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective case result SHA mismatch")
        if observed_id != (
            "prospective_relational_case_result:" + expected_sha[:20]
        ):
            raise ValueError("prospective case result ID mismatch")
        return self


class ProspectiveRelationalCampaignLaunch(StrictModel):
    schema_version: Literal[
        "prospective-relational-campaign-launch-v1"
    ] = "prospective-relational-campaign-launch-v1"

    launch_id: str
    launch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    execution_plan_id: str
    execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_plan_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    launcher_repository_head_sha: str
    scientific_repository_head_sha: str
    pinned_worktree_path: str
    case_ids: list[str]

    frozen_head_used_for_all_scientific_stages: Literal[True] = True
    all_five_cases_launched_under_one_frozen_plan: Literal[True] = True
    post_case_adaptation_allowed: Literal[False] = False
    case_replacement_allowed: Literal[False] = False
    campaign_launch_write_once: Literal[True] = True

    @model_validator(mode="after")
    def validate_launch(self) -> "ProspectiveRelationalCampaignLaunch":
        if self.case_ids != ["P06", "P07", "P08", "P09", "P10"]:
            raise ValueError("campaign launch case IDs must be P06-P10")
        body = self.model_dump(mode="json")
        observed_id = body.pop("launch_id")
        observed_sha = body.pop("launch_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective campaign launch SHA mismatch")
        if observed_id != (
            "prospective_relational_campaign_launch:" + expected_sha[:20]
        ):
            raise ValueError("prospective campaign launch ID mismatch")
        return self


class ProspectiveRelationalCampaignResult(StrictModel):
    schema_version: Literal[
        "prospective-relational-campaign-result-v1"
    ] = "prospective-relational-campaign-result-v1"

    campaign_result_id: str
    campaign_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    launch_id: str
    launch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_plan_id: str
    scientific_repository_head_sha: str

    case_result_ids: list[str]
    case_result_sha256s: list[str]
    case_dispositions: dict[str, str]
    disposition_counts: dict[str, int]

    case_count: Literal[5] = 5
    all_cases_accounted_for: Literal[True] = True
    case_replacement_performed: Literal[False] = False
    post_case_adaptation_performed: Literal[False] = False
    verifier_result_consumed_by_production: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False
    campaign_result_write_once: Literal[True] = True

    @model_validator(mode="after")
    def validate_campaign(
        self,
    ) -> "ProspectiveRelationalCampaignResult":
        expected = ["P06", "P07", "P08", "P09", "P10"]
        if list(self.case_dispositions) != expected:
            raise ValueError("campaign result cases must be ordered P06-P10")
        if len(self.case_result_ids) != 5:
            raise ValueError("campaign result requires five case result IDs")
        if len(self.case_result_sha256s) != 5:
            raise ValueError("campaign result requires five case hashes")

        observed = Counter(self.case_dispositions.values())
        if dict(sorted(observed.items())) != dict(
            sorted(self.disposition_counts.items())
        ):
            raise ValueError("campaign disposition_counts mismatch")

        body = self.model_dump(mode="json")
        observed_id = body.pop("campaign_result_id")
        observed_sha = body.pop("campaign_result_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("prospective campaign result SHA mismatch")
        if observed_id != (
            "prospective_relational_campaign_result:" + expected_sha[:20]
        ):
            raise ValueError("prospective campaign result ID mismatch")
        return self


def build_campaign_launch(
    *,
    execution_plan_id: str,
    execution_plan_sha256: str,
    execution_plan_file_sha256: str,
    launcher_repository_head_sha: str,
    scientific_repository_head_sha: str,
    pinned_worktree_path: str,
) -> ProspectiveRelationalCampaignLaunch:
    body = {
        "schema_version": "prospective-relational-campaign-launch-v1",
        "execution_plan_id": execution_plan_id,
        "execution_plan_sha256": execution_plan_sha256,
        "execution_plan_file_sha256": execution_plan_file_sha256,
        "launcher_repository_head_sha": launcher_repository_head_sha,
        "scientific_repository_head_sha": scientific_repository_head_sha,
        "pinned_worktree_path": pinned_worktree_path,
        "case_ids": ["P06", "P07", "P08", "P09", "P10"],
        "frozen_head_used_for_all_scientific_stages": True,
        "all_five_cases_launched_under_one_frozen_plan": True,
        "post_case_adaptation_allowed": False,
        "case_replacement_allowed": False,
        "campaign_launch_write_once": True,
    }
    digest = _sha256_json(body)
    return ProspectiveRelationalCampaignLaunch(
        **body,
        launch_id="prospective_relational_campaign_launch:" + digest[:20],
        launch_sha256=digest,
    )


def build_case_result(
    **kwargs: object,
) -> ProspectiveRelationalCaseResult:
    body = {
        "schema_version": "prospective-relational-case-result-v1",
        **kwargs,
        "case_replaced": False,
        "settings_adapted_after_previous_case": False,
        "endpoint_outcome_used_for_hypothesis_selection": False,
        "verifier_outcome_used_for_hypothesis_selection": False,
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha256_json(body)
    return ProspectiveRelationalCaseResult(
        **body,
        result_id="prospective_relational_case_result:" + digest[:20],
        result_sha256=digest,
    )


def build_campaign_result(
    *,
    launch: ProspectiveRelationalCampaignLaunch,
    execution_plan_id: str,
    scientific_repository_head_sha: str,
    case_results: list[ProspectiveRelationalCaseResult],
) -> ProspectiveRelationalCampaignResult:
    if [row.case_id for row in case_results] != [
        "P06", "P07", "P08", "P09", "P10"
    ]:
        raise ValueError("case results must be ordered exactly P06-P10")

    case_dispositions = {
        row.case_id: row.disposition
        for row in case_results
    }
    counts = dict(sorted(Counter(case_dispositions.values()).items()))
    body = {
        "schema_version": "prospective-relational-campaign-result-v1",
        "launch_id": launch.launch_id,
        "launch_sha256": launch.launch_sha256,
        "execution_plan_id": execution_plan_id,
        "scientific_repository_head_sha": scientific_repository_head_sha,
        "case_result_ids": [row.result_id for row in case_results],
        "case_result_sha256s": [row.result_sha256 for row in case_results],
        "case_dispositions": case_dispositions,
        "disposition_counts": counts,
        "case_count": 5,
        "all_cases_accounted_for": True,
        "case_replacement_performed": False,
        "post_case_adaptation_performed": False,
        "verifier_result_consumed_by_production": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "campaign_result_write_once": True,
    }
    digest = _sha256_json(body)
    return ProspectiveRelationalCampaignResult(
        **body,
        campaign_result_id=(
            "prospective_relational_campaign_result:" + digest[:20]
        ),
        campaign_result_sha256=digest,
    )


__all__ = [
    "ProspectiveCampaignStageRecord",
    "ProspectiveRelationalCampaignLaunch",
    "ProspectiveRelationalCampaignResult",
    "ProspectiveRelationalCaseResult",
    "build_campaign_launch",
    "build_campaign_result",
    "build_case_result",
]
