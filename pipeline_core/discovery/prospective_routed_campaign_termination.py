from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRoutedCampaignFreeze,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    ProspectiveRoutedExecutionPlan,
)
from pipeline_core.discovery.prospective_routed_regeneration_budget_accounting import (
    ProspectiveRoutedRegenerationBudgetAccountingReport,
)


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
    "PROTOCOL_TERMINATED_FROZEN_REGENERATION_BUDGET_CONFLICT",
    "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
]


class ProspectiveRoutedCampaignTerminationReport(StrictModel):
    schema_version: Literal[
        "prospective-routed-campaign-termination-report-v1"
    ] = "prospective-routed-campaign-termination-report-v1"

    report_id: str
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    source_campaign_freeze_id: str
    source_campaign_freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_execution_plan_id: str
    source_execution_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    trigger_case_id: Literal["P11"] = "P11"
    trigger_budget_report_id: str
    trigger_budget_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    termination_reason: Literal[
        "FROZEN_REGENERATION_BUDGET_CONFLICT"
    ] = "FROZEN_REGENERATION_BUDGET_CONFLICT"

    case_dispositions: dict[str, CaseDisposition]
    disposition_counts: dict[str, int]

    scientifically_executed_case_ids: list[str]
    not_run_case_ids: list[str]

    frozen_case_count: Literal[5] = 5
    scientifically_executed_case_count: int = Field(ge=0, le=5)
    not_run_case_count: int = Field(ge=0, le=5)

    trigger_required_regeneration_count: int = Field(ge=1)
    trigger_budget_conflict_count: int = Field(ge=1)
    trigger_frozen_executable_count: Literal[0] = 0

    all_frozen_cases_dispositioned: Literal[True] = True
    later_cases_verified_unexecuted: Literal[True] = True

    regeneration_executed_after_conflict: Literal[False] = False
    posthoc_execution_plan_edit_performed: Literal[False] = False
    posthoc_router_policy_edit_performed: Literal[False] = False
    case_replacement_performed: Literal[False] = False
    later_case_settings_adapted: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False

    cohort_scientific_comparison_complete: Literal[False] = False
    cohort_valid_for_protocol_diagnostics: Literal[True] = True

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "ProspectiveRoutedCampaignTerminationReport":
        expected = ["P11", "P12", "P13", "P14", "P15"]
        if list(self.case_dispositions) != expected:
            raise ValueError(
                "termination case dispositions must be ordered P11-P15"
            )

        if self.scientifically_executed_case_ids != ["P11"]:
            raise ValueError(
                "protocol-terminated cohort must record only P11 executed"
            )
        if self.not_run_case_ids != ["P12", "P13", "P14", "P15"]:
            raise ValueError(
                "protocol-terminated cohort must leave P12-P15 unrun"
            )
        if self.scientifically_executed_case_count != 1:
            raise ValueError(
                "scientifically_executed_case_count must equal one"
            )
        if self.not_run_case_count != 4:
            raise ValueError("not_run_case_count must equal four")

        counts = Counter(self.case_dispositions.values())
        if dict(sorted(counts.items())) != dict(
            sorted(self.disposition_counts.items())
        ):
            raise ValueError("termination disposition_counts mismatch")

        if (
            self.trigger_required_regeneration_count
            != self.trigger_budget_conflict_count
        ):
            raise ValueError(
                "every required P11 regeneration must be budget-conflicted "
                "for campaign termination"
            )

        body = self.model_dump(mode="json")
        observed_id = body.pop("report_id")
        observed_sha = body.pop("report_sha256")
        expected_sha = _sha256_json(body)
        if observed_sha != expected_sha:
            raise ValueError("campaign termination report SHA mismatch")
        if observed_id != (
            "prospective_routed_campaign_termination_report:"
            + expected_sha[:20]
        ):
            raise ValueError("campaign termination report ID mismatch")
        return self


def _directory_has_payload(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(path.iterdir())


def build_campaign_termination_report(
    *,
    freeze: ProspectiveRoutedCampaignFreeze,
    plan: ProspectiveRoutedExecutionPlan,
    budget_report: ProspectiveRoutedRegenerationBudgetAccountingReport,
) -> ProspectiveRoutedCampaignTerminationReport:
    expected = ["P11", "P12", "P13", "P14", "P15"]

    if freeze.case_ids != expected:
        raise ValueError("campaign freeze is not P11-P15")
    if plan.case_ids != expected:
        raise ValueError("execution plan is not P11-P15")
    if plan.source_campaign_freeze_id != freeze.freeze_id:
        raise ValueError("execution-plan/campaign-freeze ID mismatch")
    if plan.source_campaign_freeze_sha256 != freeze.freeze_sha256:
        raise ValueError("execution-plan/campaign-freeze SHA mismatch")

    if budget_report.case_id != "P11":
        raise ValueError("termination trigger must be P11")
    if budget_report.source_execution_plan_id != plan.plan_id:
        raise ValueError("budget-report/execution-plan ID mismatch")
    if budget_report.source_execution_plan_sha256 != plan.plan_sha256:
        raise ValueError("budget-report/execution-plan SHA mismatch")
    if budget_report.regeneration_executed:
        raise ValueError(
            "cannot protocol-terminate after regeneration execution"
        )
    if budget_report.llm_calls_performed != 0:
        raise ValueError(
            "protocol termination requires zero fallback LLM calls"
        )
    if budget_report.regeneration_required_count < 1:
        raise ValueError(
            "termination requires at least one frozen regeneration fallback"
        )
    if (
        budget_report.frozen_budget_conflict_count
        != budget_report.regeneration_required_count
    ):
        raise ValueError(
            "termination requires every required regeneration to conflict"
        )
    if budget_report.frozen_executable_count != 0:
        raise ValueError(
            "termination requires zero frozen executable fallbacks"
        )

    case_by_id = {row.case_id: row for row in plan.cases}
    for case_id in ["P12", "P13", "P14", "P15"]:
        run_dir = Path(case_by_id[case_id].run_dir).expanduser().resolve()
        if _directory_has_payload(run_dir):
            raise ValueError(
                "later prospective case has already been executed or "
                "materialized: " + case_id + " -> " + str(run_dir)
            )

    dispositions: dict[str, CaseDisposition] = {
        "P11":
            "PROTOCOL_TERMINATED_FROZEN_REGENERATION_BUDGET_CONFLICT",
        "P12": "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
        "P13": "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
        "P14": "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
        "P15": "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
    }
    counts = Counter(dispositions.values())

    body = {
        "schema_version":
            "prospective-routed-campaign-termination-report-v1",
        "source_campaign_freeze_id": freeze.freeze_id,
        "source_campaign_freeze_sha256": freeze.freeze_sha256,
        "source_execution_plan_id": plan.plan_id,
        "source_execution_plan_sha256": plan.plan_sha256,
        "trigger_case_id": "P11",
        "trigger_budget_report_id": budget_report.report_id,
        "trigger_budget_report_sha256": budget_report.report_sha256,
        "termination_reason": "FROZEN_REGENERATION_BUDGET_CONFLICT",
        "case_dispositions": dispositions,
        "disposition_counts": dict(sorted(counts.items())),
        "scientifically_executed_case_ids": ["P11"],
        "not_run_case_ids": ["P12", "P13", "P14", "P15"],
        "frozen_case_count": 5,
        "scientifically_executed_case_count": 1,
        "not_run_case_count": 4,
        "trigger_required_regeneration_count":
            budget_report.regeneration_required_count,
        "trigger_budget_conflict_count":
            budget_report.frozen_budget_conflict_count,
        "trigger_frozen_executable_count":
            budget_report.frozen_executable_count,
        "all_frozen_cases_dispositioned": True,
        "later_cases_verified_unexecuted": True,
        "regeneration_executed_after_conflict": False,
        "posthoc_execution_plan_edit_performed": False,
        "posthoc_router_policy_edit_performed": False,
        "case_replacement_performed": False,
        "later_case_settings_adapted": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
        "cohort_scientific_comparison_complete": False,
        "cohort_valid_for_protocol_diagnostics": True,
    }
    digest = _sha256_json(body)
    return ProspectiveRoutedCampaignTerminationReport(
        **body,
        report_id=(
            "prospective_routed_campaign_termination_report:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


__all__ = [
    "CaseDisposition",
    "ProspectiveRoutedCampaignTerminationReport",
    "build_campaign_termination_report",
]
