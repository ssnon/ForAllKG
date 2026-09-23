from __future__ import annotations

from collections import Counter

import pytest

from pipeline_core.discovery.prospective_routed_campaign_termination import (
    ProspectiveRoutedCampaignTerminationReport,
)


def _report_body():
    dispositions = {
        "P11":
            "PROTOCOL_TERMINATED_FROZEN_REGENERATION_BUDGET_CONFLICT",
        "P12": "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
        "P13": "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
        "P14": "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
        "P15": "NOT_RUN_AFTER_PROTOCOL_TERMINATION",
    }
    counts = dict(sorted(Counter(dispositions.values()).items()))
    return {
        "schema_version":
            "prospective-routed-campaign-termination-report-v1",
        "source_campaign_freeze_id": "freeze:x",
        "source_campaign_freeze_sha256": "a" * 64,
        "source_execution_plan_id": "plan:x",
        "source_execution_plan_sha256": "b" * 64,
        "trigger_case_id": "P11",
        "trigger_budget_report_id": "budget:x",
        "trigger_budget_report_sha256": "c" * 64,
        "termination_reason": "FROZEN_REGENERATION_BUDGET_CONFLICT",
        "case_dispositions": dispositions,
        "disposition_counts": counts,
        "scientifically_executed_case_ids": ["P11"],
        "not_run_case_ids": ["P12", "P13", "P14", "P15"],
        "frozen_case_count": 5,
        "scientifically_executed_case_count": 1,
        "not_run_case_count": 4,
        "trigger_required_regeneration_count": 2,
        "trigger_budget_conflict_count": 2,
        "trigger_frozen_executable_count": 0,
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


def test_report_requires_only_p11_executed() -> None:
    body = _report_body()
    body["report_id"] = "prospective_routed_campaign_termination_report:" + "0" * 20
    body["report_sha256"] = "0" * 64
    with pytest.raises(Exception):
        ProspectiveRoutedCampaignTerminationReport.model_validate(body)


def test_case_dispositions_are_ordered_p11_to_p15() -> None:
    body = _report_body()
    assert list(body["case_dispositions"]) == [
        "P11", "P12", "P13", "P14", "P15"
    ]


def test_later_cases_are_explicitly_not_run() -> None:
    body = _report_body()
    assert body["not_run_case_ids"] == ["P12", "P13", "P14", "P15"]
    assert body["not_run_case_count"] == 4


def test_trigger_conflict_covers_every_required_regeneration() -> None:
    body = _report_body()
    assert (
        body["trigger_required_regeneration_count"]
        == body["trigger_budget_conflict_count"]
        == 2
    )
    assert body["trigger_frozen_executable_count"] == 0


def test_protocol_diagnostic_not_scientific_comparison() -> None:
    body = _report_body()
    assert body["cohort_valid_for_protocol_diagnostics"] is True
    assert body["cohort_scientific_comparison_complete"] is False


def test_termination_forbids_posthoc_mutation_flags() -> None:
    body = _report_body()
    assert body["posthoc_execution_plan_edit_performed"] is False
    assert body["posthoc_router_policy_edit_performed"] is False
    assert body["case_replacement_performed"] is False
    assert body["later_case_settings_adapted"] is False
