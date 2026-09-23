from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRoutedCampaignFreeze,
)
from pipeline_core.discovery.prospective_routed_campaign_termination import (
    build_campaign_termination_report,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    ProspectiveRoutedExecutionPlan,
)
from pipeline_core.discovery.prospective_routed_regeneration_budget_accounting import (
    ProspectiveRoutedRegenerationBudgetAccountingReport,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a P11-P15 routed prospective cohort as "
            "protocol-terminated after a frozen regeneration-budget "
            "conflict. This command verifies that P12-P15 remain unrun and "
            "performs no scientific execution."
        )
    )
    parser.add_argument("--campaign-freeze", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--budget-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(
            "prospective routed campaign termination output is write-once"
        )

    freeze = ProspectiveRoutedCampaignFreeze.model_validate_json(
        args.campaign_freeze.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    plan = ProspectiveRoutedExecutionPlan.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    budget = (
        ProspectiveRoutedRegenerationBudgetAccountingReport.model_validate_json(
            args.budget_report.expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
    )

    report = build_campaign_termination_report(
        freeze=freeze,
        plan=plan,
        budget_report=budget,
    )
    write_json_exclusive(output, report)

    print("Prospective routed campaign protocol termination frozen")
    print("Report:", report.report_id)
    print("Termination reason:", report.termination_reason)
    print("Executed cases:", report.scientifically_executed_case_ids)
    print("Not run:", report.not_run_case_ids)
    print("Dispositions:", report.case_dispositions)
    print("Scientific comparison complete: false")
    print("Valid for protocol diagnostics: true")
    print("Case replacement performed: false")
    print("Later-case settings adapted: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
