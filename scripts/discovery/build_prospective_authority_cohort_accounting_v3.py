from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.pre_n10_prospective_campaign_v1 import (
    write_exact_or_validate,
)
from pipeline_core.discovery.prospective_authority_cohort_accounting_v3 import (
    build_prospective_authority_cohort_accounting_v3,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build write-once P21-P25 prospective authority cohort outcome "
            "accounting without changing any scientific outcome."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = build_prospective_authority_cohort_accounting_v3(
        execution_plan_path=args.execution_plan,
    )
    output = args.output.expanduser().resolve()
    write_exact_or_validate(output, report)

    print("P21-P25 prospective authority cohort accounting v3 complete")
    print("Report:", report.report_id)
    print("Execution plan:", report.source_execution_plan_id)
    print("Cases:", report.case_ids)
    print()
    for row in report.cases:
        print(
            row.case_id,
            "| H=", row.initial_hypothesis_count,
            "| semantic=", row.semantic_reached,
            "| entry=", row.semantic_entry_authorized,
            "| Vpre-ready=", row.initial_vpre_ready_hypothesis_count,
            "| regen=", row.regeneration_fallback_count,
            "| regen-ready=", row.regenerated_ready_for_n10_count,
            "| external=", row.external_eligible_lineage_count,
            "| Vpost=", row.vpost_completed_count,
            "| final=", row.final_status,
        )
    print()
    print("Initial hypotheses:", report.initial_hypothesis_total)
    print(
        "Case reachability alpha4/semantic/entry/Vpre:",
        report.alpha4_nonempty_case_count,
        "/", report.semantic_reached_case_count,
        "/", report.semantic_entry_authorized_case_count,
        "/", report.vpre_executed_case_count,
    )
    print(
        "Initial V_pre ready hypotheses:",
        report.initial_vpre_ready_hypothesis_total,
        "/", report.initial_vpre_hypothesis_evaluated_total,
    )
    print(
        "Regenerated semantic-admissible / N10-ready:",
        report.regenerated_semantic_admissible_count,
        "/", report.regenerated_ready_for_n10_count,
    )
    print(
        "External/N10/binding/V_post:",
        report.external_eligible_lineage_count,
        "/",
        (
            report.n10_certified_count
            + report.n10_unresolved_count
            + report.n10_rejected_count
        ),
        "/", report.binding_ready_lineage_count,
        "/", report.vpost_completed_count,
    )
    print("Final statuses:", report.final_status_counts)
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
