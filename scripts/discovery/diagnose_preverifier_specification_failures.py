from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.preverifier_specification_diagnostics import (
    build_preverifier_specification_diagnostic_report,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only P06-P10 pre-verifier failure diagnostic "
            "report before any specification repair is attempted."
        )
    )
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--campaign-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(
            "pre-verifier diagnostic report is write-once; use a fresh output path"
        )

    report = build_preverifier_specification_diagnostic_report(
        campaign_root=args.campaign_root,
        campaign_result_path=args.campaign_result,
    )
    write_json_exclusive(output, report)

    print("Pre-verifier specification diagnosis complete")
    print("Report:", report.report_id)
    print("Cases:", report.case_ids)
    print("Primary diagnostics:", report.primary_diagnostic_counts)
    print("Automatic repair eligible:", report.automatic_repair_eligible_count, "/", report.case_count)
    print()
    for row in report.cases:
        print(
            row.case_id,
            "| disposition=", row.original_disposition,
            "| primary=", row.primary_diagnostic_class,
            "| repair=", row.proposed_repair_actions,
            "| delta=", row.scientific_delta_policy,
            "| auto=", row.automatic_repair_eligible,
        )
        if row.failed_stage_name:
            print("  failed stage:", row.failed_stage_name)
        if row.failure_type:
            print("  failure type:", row.failure_type)
        if row.failure_message:
            print("  failure message:", row.failure_message)
        for claim in row.claim_diagnostics:
            print(" ", claim.claim_id, "|", claim.diagnostic_classes, "|", claim.proposed_repair_actions)
            if claim.abstention_reason:
                print("   abstention:", claim.abstention_reason)
            if claim.source_claim.reason_codes:
                print("   binding reasons:", claim.source_claim.reason_codes)
    print()
    print("Diagnosis only: true")
    print("Repair performed: false")
    print("LLM calls performed: 0")
    print("External novelty outcome used: false")
    print("Verifier outcome used: false")
    print("Old N10 status used as repair feature: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
