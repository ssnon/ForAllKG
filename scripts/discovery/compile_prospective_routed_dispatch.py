from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Report,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze import (
    ProspectiveRoutedCampaignFreeze,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    ProspectiveRoutedExecutionPlan,
)
from pipeline_core.discovery.prospective_routed_route_compiler import (
    compile_routed_dispatch,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one P11-P15 Gate-v2 result into the frozen "
            "repair/alignment/decomposition/regeneration route work bundle. "
            "This compiler performs no LLM call and no scientific mutation."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--campaign-freeze", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--binding-plan", required=True, type=Path)
    parser.add_argument("--gate-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    execution = ProspectiveRoutedExecutionPlan.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    campaign = ProspectiveRoutedCampaignFreeze.model_validate_json(
        args.campaign_freeze.expanduser().resolve().read_text(encoding="utf-8")
    )
    if execution.source_campaign_freeze_id != campaign.freeze_id:
        raise ValueError("execution-plan/campaign-freeze ID mismatch")
    if execution.source_campaign_freeze_sha256 != campaign.freeze_sha256:
        raise ValueError("execution-plan/campaign-freeze SHA mismatch")

    case_rows = [row for row in execution.cases if row.case_id == args.case_id]
    if len(case_rows) != 1:
        raise ValueError("case-id is not present exactly once in execution plan")
    case = case_rows[0]

    plan = RelationalAtomicBindingPlan.model_validate_json(
        args.binding_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    gate = PreVerifierContractGateV2Report.model_validate_json(
        args.gate_report.expanduser().resolve().read_text(encoding="utf-8")
    )

    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise ValueError("routed dispatch report is write-once")

    report = compile_routed_dispatch(
        case=case,
        binding_plan=plan,
        gate_report=gate,
        router_policy=campaign.router_policy,
    )
    write_json_exclusive(output_path, report)

    print("Prospective routed dispatch compiled")
    print("Case:", report.case_id)
    print("Report:", report.report_id)
    print("Hypotheses:", report.hypothesis_count)
    print("Claims:", report.claim_count)
    print("Routes:", report.route_action_counts)
    print()
    for row in report.hypotheses:
        print(
            row.final_hypothesis_id,
            "|",
            row.dominant_router_hint,
            "->",
            row.route_action,
        )
        for claim in row.claim_work_items:
            print(
                " ",
                claim.claim_id,
                "|",
                claim.router_hint,
                "| edit=",
                claim.specification_repair_editable_fields,
                "| source-pairs=",
                len(claim.source_alignment_candidates),
            )
        if row.regeneration_fallback_allowed:
            print("  regeneration run:", row.regeneration_run_dir)
    print()
    print("LLM calls performed: 0")
    print("Scientific mutation performed: false")
    print("Endpoint binding performed: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
