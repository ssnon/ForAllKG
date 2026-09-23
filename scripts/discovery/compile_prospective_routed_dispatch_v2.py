from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    PreVerifierContractGateV2Report,
)
from pipeline_core.discovery.prospective_regeneration_downstream_v2 import (
    ProspectiveRegenerationDownstreamV2Freeze,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
)
from pipeline_core.discovery.prospective_routed_campaign_freeze_v2 import (
    ProspectiveRoutedCampaignFreezeV2,
)
from pipeline_core.discovery.prospective_routed_execution_plan_v2 import (
    ProspectiveRoutedExecutionPlanV2,
)
from pipeline_core.discovery.prospective_routed_route_compiler_v2 import (
    compile_routed_dispatch_v2,
)
from pipeline_core.discovery.relational_atomic_binding_plan import (
    RelationalAtomicBindingPlan,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def _require_exact_path(
    *,
    actual: Path,
    frozen: str,
    label: str,
) -> None:
    expected = Path(frozen).expanduser().resolve()
    if actual != expected:
        raise ValueError(
            label
            + " path differs from frozen execution plan: "
            + str(actual)
            + " != "
            + str(expected)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one P16-P20 Gate-v2 result into the frozen routed-v2 "
            "work bundle. Regeneration is represented only by "
            "regeneration-unit-v2/downstream-v2 lineage paths; this compiler "
            "never emits or executes a regeneration full-E2E argv."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--campaign-freeze", required=True, type=Path)
    parser.add_argument(
        "--regeneration-unit-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--regeneration-downstream-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--binding-plan", required=True, type=Path)
    parser.add_argument("--gate-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    execution = ProspectiveRoutedExecutionPlanV2.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    campaign = ProspectiveRoutedCampaignFreezeV2.model_validate_json(
        args.campaign_freeze.expanduser().resolve().read_text(encoding="utf-8")
    )
    unit = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        args.regeneration_unit_freeze.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    downstream = ProspectiveRegenerationDownstreamV2Freeze.model_validate_json(
        args.regeneration_downstream_freeze.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )

    if execution.source_campaign_freeze_id != campaign.freeze_id:
        raise ValueError("execution-plan/campaign-freeze ID mismatch")
    if execution.source_campaign_freeze_sha256 != campaign.freeze_sha256:
        raise ValueError("execution-plan/campaign-freeze SHA mismatch")

    if execution.source_regeneration_unit_freeze_id != unit.freeze_id:
        raise ValueError("execution-plan/regeneration-unit ID mismatch")
    if execution.source_regeneration_unit_freeze_sha256 != unit.freeze_sha256:
        raise ValueError("execution-plan/regeneration-unit SHA mismatch")

    if (
        execution.source_regeneration_downstream_freeze_id
        != downstream.freeze_id
    ):
        raise ValueError("execution-plan/downstream ID mismatch")
    if (
        execution.source_regeneration_downstream_freeze_sha256
        != downstream.freeze_sha256
    ):
        raise ValueError("execution-plan/downstream SHA mismatch")

    if downstream.source_regeneration_unit_freeze_id != unit.freeze_id:
        raise ValueError("downstream/regeneration-unit ID mismatch")
    if downstream.source_regeneration_unit_freeze_sha256 != unit.freeze_sha256:
        raise ValueError("downstream/regeneration-unit SHA mismatch")

    case_rows = [
        row for row in execution.cases if row.case_id == args.case_id
    ]
    if len(case_rows) != 1:
        raise ValueError(
            "case-id is not present exactly once in execution plan"
        )
    case = case_rows[0]

    binding_path = args.binding_plan.expanduser().resolve()
    gate_path = args.gate_report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    _require_exact_path(
        actual=binding_path,
        frozen=case.full_binding_plan_path,
        label="binding plan",
    )
    _require_exact_path(
        actual=gate_path,
        frozen=case.pre_route_gate_path,
        label="pre-route gate",
    )
    _require_exact_path(
        actual=output_path,
        frozen=case.routed_dispatch_path,
        label="dispatch output",
    )

    if output_path.exists():
        raise ValueError("routed dispatch-v2 report is write-once")

    plan = RelationalAtomicBindingPlan.model_validate_json(
        binding_path.read_text(encoding="utf-8")
    )
    gate = PreVerifierContractGateV2Report.model_validate_json(
        gate_path.read_text(encoding="utf-8")
    )

    report = compile_routed_dispatch_v2(
        case=case,
        binding_plan=plan,
        gate_report=gate,
        router_policy=campaign.router_policy,
        execution_plan_id=execution.plan_id,
        execution_plan_sha256=execution.plan_sha256,
        campaign_freeze_id=campaign.freeze_id,
        campaign_freeze_sha256=campaign.freeze_sha256,
        regeneration_unit_freeze_id=unit.freeze_id,
        regeneration_unit_freeze_sha256=unit.freeze_sha256,
        regeneration_downstream_freeze_id=downstream.freeze_id,
        regeneration_downstream_freeze_sha256=downstream.freeze_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(output_path, report)

    print("Prospective routed dispatch-v2 compiled")
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
        if row.regeneration_fallback is not None:
            print(
                "  regeneration unit:",
                row.regeneration_fallback.unit_result_path,
            )
            print(
                "  regeneration downstream:",
                row.regeneration_fallback.downstream_report_path,
            )

    print()
    print("Regeneration full E2E argv generated: false")
    print("LLM calls performed: 0")
    print("Scientific mutation performed: false")
    print("Endpoint binding performed: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
