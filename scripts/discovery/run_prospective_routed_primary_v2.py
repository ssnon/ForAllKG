from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_routed_execution_plan_v2 import ProspectiveRoutedExecutionPlanV2
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    InstructorSourceAlignmentAuditBackend,
    execute_primary_materialization_v2,
)
from pipeline_core.discovery.prospective_routed_route_compiler_v2 import ProspectiveRoutedDispatchReportV2
from pipeline_core.discovery.relational_atomic_binding_plan import RelationalAtomicBindingPlan
from pipeline_core.discovery.relational_scientific_verifier_shadow import write_json_exclusive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--binding-plan", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    args = parser.parse_args()

    execution = ProspectiveRoutedExecutionPlanV2.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    cases = [row for row in execution.cases if row.case_id == args.case_id]
    if len(cases) != 1:
        raise ValueError("case is not present exactly once in execution plan")
    case = cases[0]

    binding_path = args.binding_plan.expanduser().resolve()
    dispatch_path = args.dispatch.expanduser().resolve()
    if binding_path != Path(case.full_binding_plan_path).expanduser().resolve():
        raise ValueError("binding plan path differs from frozen execution plan")
    if dispatch_path != Path(case.routed_dispatch_path).expanduser().resolve():
        raise ValueError("dispatch path differs from frozen execution plan")

    binding = RelationalAtomicBindingPlan.model_validate_json(binding_path.read_text(encoding="utf-8"))
    dispatch = ProspectiveRoutedDispatchReportV2.model_validate_json(dispatch_path.read_text(encoding="utf-8"))
    if dispatch.source_execution_plan_id != execution.plan_id:
        raise ValueError("dispatch/execution-plan ID mismatch")
    if dispatch.source_execution_plan_sha256 != execution.plan_sha256:
        raise ValueError("dispatch/execution-plan SHA mismatch")

    routed = Path(case.routed_dispatch_path).expanduser().resolve().parent
    primary_root = routed / "primary_v2"
    plan_path = primary_root / "relational_atomic_binding_plan.primary.json"
    gate_path = primary_root / "contract_gate_v2.primary.json"
    report_path = primary_root / "primary_materialization.v2.json"
    telemetry_path = primary_root / "source_alignment.audit.telemetry.jsonl"

    for path in (plan_path, gate_path, report_path):
        if path.exists():
            raise ValueError("primary-v2 output is write-once and already exists: " + str(path))

    backend = InstructorSourceAlignmentAuditBackend(
        model=case.route_audit_model,
        api_key_env=execution.settings.api_key_env,
        base_url=execution.settings.base_url,
        temperature=execution.settings.route_temperature,
        parse_retries=execution.settings.route_parse_retries,
        timeout_seconds=execution.settings.route_timeout_seconds,
        telemetry_path=telemetry_path,
        telemetry_context={
            "case_id": case.case_id,
            "execution_plan_id": execution.plan_id,
            "dispatch_report_id": dispatch.report_id,
        },
    )

    primary_plan, primary_gate, report = execute_primary_materialization_v2(
        binding_plan=binding,
        dispatch=dispatch,
        audit_backend=backend,
        primary_root=primary_root,
    )

    primary_root.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(plan_path, primary_plan)
    write_json_exclusive(gate_path, primary_gate)
    write_json_exclusive(report_path, report)

    print("Prospective routed primary-v2 complete")
    print("Case:", report.case_id)
    print("Report:", report.report_id)
    print(
        "Source alignments materialized/ambiguous/semantic-reject/audit-fail:",
        report.materialized_source_alignment_count, "/",
        report.source_alignment_ambiguous_count, "/",
        report.source_alignment_semantic_reject_count, "/",
        report.source_alignment_audit_failure_count,
    )
    print(
        "Decompositions materialized/unavailable:",
        report.materialized_decomposition_count, "/",
        report.unavailable_decomposition_count,
    )
    print("Recovered without regeneration:", report.recovered_without_regeneration_count)
    print(
        "Regeneration fallback required:",
        report.regeneration_fallback_required_count, "/",
        report.hypothesis_count,
    )
    print("Source-alignment audit LLM calls:", report.source_alignment_audit_llm_calls)
    print()

    for row in report.hypotheses:
        print(
            row.final_hypothesis_id, "|", row.route_action, "|",
            row.primary_status, "| fallback=", row.regeneration_fallback_required,
        )
        print("  primary Gate ready:", row.primary_gate_ready_claim_ids)
        print("  primary Gate novelty-ready:", row.primary_gate_novelty_bearing_ready_claim_ids)
        for result in row.source_alignment_results:
            print(
                "  ALIGN", result.claim_id, "|", result.status,
                "| candidates=", result.candidate_count,
                "| audit=", result.semantic_audit_passed,
            )
        for result in row.decomposition_results:
            print(
                "  DECOMPOSE", result.claim_id, "|", result.status,
                "| components=", result.component_claim_ids,
            )
            if result.reason_codes:
                print("    reasons:", result.reason_codes)

    print()
    print("Regeneration performed: false")
    print("Endpoint binding performed: false")
    print("Verifier observed: false")
    print("Primary binding plan:", plan_path)
    print("Primary Gate v2:", gate_path)
    print("Output:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
