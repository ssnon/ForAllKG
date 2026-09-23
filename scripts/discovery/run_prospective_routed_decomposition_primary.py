from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_routed_decomposition_primary import (
    execute_decomposition_primary,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    ProspectiveRoutedExecutionPlan,
)
from pipeline_core.discovery.prospective_routed_route_compiler import (
    ProspectiveRoutedDispatchReport,
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
            "Execute the frozen deterministic atomic-decomposition primary "
            "for a routed prospective case. This stage does not execute "
            "source alignment, specification repair, regeneration, endpoint "
            "binding, retrieval, or verifier logic."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--binding-plan", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    execution = ProspectiveRoutedExecutionPlan.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    cases = [
        row
        for row in execution.cases
        if row.case_id == args.case_id
    ]
    if len(cases) != 1:
        raise ValueError("case is not present exactly once in execution plan")
    case = cases[0]

    binding_plan = RelationalAtomicBindingPlan.model_validate_json(
        args.binding_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    dispatch = ProspectiveRoutedDispatchReport.model_validate_json(
        args.dispatch.expanduser().resolve().read_text(encoding="utf-8")
    )
    if dispatch.case_id != case.case_id:
        raise ValueError("dispatch/execution-plan case mismatch")

    post_plan_path = Path(case.post_route_binding_plan_path).expanduser().resolve()
    post_gate_path = Path(case.post_route_gate_path).expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    for path in (post_plan_path, post_gate_path, output_path):
        if path.exists():
            raise ValueError(
                "routed decomposition primary outputs are write-once: "
                + str(path)
            )

    post_plan, post_gate, report = execute_decomposition_primary(
        binding_plan=binding_plan,
        dispatch=dispatch,
    )

    post_plan_path.parent.mkdir(parents=True, exist_ok=True)
    post_gate_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_json_exclusive(post_plan_path, post_plan)
    write_json_exclusive(post_gate_path, post_gate)
    write_json_exclusive(output_path, report)

    print("Prospective routed decomposition primary complete")
    print("Case:", report.case_id)
    print("Report:", report.report_id)
    print(
        "Primary materialized/unavailable:",
        report.primary_materialized_count,
        "/",
        report.primary_unavailable_count,
    )
    print(
        "Deterministic decomposition available/unavailable:",
        report.deterministic_decomposition_count,
        "/",
        report.unavailable_decomposition_count,
    )
    print(
        "Regeneration fallback required:",
        report.regeneration_fallback_required_count,
        "/",
        report.hypothesis_count,
    )
    print()

    for row in report.hypotheses:
        print(
            row.final_hypothesis_id,
            "|",
            row.primary_status,
            "| fallback=",
            row.regeneration_fallback_required,
        )
        print("  source claims:", row.source_claim_ids)
        print("  output claims:", row.output_claim_ids)
        print(
            "  post-gate ready:",
            row.post_gate_ready_claim_ids,
        )
        print(
            "  post-gate novelty-ready:",
            row.post_gate_novelty_bearing_ready_claim_ids,
        )
        for item in row.decompositions:
            print(
                "  ",
                item.composite_claim_id,
                "|",
                item.status,
                "| components=",
                item.component_claim_ids,
            )
            if item.reason_codes:
                print("    reasons:", item.reason_codes)

    print()
    print("Decomposition LLM calls: 0")
    print("Source alignment performed: false")
    print("Regeneration performed: false")
    print("Endpoint binding performed: false")
    print("Post-route plan:", post_plan_path)
    print("Post-route gate:", post_gate_path)
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
