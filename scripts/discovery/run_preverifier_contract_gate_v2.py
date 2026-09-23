from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.preverifier_contract_gate_v2 import (
    build_preverifier_contract_gate_v2,
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
            "Run the deterministic Pre-Verifier Contract Gate v2 over a "
            "relational atomic binding plan before endpoint binding. "
            "The gate performs no repair, regeneration, retrieval, endpoint "
            "LLM call, or novelty assessment."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not plan_path.is_file():
        raise ValueError("missing binding plan: " + str(plan_path))
    if output_path.exists():
        raise ValueError(
            "pre-verifier contract gate v2 output is write-once"
        )

    plan = RelationalAtomicBindingPlan.model_validate_json(
        plan_path.read_text(encoding="utf-8")
    )
    report = build_preverifier_contract_gate_v2(plan=plan)
    write_json_exclusive(output_path, report)

    print("Pre-Verifier Contract Gate v2 complete")
    print("Report:", report.report_id)
    print("Claims:", report.claim_count)
    print("Ready:", report.ready_claim_count)
    print("Not ready:", report.not_ready_claim_count)
    print(
        "Novelty-bearing ready:",
        report.novelty_bearing_ready_claim_count,
    )
    print("Router hints:", report.router_hint_counts)
    print()
    for row in report.rows:
        print(
            row.final_hypothesis_id,
            row.claim_id,
            "|",
            row.gate_status,
            "|",
            row.router_hint,
        )
        if row.binding_contract_reason_codes:
            print(
                "  binding:",
                row.binding_contract_reason_codes,
            )
        if row.source_contract_reason_codes:
            print(
                "  source:",
                row.source_contract_reason_codes,
            )
        print(
            "  exact prediction/falsifier:",
            row.prediction_exact_source_binding_count,
            "/",
            row.falsifier_exact_source_binding_count,
            "| shared observable:",
            row.shared_observable_identity_satisfied,
        )

    print()
    print("Endpoint binding performed: false")
    print("Repair performed: false")
    print("Regeneration performed: false")
    print("Novelty assessment performed: false")
    print("Output:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
