from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.prospective_routed_decomposition_primary import (
    ProspectiveRoutedDecompositionPrimaryReport,
)
from pipeline_core.discovery.prospective_routed_execution_plan import (
    ProspectiveRoutedExecutionPlan,
)
from pipeline_core.discovery.prospective_routed_regeneration_budget_accounting import (
    build_regeneration_budget_accounting,
)
from pipeline_core.discovery.prospective_routed_route_compiler import (
    ProspectiveRoutedDispatchReport,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze read-only accounting for routed regeneration fallback "
            "budget compatibility. This stage performs no LLM call and never "
            "executes a compiled regeneration command."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--decomposition-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(
            "regeneration budget accounting output is write-once"
        )

    plan = ProspectiveRoutedExecutionPlan.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    dispatch = ProspectiveRoutedDispatchReport.model_validate_json(
        args.dispatch.expanduser().resolve().read_text(encoding="utf-8")
    )
    decomposition = (
        ProspectiveRoutedDecompositionPrimaryReport.model_validate_json(
            args.decomposition_report.expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
    )

    report = build_regeneration_budget_accounting(
        plan=plan,
        dispatch=dispatch,
        decomposition=decomposition,
    )
    write_json_exclusive(output, report)

    print("Prospective routed regeneration budget accounting frozen")
    print("Case:", report.case_id)
    print("Report:", report.report_id)
    print("Regeneration required:", report.regeneration_required_count)
    print("Frozen budget conflicts:", report.frozen_budget_conflict_count)
    print("Frozen executable:", report.frozen_executable_count)
    print("Dispositions:", report.disposition_counts)
    print()

    for row in report.hypotheses:
        print(
            row.final_hypothesis_id,
            "|",
            row.disposition,
            "| required=",
            row.regeneration_fallback_required,
            "| full-e2e=",
            row.compiled_regeneration_is_full_e2e,
        )
        print(
            "  frozen attempts/LLM max:",
            row.frozen_fallback_max_attempts,
            "/",
            row.frozen_fallback_llm_calls_per_hypothesis_max,
        )
        if row.reason_codes:
            print("  reasons:", row.reason_codes)

    print()
    print("Regeneration executed: false")
    print("LLM calls performed: 0")
    print("Execution plan modified: false")
    print("Dispatch modified: false")
    print("Later-case settings adapted: false")
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
