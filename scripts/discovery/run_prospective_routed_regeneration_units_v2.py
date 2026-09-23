from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
)
from pipeline_core.discovery.prospective_routed_execution_plan_v2 import (
    ProspectiveRoutedExecutionPlanV2,
)
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    ProspectiveRoutedPrimaryMaterializationReportV2,
)
from pipeline_core.discovery.prospective_routed_regeneration_executor_v2 import (
    execute_routed_regeneration_units_v2,
)
from pipeline_core.discovery.prospective_routed_route_compiler_v2 import (
    ProspectiveRoutedDispatchReportV2,
)
from pipeline_core.discovery.relational_scientific_verifier_shadow import (
    write_json_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute only the frozen regeneration-unit-v2 fallback for "
            "hypotheses marked fallback-required after routed primary-v2. "
            "Each lineage receives exactly one structured hypothesis "
            "generation call and zero repair calls. No downstream evaluation "
            "is performed here."
        )
    )
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument(
        "--regeneration-unit-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--primary-report", required=True, type=Path)
    args = parser.parse_args()

    execution = ProspectiveRoutedExecutionPlanV2.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    unit_freeze = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        args.regeneration_unit_freeze.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )

    if execution.source_regeneration_unit_freeze_id != unit_freeze.freeze_id:
        raise ValueError("execution-plan/regeneration-unit ID mismatch")
    if (
        execution.source_regeneration_unit_freeze_sha256
        != unit_freeze.freeze_sha256
    ):
        raise ValueError("execution-plan/regeneration-unit SHA mismatch")

    cases = [
        row for row in execution.cases if row.case_id == args.case_id
    ]
    if len(cases) != 1:
        raise ValueError("case is not present exactly once in execution plan")
    case = cases[0]

    dispatch_path = args.dispatch.expanduser().resolve()
    if dispatch_path != Path(case.routed_dispatch_path).expanduser().resolve():
        raise ValueError("dispatch path differs from frozen execution plan")

    dispatch = ProspectiveRoutedDispatchReportV2.model_validate_json(
        dispatch_path.read_text(encoding="utf-8")
    )
    primary = ProspectiveRoutedPrimaryMaterializationReportV2.model_validate_json(
        args.primary_report.expanduser().resolve().read_text(encoding="utf-8")
    )
    if dispatch.case_id != case.case_id or primary.case_id != case.case_id:
        raise ValueError("case lineage mismatch")

    context_path = Path(case.hypothesis_context_path).expanduser().resolve()
    if not context_path.is_file():
        raise ValueError("frozen hypothesis context missing: " + str(context_path))
    context = HypothesisContext.model_validate_json(
        context_path.read_text(encoding="utf-8")
    )

    # Parse retry is mechanical structured-output recovery, not an additional
    # scientific generation budget. One backend.generate invocation remains
    # authoritative under regeneration-unit-v2.
    mechanical_parse_retries = 1

    def backend_factory(final_hypothesis_id, lineage):
        expected_context = Path(
            lineage.input_hypothesis_context_path
        ).expanduser().resolve()
        if expected_context != context_path:
            raise ValueError(
                final_hypothesis_id
                + ": regeneration lineage/context path mismatch"
            )
        telemetry = (
            Path(lineage.lineage_dir)
            / "regeneration_unit_v2.telemetry.jsonl"
        )
        return InstructorOpenAICompatibleHypothesisBackend(
            model=case.regeneration_model,
            api_key_env=execution.settings.api_key_env,
            base_url=execution.settings.base_url,
            instructor_mode="JSON",
            temperature=0.0,
            parse_retries=mechanical_parse_retries,
            timeout=execution.settings.route_timeout_seconds,
            telemetry_path=telemetry,
            telemetry_context={
                "case_id": case.case_id,
                "final_hypothesis_id": final_hypothesis_id,
                "execution_plan_id": execution.plan_id,
                "regeneration_unit_freeze_id": unit_freeze.freeze_id,
            },
        )

    report, raw_results = execute_routed_regeneration_units_v2(
        context=context,
        dispatch=dispatch,
        primary=primary,
        unit_freeze=unit_freeze,
        backend_factory=backend_factory,
    )

    for row in report.hypotheses:
        lineage = row.regeneration_lineage
        result = raw_results[row.final_hypothesis_id]

        result_path = Path(lineage.unit_result_path)
        portfolio_path = Path(lineage.regenerated_portfolio_path)
        for path in (result_path, portfolio_path):
            if path.exists():
                raise ValueError(
                    "regeneration-unit-v2 output is write-once: " + str(path)
                )

        result_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_exclusive(result_path, result)
        if result.compiled_portfolio is not None:
            write_json_exclusive(
                portfolio_path,
                result.compiled_portfolio,
            )

    routed = dispatch_path.parent
    report_path = routed / "regeneration_unit_v2.execution.json"
    if report_path.exists():
        raise ValueError(
            "regeneration-unit-v2 execution report is write-once"
        )
    write_json_exclusive(report_path, report)

    print("Prospective routed regeneration-unit-v2 complete")
    print("Case:", report.case_id)
    print("Report:", report.report_id)
    print("Fallback hypotheses:", report.regeneration_required_count)
    print("Generation calls:", report.generation_call_count)
    print("Repair calls:", report.repair_call_count)
    print(
        "Generated/abstained/compile-rejected/call-failed:",
        report.generated_and_compiled_count,
        "/",
        report.generated_abstention_count,
        "/",
        report.compile_rejected_count,
        "/",
        report.generation_failed_count,
    )
    print()
    for row in report.hypotheses:
        print(
            row.final_hypothesis_id,
            "|",
            row.unit_status,
            "| regenerated hypotheses=",
            row.regenerated_hypothesis_count,
        )
        print(
            "  result:",
            row.regeneration_lineage.unit_result_path,
        )
        if row.regenerated_portfolio_written:
            print(
                "  portfolio:",
                row.regeneration_lineage.regenerated_portfolio_path,
            )
    print()
    print("Full E2E rerun performed: false")
    print("Retrieval performed: false")
    print("Semantic critic performed: false")
    print("N9/N10 performed: false")
    print("Endpoint/verifier performed: false")
    print("Previous hypothesis text consumed: false")
    print("Output:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
