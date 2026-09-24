from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_semantic_llm import InstructorOpenAICompatibleSemanticCriticBackend
from pipeline_core.discovery.prospective_regeneration_downstream_semantic_v2 import execute_regeneration_downstream_semantic_v2
from pipeline_core.discovery.prospective_regeneration_downstream_v2 import ProspectiveRegenerationDownstreamV2Freeze
from pipeline_core.discovery.prospective_routed_execution_plan_v2 import ProspectiveRoutedExecutionPlanV2
from pipeline_core.discovery.prospective_routed_regeneration_executor_v2 import ProspectiveRoutedRegenerationExecutionReportV2
from pipeline_core.discovery.relational_scientific_verifier_shadow import write_json_exclusive


def _write_json_exclusive(path: Path, value: object) -> None:
    if path.exists():
        raise ValueError("write-once artifact already exists: " + str(path))
    write_json_exclusive(path, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--downstream-freeze", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--regeneration-execution", required=True, type=Path)
    args = parser.parse_args()

    execution = ProspectiveRoutedExecutionPlanV2.model_validate_json(
        args.execution_plan.expanduser().resolve().read_text(encoding="utf-8")
    )
    downstream = ProspectiveRegenerationDownstreamV2Freeze.model_validate_json(
        args.downstream_freeze.expanduser().resolve().read_text(encoding="utf-8")
    )
    if execution.source_regeneration_downstream_freeze_id != downstream.freeze_id:
        raise ValueError("execution-plan/downstream-freeze ID mismatch")
    if execution.source_regeneration_downstream_freeze_sha256 != downstream.freeze_sha256:
        raise ValueError("execution-plan/downstream-freeze SHA mismatch")

    cases = [row for row in execution.cases if row.case_id == args.case_id]
    if len(cases) != 1:
        raise ValueError("case is not present exactly once in execution plan")
    case = cases[0]

    regeneration_execution = ProspectiveRoutedRegenerationExecutionReportV2.model_validate_json(
        args.regeneration_execution.expanduser().resolve().read_text(encoding="utf-8")
    )
    if regeneration_execution.case_id != case.case_id:
        raise ValueError("regeneration execution/case mismatch")

    context_path = Path(case.hypothesis_context_path).expanduser().resolve()
    context = HypothesisContext.model_validate_json(context_path.read_text(encoding="utf-8"))

    def backend_factory(final_hypothesis_id, semantic_prefix):
        return InstructorOpenAICompatibleSemanticCriticBackend(
            model=case.critic_model,
            api_key_env=execution.settings.api_key_env,
            base_url=execution.settings.base_url,
            instructor_mode="JSON",
            temperature=0.0,
            parse_retries=downstream.policy.budget.provider_parse_retries_max,
            timeout=execution.settings.route_timeout_seconds,
            telemetry_path=Path(str(semantic_prefix) + ".telemetry.jsonl"),
            telemetry_context={
                "case_id": case.case_id,
                "source_final_hypothesis_id": final_hypothesis_id,
                "execution_plan_id": execution.plan_id,
                "downstream_freeze_id": downstream.freeze_id,
            },
        )

    report, outcomes = execute_regeneration_downstream_semantic_v2(
        context=context,
        regeneration_execution=regeneration_execution,
        downstream_freeze=downstream,
        backend_factory=backend_factory,
    )

    for row in report.lineages:
        outcome = outcomes.get(row.source_final_hypothesis_id)
        if outcome is None:
            continue
        prefix = Path(row.semantic_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        _write_json_exclusive(Path(str(prefix) + ".hard_evaluation.json"), outcome.evaluation)
        _write_json_exclusive(Path(str(prefix) + ".run.json"), outcome.run_record)
        _write_json_exclusive(Path(str(prefix) + ".reference_audit.json"), outcome.reference_audit)
        if outcome.generation is not None:
            _write_json_exclusive(Path(str(prefix) + ".draft.json"), outcome.generation.draft)
        if outcome.sanitized_draft is not None:
            _write_json_exclusive(Path(str(prefix) + ".sanitized.draft.json"), outcome.sanitized_draft)
        if outcome.review is not None:
            _write_json_exclusive(Path(str(prefix) + ".review.json"), outcome.review)

    routed = Path(case.routed_dispatch_path).expanduser().resolve().parent
    report_path = routed / "regeneration_downstream_v2.semantic.json"
    _write_json_exclusive(report_path, report)

    print("Regeneration downstream-v2 semantic checkpoint complete")
    print("Case:", report.case_id)
    print("Report:", report.report_id)
    print("Lineages:", report.lineage_count)
    print(
        "Accepted/hard-gate-failed/review-rejected/stage-failed:",
        report.semantic_accepted_count, "/",
        report.hard_gate_failed_count, "/",
        report.semantic_review_rejected_count, "/",
        report.semantic_stage_failed_count,
    )
    print(
        "Eligible for external novelty:",
        report.eligible_for_external_novelty_count, "/",
        report.lineage_count,
    )
    print()
    for row in report.lineages:
        print(
            row.source_final_hypothesis_id,
            "|", row.status,
            "| hard-gate=", row.hard_gate_passed,
            "| generated=", row.semantic_generation_performed,
            "| accepted=", row.semantic_review_accepted,
            "| external-next=", row.eligible_for_external_novelty,
        )
        if row.failure_type:
            print("  failure:", row.failure_type, row.failure_message)

    print()
    print("External novelty performed: false")
    print("N9/N10 performed: false")
    print("Binding plan/Gate v2 performed: false")
    print("Endpoint/verifier performed: false")
    print("Downstream retry performed: false")
    print("Second regeneration performed: false")
    print("Output:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
