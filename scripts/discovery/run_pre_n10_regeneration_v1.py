from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
from pipeline_core.discovery.hypothesis_llm import (
    InstructorOpenAICompatibleHypothesisBackend,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    execute_pre_n10_regeneration_v1,
)
from pipeline_core.discovery.pre_n10_source_alignment_primary_v1 import (
    PreN10SourceAlignmentPrimaryReportV1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    ProspectiveRegenerationUnitV2Freeze,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one-shot fresh regeneration for pre-N10 contract "
            "lineages that remain unrecovered after the bounded primary. "
            "The generation unit consumes only the frozen HypothesisContext "
            "and performs no semantic critic, claim decomposition, retrieval, "
            "novelty assessment, N9, or N10."
        )
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--primary-report", required=True, type=Path)
    parser.add_argument(
        "--regeneration-unit-freeze",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    context = HypothesisContext.model_validate_json(
        args.context.expanduser().resolve().read_text(encoding="utf-8")
    )
    primary = PreN10SourceAlignmentPrimaryReportV1.model_validate_json(
        args.primary_report.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    freeze = ProspectiveRegenerationUnitV2Freeze.model_validate_json(
        args.regeneration_unit_freeze.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )

    root = args.output_dir.expanduser().resolve()

    def backend_factory(source_hypothesis_id: str, lineage_dir: Path):
        return InstructorOpenAICompatibleHypothesisBackend(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            instructor_mode="JSON",
            temperature=0.0,
            parse_retries=args.parse_retries,
            timeout=args.timeout_seconds,
            telemetry_path=(
                lineage_dir / "regeneration.telemetry.jsonl"
            ),
            telemetry_context={
                "pipeline": "pre_n10_regeneration_v1",
                "source_hypothesis_id": source_hypothesis_id,
                "source_primary_report_id": primary.report_id,
                "regeneration_unit_freeze_id": freeze.freeze_id,
            },
        )

    report, _raw = execute_pre_n10_regeneration_v1(
        context=context,
        primary_report=primary,
        unit_freeze=freeze,
        backend_factory=backend_factory,
        output_root=root,
    )

    print("Pre-N10 regeneration-v1 complete")
    print("Report:", report.report_id)
    print("Fallback lineages:", report.regeneration_required_count)
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
    for row in report.lineages:
        print(
            row.source_hypothesis_id,
            "|",
            row.unit_status,
            "| regenerated hypotheses=",
            row.regenerated_hypothesis_count,
        )
        print("  result:", row.unit_result_path)
        if row.regenerated_portfolio_path:
            print("  portfolio:", row.regenerated_portfolio_path)

    print()
    print("Full E2E rerun performed: false")
    print("Semantic critic performed: false")
    print("Claim decomposition performed: false")
    print("Pre-N10 contract re-entry performed: false")
    print("Retrieval performed: false")
    print("N9/N10 performed: false")
    print("Previous hypothesis text consumed: false")
    print("Novelty outcome consumed: false")
    print("Verifier outcome consumed: false")
    print("Output:", root / "regeneration.execution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
