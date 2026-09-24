from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.pre_n10_scientific_contract_v1 import (
    PreN10ScientificContractReportV1,
)
from pipeline_core.discovery.pre_n10_source_alignment_primary_v1 import (
    execute_pre_n10_source_alignment_primary_v1,
)
from pipeline_core.discovery.prospective_routed_primary_materializer_v2 import (
    InstructorSourceAlignmentAuditBackend,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the bounded pre-N10 zero-scientific-delta source "
            "alignment primary. This stage may rebind only to existing "
            "HypothesisCard prediction/falsifier surfaces and performs no "
            "retrieval, novelty assessment, N9, N10, or regeneration."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--query-plan", required=True, type=Path)
    parser.add_argument("--contract-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    contract = PreN10ScientificContractReportV1.model_validate_json(
        args.contract_report.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    root = args.output_dir.expanduser().resolve()
    backend = InstructorSourceAlignmentAuditBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=0.0,
        parse_retries=args.parse_retries,
        timeout_seconds=args.timeout_seconds,
        telemetry_path=root / "source_alignment.telemetry.jsonl",
        telemetry_context={
            "pipeline": "pre_n10_source_alignment_primary_v1",
        },
    )

    _plan, post_contract, report = (
        execute_pre_n10_source_alignment_primary_v1(
            portfolio_path=args.portfolio,
            query_plan_path=args.query_plan,
            contract_report=contract,
            audit_backend=backend,
            output_root=root,
        )
    )

    print("Pre-N10 source-alignment primary complete")
    print("Report:", report.report_id)
    print(
        "Alignments materialized/unavailable/ambiguous/"
        "semantic-reject/audit-fail:",
        report.materialized_alignment_count,
        "/",
        report.unavailable_alignment_count,
        "/",
        report.ambiguous_alignment_count,
        "/",
        report.semantic_reject_count,
        "/",
        report.audit_failure_count,
    )
    print(
        "Recovered for N10 / regeneration fallback:",
        report.recovered_for_n10_count,
        "/",
        report.regeneration_fallback_required_count,
    )
    print("Source-alignment audit LLM calls:", report.audit_llm_call_count)

    for row in report.hypotheses:
        print(
            row.hypothesis_id,
            "|",
            row.source_contract_status,
            "->",
            row.post_contract_status,
            "| recovered=",
            row.recovered_for_n10,
            "| fallback=",
            row.regeneration_fallback_required,
        )
        for claim in row.source_alignment_results:
            print(
                " ",
                claim.claim_id,
                "|",
                claim.status,
                "| candidates=",
                claim.candidate_count,
            )

    print("Post-primary disposition:", post_contract.disposition)
    print("Retrieval performed: false")
    print("Novelty assessment performed: false")
    print("N9 performed: false")
    print("N10 performed: false")
    print("Regeneration performed: false")
    print("Output dir:", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
