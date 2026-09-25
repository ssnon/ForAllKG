from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_core.discovery.pre_n10_primary_router_runtime_v1 import (
    execute_pre_n10_primary_router_runtime_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the unified fail-closed pre-N10 primary router. "
            "At most one homogeneous primary intervention is allowed per "
            "hypothesis; mixed intervention populations go directly to the "
            "one-shot regeneration fallback without chained primary repairs."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--query-plan", required=True, type=Path)
    parser.add_argument("--contract-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--model", required=True)
    parser.add_argument("--specification-repair-model", default=None)
    parser.add_argument("--specification-audit-model", default=None)
    parser.add_argument("--source-alignment-model", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main() -> int:
    args = _parser().parse_args()

    routed, post_contract, report = (
        execute_pre_n10_primary_router_runtime_v1(
            portfolio_path=args.portfolio,
            query_plan_path=args.query_plan,
            contract_report_path=args.contract_report,
            output_root=args.output_dir,
            model=args.model,
            specification_repair_model=args.specification_repair_model,
            specification_audit_model=args.specification_audit_model,
            source_alignment_model=args.source_alignment_model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=args.temperature,
            parse_retries=args.parse_retries,
            timeout_seconds=args.timeout_seconds,
        )
    )

    print("Pre-N10 unified primary router complete")
    print("Report:", report.report_id)
    print("Post query plan:", routed.plan_id)
    print("Post contract:", post_contract.report_id)
    print("Routes:", report.route_counts)
    print("Primary interventions:", report.primary_intervention_count)
    print("Recovered for N10:", report.recovered_for_n10_count)
    print(
        "Regeneration fallback:",
        report.regeneration_fallback_required_count,
    )
    print("Mixed-route fallback:", report.mixed_route_fallback_count)
    print("Retrieval performed: false")
    print("N9 performed: false")
    print("N10 performed: false")
    print("Regeneration performed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
