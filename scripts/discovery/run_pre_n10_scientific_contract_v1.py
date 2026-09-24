from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pipeline_core.discovery.external_novelty_llm import InstructorOpenAICompatibleExternalNoveltyBackend
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.novelty_claim_decomposition import LiteratureQueryPlanner, NoveltyClaimDecomposer
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import build_pre_n10_scientific_contract_v1
from pipeline_core.discovery.relational_scientific_verifier_shadow import write_json_exclusive


def _write_json_exclusive(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise ValueError("output is write-once: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile Alpha4 hypotheses into canonical novelty claims and run the strict "
            "scientific contract gate before retrieval, N9, or N10. This stage performs "
            "claim decomposition only and has no novelty or production-selection authority."
        )
    )
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-claims", type=int, default=4)
    parser.add_argument("--max-queries-per-claim", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=1)
    parser.add_argument("--query-plan-output", required=True, type=Path)
    parser.add_argument("--contract-output", required=True, type=Path)
    parser.add_argument("--prompt-output", type=Path, default=None)
    parser.add_argument("--specification-audit-output", type=Path, default=None)
    args = parser.parse_args()

    portfolio_path = args.portfolio.expanduser().resolve()
    query_plan_output = args.query_plan_output.expanduser().resolve()
    contract_output = args.contract_output.expanduser().resolve()
    for path, label in [
        (query_plan_output, "query-plan"),
        (contract_output, "contract"),
    ]:
        if path.exists():
            raise ValueError(label + " output is write-once")
    if args.prompt_output and args.prompt_output.expanduser().resolve().exists():
        raise ValueError("prompt output is write-once")
    if args.specification_audit_output and args.specification_audit_output.expanduser().resolve().exists():
        raise ValueError("specification audit output is write-once")

    portfolio = HypothesisPortfolio.model_validate_json(portfolio_path.read_text(encoding="utf-8"))
    backend = InstructorOpenAICompatibleExternalNoveltyBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        parse_retries=args.parse_retries,
        capture_prompts=bool(args.prompt_output),
    )
    decomposer = NoveltyClaimDecomposer(
        backend,
        max_claims_per_hypothesis=args.max_claims,
        max_queries_per_claim=args.max_queries_per_claim,
    )
    decompositions = [decomposer.decompose(row) for row in portfolio.hypotheses]
    query_plan = LiteratureQueryPlanner().build(portfolio, decompositions)
    query_plan_output.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(query_plan_output, query_plan)

    if args.prompt_output:
        _write_json_exclusive(
            args.prompt_output,
            {
                "schema_version": "pre-n10-claim-decomposition-prompts-v1",
                "source_portfolio_id": portfolio.portfolio_id,
                "model": args.model,
                "records": [asdict(row) for row in backend.prompt_records],
            },
        )
    if args.specification_audit_output:
        _write_json_exclusive(
            args.specification_audit_output,
            {
                "schema_version": "pre-n10-specification-sanitization-audit-v1",
                "source_portfolio_id": portfolio.portfolio_id,
                "diagnostic_only": True,
                "production_authority": False,
                "records": list(decomposer.specification_sanitization_records),
            },
        )

    report = build_pre_n10_scientific_contract_v1(
        portfolio_path=portfolio_path,
        query_plan_path=query_plan_output,
        claim_decomposition_request_count=len(portfolio.hypotheses),
    )
    contract_output.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(contract_output, report)

    print("Pre-N10 scientific contract compilation complete")
    print("Report:", report.report_id)
    print("Hypotheses ready/intervention:", report.ready_hypothesis_count, "/", report.intervention_required_hypothesis_count)
    print("Claims ready/not-ready/novelty-bearing-ready:", report.ready_claim_count, "/", report.not_ready_claim_count, "/", report.novelty_bearing_ready_claim_count)
    print("Router hints:", report.router_hint_counts)
    print("Disposition:", report.disposition)
    for hypothesis in report.hypotheses:
        print(
            hypothesis.hypothesis_id,
            "|",
            hypothesis.contract_status,
            "| ready=",
            hypothesis.ready_claim_count,
            "/",
            hypothesis.claim_count,
            "| novelty-ready=",
            hypothesis.novelty_bearing_ready_claim_count,
        )
        for claim in hypothesis.claims:
            if claim.contract_status == "READY_FOR_N10_CONTRACT":
                continue
            print(
                " ",
                claim.claim_id,
                "|",
                claim.router_hint,
                "| binding=",
                claim.binding_contract_reason_codes,
                "| source=",
                claim.source_contract_reason_codes,
            )
    print("Claim-decomposition requests:", report.claim_decomposition_request_count)
    print("Retrieval performed: false")
    print("Novelty assessment performed: false")
    print("N9 performed: false")
    print("N10 performed: false")
    print("Repair performed: false")
    print("Regeneration performed: false")
    print("Production selection changed: false")
    print("Query plan:", query_plan_output)
    print("Contract:", contract_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
