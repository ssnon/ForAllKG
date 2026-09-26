from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pipeline_core.discovery.atomic_scientific_source_provenance import (
    build_atomic_scientific_source_binding_bundle,
)
from pipeline_core.discovery.external_novelty_llm import InstructorOpenAICompatibleExternalNoveltyBackend
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.novelty_claim_decomposition import LiteratureQueryPlanner, NoveltyClaimDecomposer
from pipeline_core.discovery.pre_n10_canonical_source_reference_v1 import (
    build_pre_n10_canonical_source_reference_report_v1,
)
from pipeline_core.discovery.pre_n10_scientific_contract_v1 import build_pre_n10_scientific_contract_v1
from pipeline_core.discovery.pre_n10_scientific_contract_v2 import (
    build_pre_n10_scientific_contract_v2,
)
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
    parser.add_argument("--source-binding-bundle-output", type=Path, default=None)
    parser.add_argument("--canonical-source-reference-output", type=Path, default=None)
    parser.add_argument("--contract-v2-output", type=Path, default=None)
    parser.add_argument("--prompt-output", type=Path, default=None)
    parser.add_argument("--specification-audit-output", type=Path, default=None)
    args = parser.parse_args()

    portfolio_path = args.portfolio.expanduser().resolve()
    query_plan_output = args.query_plan_output.expanduser().resolve()
    contract_output = args.contract_output.expanduser().resolve()

    canonical_outputs = [
        args.source_binding_bundle_output,
        args.canonical_source_reference_output,
        args.contract_v2_output,
    ]
    canonical_output_count = sum(value is not None for value in canonical_outputs)
    if canonical_output_count not in {0, 3}:
        raise ValueError(
            "canonical pre-N10 sidecar outputs must be supplied all-or-none"
        )

    source_binding_bundle_output = (
        args.source_binding_bundle_output.expanduser().resolve()
        if args.source_binding_bundle_output is not None
        else None
    )
    canonical_source_reference_output = (
        args.canonical_source_reference_output.expanduser().resolve()
        if args.canonical_source_reference_output is not None
        else None
    )
    contract_v2_output = (
        args.contract_v2_output.expanduser().resolve()
        if args.contract_v2_output is not None
        else None
    )

    output_paths = [
        (query_plan_output, "query-plan"),
        (contract_output, "contract"),
    ]
    if source_binding_bundle_output is not None:
        output_paths.extend(
            [
                (source_binding_bundle_output, "source-binding bundle"),
                (
                    canonical_source_reference_output,
                    "canonical source-reference report",
                ),
                (contract_v2_output, "contract-v2"),
            ]
        )

    for path, label in output_paths:
        assert path is not None
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

    canonical_report = None
    contract_v2 = None
    if source_binding_bundle_output is not None:
        assert canonical_source_reference_output is not None
        assert contract_v2_output is not None

        source_binding_bundle = (
            build_atomic_scientific_source_binding_bundle(
                source_portfolio_id=portfolio.portfolio_id,
                query_plan=query_plan,
                records=list(decomposer.atomic_source_binding_records),
            )
        )
        write_json_exclusive(
            source_binding_bundle_output,
            source_binding_bundle,
        )

        canonical_report = (
            build_pre_n10_canonical_source_reference_report_v1(
                portfolio_path=portfolio_path,
                query_plan_path=query_plan_output,
                source_binding_bundle_path=source_binding_bundle_output,
            )
        )
        write_json_exclusive(
            canonical_source_reference_output,
            canonical_report,
        )

        contract_v2 = build_pre_n10_scientific_contract_v2(
            portfolio_path=portfolio_path,
            query_plan_path=query_plan_output,
            canonical_source_reference_path=(
                canonical_source_reference_output
            ),
            claim_decomposition_request_count=len(portfolio.hypotheses),
        )
        write_json_exclusive(contract_v2_output, contract_v2)

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
    if source_binding_bundle_output is not None:
        assert canonical_source_reference_output is not None
        assert contract_v2_output is not None
        assert canonical_report is not None
        assert contract_v2 is not None
        print("Source-binding bundle:", source_binding_bundle_output)
        print(
            "Canonical source-reference report:",
            canonical_source_reference_output,
        )
        print("Stable-ID contract V2:", contract_v2_output)
        print(
            "Canonical source-reference disagreements:",
            canonical_report.disagreement_count,
        )
        print("V2 disposition:", contract_v2.disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
