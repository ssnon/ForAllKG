from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.registry import get_domain_profile
from pipeline_core.discovery.grounded_factor_projection import (
    GroundedFactorProjectionReport,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    load_literature_provider_plan,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)
from pipeline_core.discovery.prior_art_retrieval import LiteratureRetriever
from pipeline_core.discovery.scientific_relation_ir import (
    ScientificRelationIRReport,
)
from pipeline_core.discovery.scientific_relation_second_pass import (
    InstructorSemanticExpansionBackend,
    ScientificRelationSecondPassReport,
    build_resolution_targets,
    build_resolution_transport_plan,
    build_second_pass_plan,
    build_second_pass_transport_plan,
    merge_second_pass_resolution_packets,
    run_semantic_expansion,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a retrieval-only semantic second pass over typed scientific "
            "relations, then perform bounded DOI/title metadata resolution. "
            "No retrieved result becomes scientific or novelty authority."
        )
    )
    parser.add_argument("--relation-ir", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--source-portfolio-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--providers", default="auto")
    parser.add_argument("--provider-plan", default=None, type=Path)
    parser.add_argument("--results-per-query", type=int, default=16)
    parser.add_argument("--max-queries-per-claim", type=int, default=8)
    parser.add_argument("--max-resolution-queries", type=int, default=24)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--save-prompts", action="store_true")
    args = parser.parse_args()

    relation_ir = ScientificRelationIRReport.model_validate_json(
        args.relation_ir.read_text(encoding="utf-8")
    )
    projection = GroundedFactorProjectionReport.model_validate_json(
        args.projection.read_text(encoding="utf-8")
    )
    domain_profile = get_domain_profile(args.domain_profile)

    backend = InstructorSemanticExpansionBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        capture_prompts=args.save_prompts,
    )

    drafts = run_semantic_expansion(
        relation_ir_report=relation_ir,
        domain_profile=domain_profile,
        backend=backend,
    )

    plan = build_second_pass_plan(
        relation_ir_report=relation_ir,
        projection_report=projection,
        drafts_by_claim_id=drafts,
        domain_profile=domain_profile,
        source_portfolio_id=args.source_portfolio_id,
        max_queries_per_claim=args.max_queries_per_claim,
    )
    transport = build_second_pass_transport_plan(plan)

    prefix = args.output_prefix
    draft_path = prefix.with_suffix(".semantic_expansions.json")
    plan_path = prefix.with_suffix(".semantic_plan.json")
    transport_path = prefix.with_suffix(".transport_query_plan.json")
    provider_path = prefix.with_suffix(".provider_plan.json")
    delta_path = prefix.with_suffix(".delta_prior_art.json")
    resolution_plan_path = prefix.with_suffix(".resolution_query_plan.json")
    resolution_packet_path = prefix.with_suffix(".resolution_prior_art.json")
    resolved_path = prefix.with_suffix(".resolved_prior_art.json")
    report_path = prefix.with_suffix(".report.json")
    prompt_path = prefix.with_suffix(".prompts.json")

    _write(
        draft_path,
        {
            claim_id: draft.model_dump(mode="json")
            for claim_id, draft in drafts.items()
        },
    )
    _write(plan_path, plan)
    _write(transport_path, transport)

    if args.provider_plan is not None:
        provider_plan = load_literature_provider_plan(
            args.provider_plan
        )
    else:
        raw = str(args.providers or "").strip()
        requested = (
            None
            if raw.lower() == "auto"
            else [
                value.strip()
                for value in raw.split(",")
                if value.strip()
            ]
        )
        provider_plan = resolve_literature_provider_plan(
            requested=requested
        )

    require_standard_or_full_auto_plan(provider_plan)
    _write(provider_path, provider_plan)

    providers = build_literature_providers(provider_plan)
    retriever = LiteratureRetriever(
        providers,
        results_per_query=args.results_per_query,
    )
    delta_packet = retriever.retrieve(transport).packet
    _write(delta_path, delta_packet)

    resolution_targets = build_resolution_targets(
        delta_packet,
        max_queries=args.max_resolution_queries,
    )
    resolution_plan = build_resolution_transport_plan(
        source_portfolio_id=args.source_portfolio_id,
        targets=resolution_targets,
    )
    _write(resolution_plan_path, resolution_plan)

    if resolution_plan.queries:
        resolution_packet = retriever.retrieve(
            resolution_plan
        ).packet
    else:
        from pipeline_core.discovery.scientific_relation_second_pass import (
            _empty_packet,
        )
        resolution_packet = _empty_packet(
            resolution_plan
        )

    _write(resolution_packet_path, resolution_packet)

    resolved_packet = merge_second_pass_resolution_packets(
        semantic_packet=delta_packet,
        resolution_packet=resolution_packet,
        semantic_transport_plan=transport,
        resolution_plan=resolution_plan,
        resolution_targets=resolution_targets,
    )
    _write(resolved_path, resolved_packet)

    before_abstracts = sum(
        bool(work.abstract)
        for work in delta_packet.works
    )
    after_abstracts = sum(
        bool(work.abstract)
        for work in resolved_packet.works
    )

    report = ScientificRelationSecondPassReport(
        report_id=(
            "scientific_relation_second_pass_report:"
            + __import__("hashlib").sha256(
                (
                    plan.plan_id
                    + "|"
                    + delta_packet.packet_id
                    + "|"
                    + resolved_packet.packet_id
                ).encode("utf-8")
            ).hexdigest()[:20]
        ),
        source_second_pass_plan_id=plan.plan_id,
        source_delta_packet_id=delta_packet.packet_id,
        source_resolution_packet_id=(
            resolution_packet.packet_id
            if resolution_packet.works
            else None
        ),
        source_resolved_packet_id=resolved_packet.packet_id,
        semantic_query_count=plan.query_count,
        resolution_query_count=len(resolution_plan.queries),
        raw_second_pass_work_count=len(delta_packet.works),
        resolved_unique_work_count=len(resolved_packet.works),
        abstract_work_count_before_resolution=before_abstracts,
        abstract_work_count_after_resolution=after_abstracts,
        supplementary_doi_target_count=sum(
            row.reason == "SUPPLEMENTARY_DOI_BASE_LOOKUP"
            for row in resolution_targets
        ),
        missing_abstract_target_count=sum(
            row.reason
            in {
                "MISSING_ABSTRACT_DOI_LOOKUP",
                "MISSING_ABSTRACT_TITLE_LOOKUP",
            }
            for row in resolution_targets
        ),
    )
    _write(report_path, report)

    if args.save_prompts:
        _write(prompt_path, backend.prompt_records)

    print("Scientific relation second-pass shadow complete")
    print("Claims:", plan.claim_count)
    print("Semantic queries:", plan.query_count)
    print("Providers:", provider_plan.active_providers)
    print("Delta unique works:", len(delta_packet.works))
    print("Delta abstracts:", before_abstracts)
    print("Resolution queries:", len(resolution_plan.queries))
    print(
        "Supplementary DOI targets:",
        report.supplementary_doi_target_count,
    )
    print(
        "Missing-abstract targets:",
        report.missing_abstract_target_count,
    )
    print("Resolved unique works:", len(resolved_packet.works))
    print("Resolved abstracts:", after_abstracts)

    by_claim: dict[str, list] = {}
    for binding in plan.bindings:
        by_claim.setdefault(binding.claim_id, []).append(binding)

    for claim_id, rows in by_claim.items():
        print()
        print("Claim:", claim_id)
        for row in rows:
            print(
                " ",
                row.role,
                "| projections=",
                row.basis_projection_ids,
                "| query=",
                row.query_text,
            )
            if row.generated_endpoint_terms:
                print(
                    "   generated endpoint terms:",
                    row.generated_endpoint_terms,
                )
            if row.generated_counterevidence_terms:
                print(
                    "   generated counter terms:",
                    row.generated_counterevidence_terms,
                )

    print()
    print("Diagnostic only: true")
    print("Semantic expansion is retrieval-only: true")
    print("Relation adjudication performed: false")
    print("Absence-based novelty authorized: false")
    print("Positive non-obviousness authority created: false")
    print("N9 contract changed: false")
    print("N10 contract changed: false")
    print("Production selection changed: false")
    print("Semantic expansions:", draft_path)
    print("Semantic plan:", plan_path)
    print("Resolved prior-art packet:", resolved_path)
    print("Report:", report_path)
    if args.save_prompts:
        print("Prompts:", prompt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
