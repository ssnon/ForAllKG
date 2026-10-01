from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.registry import get_domain_profile
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.lower_order_prior_art_saturation import (
    build_lower_order_saturation_query_plan,
    build_lower_order_saturation_report,
)
from pipeline_core.discovery.node_mapping import (
    DEFAULT_EMBED_MODEL,
    SentenceTransformerEncoder,
)
from pipeline_core.discovery.prior_art_matching import (
    ClaimPriorArtCompiler,
    PriorArtRanker,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    load_literature_provider_plan,
    require_standard_or_full_auto_plan,
)
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureRetriever,
)
from pipeline_core.discovery.prior_art_review_audit import (
    prior_art_review_audit_scope,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_path(prefix: Path, suffix: str) -> Path:
    # Append an artifact suffix without replacing dotted prefix components.
    suffix = str(suffix).lstrip(".")
    return prefix.parent / f"{prefix.name}.{suffix}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Shadow exact-verification retrieval for canonical atomic novelty "
            "claims, plus same-work component closure and residual novelty "
            "diagnostics. This pass has no novelty/N9/N10/production authority."
        )
    )
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--query-plan", required=True, type=Path)
    p.add_argument("--external-report", required=True, type=Path)
    p.add_argument("--provider-plan", required=True, type=Path)
    p.add_argument("--domain-profile", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--device", default=None)
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--max-ranked-works", type=int, default=12)
    p.add_argument("--parse-retries", type=int, default=3)
    p.add_argument("--output-prefix", required=True, type=Path)
    p.add_argument("--save-prompts", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    base_plan = LiteratureQueryPlan.model_validate_json(
        args.query_plan.read_text(encoding="utf-8")
    )
    external_report = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(encoding="utf-8")
    )

    if base_plan.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("query plan / portfolio mismatch")
    if external_report.source_portfolio_id != portfolio.portfolio_id:
        raise ValueError("external report / portfolio mismatch")

    domain_profile = get_domain_profile(args.domain_profile)
    if portfolio.domain_profile_id != domain_profile.profile_id:
        raise ValueError("portfolio/domain profile mismatch")

    saturation_plan, targets = build_lower_order_saturation_query_plan(base_plan)

    prefix = args.output_prefix.expanduser().resolve()
    _write(_artifact_path(prefix, "queries.json"), saturation_plan)
    _write(
        _artifact_path(prefix, "targets.json"),
        {
            "schema_version": "lower-order-prior-art-saturation-targets-v1",
            "source_query_plan_id": base_plan.plan_id,
            "targets": [row.model_dump(mode="json") for row in targets],
            "shadow_only": True,
            "production_selection_changed": False,
        },
    )

    provider_plan = load_literature_provider_plan(args.provider_plan)
    require_standard_or_full_auto_plan(provider_plan)
    providers = build_literature_providers(provider_plan)

    packet = LiteratureRetriever(
        providers,
        results_per_query=args.results_per_query,
    ).retrieve(saturation_plan).packet
    _write(_artifact_path(prefix, "prior_art.json"), packet)

    encoder = SentenceTransformerEncoder(
        args.embed_model,
        device=args.device,
    )
    ranker = PriorArtRanker(
        encoder,
        max_ranked_works_per_claim=args.max_ranked_works,
        domain_profile=domain_profile,
    )
    backend = InstructorOpenAICompatibleExternalNoveltyBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=0.0,
        parse_retries=args.parse_retries,
        capture_prompts=args.save_prompts,
        evidence_grounded_review=False,
        telemetry_path=_artifact_path(prefix, "telemetry.jsonl"),
        telemetry_context={
            "pipeline": "lower_order_prior_art_saturation_shadow",
        },
    )
    compiler = ClaimPriorArtCompiler(
        require_grounded_evidence_spans=False,
        domain_profile=domain_profile,
    )

    claim_by_id = {
        claim.claim_id: claim
        for group in saturation_plan.claims
        for claim in group.claims
    }
    work_index = {row.work_id: row for row in packet.works}

    reviews = []
    with prior_art_review_audit_scope(
        assessment_kind="lower_order_prior_art_saturation_shadow",
        source_portfolio_id=portfolio.portfolio_id,
        query_plan_id=saturation_plan.plan_id,
        prior_art_packet_id=packet.packet_id,
    ):
        for target in targets:
            claim = claim_by_id[target.claim_id]
            candidates = ranker.rank(
                claim,
                packet,
                saturation_plan,
            )
            review_input = []
            for ranked in candidates.ranked_works:
                work = work_index[ranked.work_id]
                review_input.append(
                    {
                        "work_id": work.work_id,
                        "title": work.title,
                        "year": work.year,
                        "doi": work.doi,
                        "abstract": work.abstract,
                        "semantic_similarity": ranked.semantic_similarity,
                        "lexical_coverage": ranked.lexical_coverage,
                        "reaction_domain_relevance": ranked.reaction_domain_relevance,
                        "catalyst_scope_relevance": ranked.catalyst_scope_relevance,
                        "relevance_score": ranked.relevance_score,
                    }
                )
            draft = backend.review_exact_verification_claim(claim, review_input)
            reviews.append(
                compiler.compile(
                    claim,
                    candidates,
                    draft,
                    packet,
                    saturation_plan,
                )
            )

    _write(
        _artifact_path(prefix, "claim_reviews.json"),
        {
            "schema_version": "lower-order-prior-art-saturation-claim-reviews-v1",
            "source_portfolio_id": portfolio.portfolio_id,
            "source_query_plan_id": saturation_plan.plan_id,
            "source_prior_art_packet_id": packet.packet_id,
            "reviews": [row.model_dump(mode="json") for row in reviews],
            "shadow_only": True,
            "novelty_authority_created": False,
            "production_selection_changed": False,
        },
    )

    report = build_lower_order_saturation_report(
        base_plan=base_plan,
        source_external_report=external_report,
        saturation_plan=saturation_plan,
        saturation_packet_id=packet.packet_id,
        targets=targets,
        saturation_reviews=reviews,
    )
    _write(_artifact_path(prefix, "report.json"), report)

    if args.save_prompts:
        prompt_dir = prefix.parent / (prefix.name + ".prompts")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(backend.prompt_records, start=1):
            safe = "".join(
                ch if ch.isalnum() or ch in "_-" else "_"
                for ch in row.name
            )
            (prompt_dir / f"{index:02d}_{safe}.txt").write_text(
                "\n".join(
                    [
                        f"prompt_sha256: {row.prompt_sha256}",
                        "",
                        "SYSTEM",
                        "======",
                        row.system_prompt,
                        "",
                        "USER",
                        "====",
                        row.user_prompt,
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    print("Lower-order prior-art saturation shadow complete")
    print("Portfolio:", portfolio.portfolio_id)
    print("Exact-verification queries:", len(saturation_plan.queries))
    print("Targets:", len(targets))
    print("Retrieved works:", len(packet.works))
    print("Status counts:", report.saturation_status_counts)
    print("Residual classes:", report.residual_class_counts)
    print("Relation-backed targets:", report.relation_backed_target_count)
    print("Same-work closure composites:", report.same_work_closure_composite_count)
    print("Residual-gap composites:", report.residual_gap_composite_count)
    print("Report:", _artifact_path(prefix, "report.json"))
    print()

    for summary in report.hypothesis_summaries:
        print(
            summary.hypothesis_id,
            "| external=", summary.source_external_status,
            "| targets=", summary.target_claim_count,
            "| backed=", summary.relation_backed_target_count,
            "| unresolved=", summary.unresolved_target_count,
            "| same_work=", summary.same_work_closure_composite_count,
            "| residual=", summary.residual_gap_composite_count,
        )

    print()
    for row in report.composite_residuals:
        print(
            "COMPOSITE",
            row.composite_claim_id,
            "|", row.residual_class,
            "| components",
            f"{len(row.saturated_component_claim_ids)}/{len(row.component_claim_ids)}",
            "| same_work=",
            len(row.same_work_component_closure_work_ids),
        )

    print()
    print("SHADOW_ONLY=True")
    print("NOVELTY_AUTHORITY_CREATED=False")
    print("N9_AUTHORITY_CREATED=False")
    print("N10_AUTHORITY_CREATED=False")
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
