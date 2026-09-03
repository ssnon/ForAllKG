from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxisSynthesisReport
from domains.registry import get_domain_profile
from pipeline_core.discovery.external_novelty import ExternalNoveltyAssessor
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    HypothesisNoveltyClaims,
    LiteratureQueryPlan,
    NoveltyClaimInferenceProvenance,
    PriorArtPacket,
)
from pipeline_core.discovery.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.prior_art_retrieval import (
    LiteratureRetriever,
    canonicalize_prior_art_packet,
)
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    load_literature_provider_plan,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)
from pipeline_core.discovery.node_mapping import DEFAULT_EMBED_MODEL, SentenceTransformerEncoder
from pipeline_core.discovery.novelty_claim_decomposition import (
    LiteratureQueryPlanner,
    NoveltyClaimDecomposer,
)
from pipeline_core.discovery.novelty_inference_provenance import (
    attach_atomic_inference_provenance,
)
from pipeline_core.discovery.prior_art_matching import ClaimPriorArtCompiler, PriorArtRanker
from pipeline_core.discovery.prior_art_review_audit import (
    prior_art_review_audit_scope,
)
from pipeline_core.discovery.prior_art_memory import (
    PriorArtMemoryMatcher,
    augment_prior_art_packet_with_memory,
    build_prior_art_memory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess generated hypotheses against external prior-art metadata "
            "without promoting external literature into positive premises."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--domain-profile", required=True)
    parser.add_argument("--lineage", default=None)
    parser.add_argument(
        "--inference-audit",
        default=None,
        help=(
            "Optional Alpha4 discovery-axis inference artifact. "
            "When supplied, accepted inference provenance is preserved "
            "on atomic novelty claims without changing search queries "
            "or scientific authority."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--device", default=None)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--providers",
        default="auto",
        help=(
            "Provider set. Default 'auto' resolves OpenAlex+Crossref, "
            "plus Semantic Scholar only when SEMANTIC_SCHOLAR_API_KEY "
            "is configured. Advanced explicit sets may use "
            "openalex,crossref,semantic_scholar."
        ),
    )
    parser.add_argument(
        "--provider-plan",
        default=None,
        help=(
            "Optional frozen literature-provider plan JSON. When supplied, "
            "the run uses exactly that provider set and verifies environment "
            "configuration has not drifted."
        ),
    )
    parser.add_argument("--results-per-query", type=int, default=12)
    parser.add_argument("--max-claims", type=int, default=4)
    parser.add_argument("--max-queries-per-claim", type=int, default=2)
    parser.add_argument("--max-ranked-works", type=int, default=8)
    parser.add_argument("--min-unique-works-for-absence", type=int, default=10)
    parser.add_argument("--min-abstract-works-for-absence", type=int, default=5)
    parser.add_argument("--min-abstract-works-per-core-claim", type=int, default=3)
    parser.add_argument(
        "--reuse-query-plan",
        default=None,
        help="Reuse an existing .claims_queries.json for an exact alpha5 -> alpha5.1 A/B rerun.",
    )
    parser.add_argument("--reuse-prior-art", default=None)
    parser.add_argument(
        "--prior-art-memory-query-plan",
        default=None,
        help=(
            "Optional historical LiteratureQueryPlan whose reviewed "
            "prior art may be re-exposed to compatible current claims."
        ),
    )
    parser.add_argument(
        "--prior-art-memory-report",
        default=None,
        help=(
            "Historical ExternalNoveltyReport corresponding to "
            "--prior-art-memory-query-plan."
        ),
    )
    parser.add_argument(
        "--prior-art-memory-packet",
        default=None,
        help=(
            "Historical PriorArtPacket containing works referenced "
            "by --prior-art-memory-report."
        ),
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--save-prompts", action="store_true")
    return parser.parse_args()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _unique_strings(
    values: list[object],
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value or "").strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        output.append(text)

    return output


def _load_inference_provenance_map(
    *,
    path: str | Path,
    portfolio: HypothesisPortfolio,
) -> dict[str, NoveltyClaimInferenceProvenance]:
    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get("schema_version")
        != "discovery-axis-inference-artifact-v2"
    ):
        raise ValueError(
            "Unexpected Alpha4 inference artifact schema."
        )

    if (
        payload.get("portfolio_id")
        != portfolio.portfolio_id
    ):
        raise ValueError(
            "Inference artifact / portfolio provenance mismatch."
        )

    records = payload.get("records")

    if not isinstance(records, list):
        raise ValueError(
            "Inference artifact records must be a list."
        )

    result: dict[
        str,
        NoveltyClaimInferenceProvenance,
    ] = {}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                "Inference artifact final record must be an object."
            )

        final_hypothesis_id = str(
            record.get("final_hypothesis_id")
            or ""
        ).strip()

        source_review_hypothesis_id = str(
            record.get(
                "source_review_hypothesis_id"
            )
            or ""
        ).strip()

        axis_id = str(
            record.get("axis_id")
            or ""
        ).strip()

        review = record.get("review")

        if (
            not final_hypothesis_id
            or not source_review_hypothesis_id
            or not axis_id
            or not isinstance(review, dict)
        ):
            raise ValueError(
                "Incomplete inference provenance record."
            )

        if (
            str(
                review.get("hypothesis_id")
                or ""
            )
            != source_review_hypothesis_id
        ):
            raise ValueError(
                "Inference source-review hypothesis mismatch."
            )

        if (
            str(
                review.get("axis_id")
                or ""
            )
            != axis_id
        ):
            raise ValueError(
                "Inference axis binding mismatch."
            )

        assertions = review.get("assertions")

        if not isinstance(assertions, list):
            raise ValueError(
                "Inference review assertions must be a list."
            )

        assertion_ids = []
        source_classes = []
        grounded_statement_ids = []
        axis_basis = []

        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue

            assertion_ids.append(
                assertion.get("assertion_id")
            )

            source_classes.append(
                assertion.get("source_class")
            )

            grounded_statement_ids.extend(
                assertion.get(
                    "grounded_statement_ids"
                )
                or []
            )

            axis_basis.extend(
                assertion.get("axis_basis")
                or []
            )

        result[
            final_hypothesis_id
        ] = NoveltyClaimInferenceProvenance(
            final_hypothesis_id=(
                final_hypothesis_id
            ),
            source_review_hypothesis_id=(
                source_review_hypothesis_id
            ),
            axis_id=axis_id,
            review_status=str(
                review.get("status")
                or record.get("status")
                or ""
            ),
            assertion_ids=_unique_strings(
                assertion_ids
            ),
            source_classes=_unique_strings(
                source_classes
            ),
            grounded_statement_ids=_unique_strings(
                grounded_statement_ids
            ),
            axis_basis=_unique_strings(
                axis_basis
            ),
        )

    expected_ids = {
        row.hypothesis_id
        for row in portfolio.hypotheses
    }

    actual_ids = set(result)

    if actual_ids != expected_ids:
        raise ValueError(
            "Inference artifact final hypothesis IDs do not "
            "match the external-novelty portfolio: "
            f"expected={sorted(expected_ids)}, "
            f"actual={sorted(actual_ids)}"
        )

    return result


def _attach_inference_provenance(
    *,
    decompositions: list[
        HypothesisNoveltyClaims
    ],
    provenance_by_hypothesis: dict[
        str,
        NoveltyClaimInferenceProvenance,
    ],
) -> list[HypothesisNoveltyClaims]:
    if not provenance_by_hypothesis:
        return decompositions

    output: list[
        HypothesisNoveltyClaims
    ] = []

    for group in decompositions:
        provenance = (
            provenance_by_hypothesis.get(
                group.hypothesis_id
            )
        )

        claims = [
            claim.model_copy(
                update={
                    "inference_provenance":
                        provenance
                }
            )
            for claim in group.claims
        ]

        output.append(
            group.model_copy(
                update={
                    "claims": claims
                }
            )
        )

    return output


def main() -> None:
    args = parse_args()

    memory_args = (
        args.prior_art_memory_query_plan,
        args.prior_art_memory_report,
        args.prior_art_memory_packet,
    )
    memory_enabled = any(memory_args)

    if memory_enabled and not all(memory_args):
        raise ValueError(
            "Prior-art memory requires all three inputs: "
            "--prior-art-memory-query-plan, "
            "--prior-art-memory-report, and "
            "--prior-art-memory-packet."
        )

    domain_profile = get_domain_profile(args.domain_profile)
    portfolio = HypothesisPortfolio.model_validate_json(
        Path(args.portfolio).read_text(encoding="utf-8")
    )
    if portfolio.domain_profile_id != domain_profile.profile_id:
        raise ValueError(
            "Portfolio/domain profile mismatch: "
            f"portfolio={portfolio.domain_profile_id!r}, "
            f"requested={domain_profile.profile_id!r}"
        )
    lineage = (
        DiscoveryAxisSynthesisReport.model_validate_json(
            Path(args.lineage).read_text(encoding="utf-8")
        )
        if args.lineage
        else None
    )

    inference_provenance_by_hypothesis = (
        _load_inference_provenance_map(
            path=args.inference_audit,
            portfolio=portfolio,
        )
        if args.inference_audit
        else {}
    )

    policy = ExternalNoveltyPolicy(
        max_claims_per_hypothesis=args.max_claims,
        max_queries_per_claim=args.max_queries_per_claim,
        max_ranked_works_per_claim=args.max_ranked_works,
        min_unique_works_for_absence=args.min_unique_works_for_absence,
        min_abstract_works_for_absence=args.min_abstract_works_for_absence,
        min_abstract_works_per_core_claim=args.min_abstract_works_per_core_claim,
    )
    backend = InstructorOpenAICompatibleExternalNoveltyBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        capture_prompts=args.save_prompts,
    )
    decomposer = NoveltyClaimDecomposer(
        backend,
        max_claims_per_hypothesis=policy.max_claims_per_hypothesis,
        max_queries_per_claim=policy.max_queries_per_claim,
    )
    if args.reuse_query_plan:
        plan = LiteratureQueryPlan.model_validate_json(
            Path(args.reuse_query_plan).read_text(encoding="utf-8")
        )
        if plan.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("--reuse-query-plan source_portfolio_id mismatch")
    else:
        decompositions = [
            decomposer.decompose(row)
            for row in portfolio.hypotheses
        ]

        if args.inference_audit:
            decompositions = (
                attach_atomic_inference_provenance(
                    decompositions=decompositions,
                    inference_audit_path=(
                        args.inference_audit
                    ),
                    fallback_by_hypothesis=(
                        inference_provenance_by_hypothesis
                    ),
                )
            )

        plan = LiteratureQueryPlanner().build(
            portfolio,
            decompositions,
        )

    prefix = Path(args.output_prefix)

    # Diagnostic-only sidecar containing specification values
    # before and after branch-specific sanitization.
    #
    # It is intentionally outside LiteratureQueryPlan and is not
    # consumed by retrieval, evidence closure, N9, or N10.
    #
    # In --reuse-query-plan mode no decomposition occurs, so the
    # record list is deliberately empty rather than reusing stale
    # diagnostic data from another run.
    _write(
        prefix.with_suffix(
            ".specification_sanitization.json"
        ),
        {
            "schema_version": (
                "novelty-specification-"
                "sanitization-audit-v1"
            ),
            "source_portfolio_id": (
                portfolio.portfolio_id
            ),
            "diagnostic_only": True,
            "decomposition_executed": (
                not bool(args.reuse_query_plan)
            ),
            "records": list(
                decomposer.specification_sanitization_records
            ),
        },
    )

    report_path = prefix.with_suffix(".report.json")
    # Remove a stale final report before the assessment starts. If the
    # assessment crashes, downstream stages cannot accidentally consume a
    # report from an older portfolio/run.
    if report_path.exists():
        report_path.unlink()
    _write(prefix.with_suffix(".claims_queries.json"), plan)

    provider_plan = None
    provider_plan_path = None
    if args.reuse_prior_art:
        packet = PriorArtPacket.model_validate_json(
            Path(args.reuse_prior_art).read_text(encoding="utf-8")
        )
        if packet.source_query_plan_id != plan.plan_id:
            raise ValueError(
                "--reuse-prior-art packet was produced from a different query plan; "
                "reuse the matching .claims_queries.json or run retrieval again."
            )
        packet = canonicalize_prior_art_packet(packet)
    else:
        if args.provider_plan:
            provider_plan = load_literature_provider_plan(
                args.provider_plan
            )
        else:
            provider_arg = str(args.providers or "").strip()
            if provider_arg.lower() == "auto":
                requested = None
            else:
                requested = [
                    x.strip()
                    for x in provider_arg.split(",")
                    if x.strip()
                ]
            provider_plan = resolve_literature_provider_plan(
                requested=requested
            )
        require_standard_or_full_auto_plan(provider_plan)
        provider_plan_path = prefix.with_suffix(
            ".provider_plan.json"
        )
        _write(provider_plan_path, provider_plan)
        providers = build_literature_providers(
            provider_plan
        )
        packet = LiteratureRetriever(
            providers,
            results_per_query=args.results_per_query,
        ).retrieve(plan).packet
    encoder = SentenceTransformerEncoder(
        args.embed_model,
        device=args.device,
    )

    if memory_enabled:
        memory_plan = LiteratureQueryPlan.model_validate_json(
            Path(
                args.prior_art_memory_query_plan
            ).read_text(encoding="utf-8")
        )
        memory_report = ExternalNoveltyReport.model_validate_json(
            Path(
                args.prior_art_memory_report
            ).read_text(encoding="utf-8")
        )
        memory_packet = PriorArtPacket.model_validate_json(
            Path(
                args.prior_art_memory_packet
            ).read_text(encoding="utf-8")
        )

        if (
            memory_report.source_portfolio_id
            != memory_plan.source_portfolio_id
        ):
            raise ValueError(
                "prior-art memory plan/report portfolio mismatch"
            )

        if (
            memory_packet.source_portfolio_id
            != memory_plan.source_portfolio_id
        ):
            raise ValueError(
                "prior-art memory plan/packet portfolio mismatch"
            )

        if (
            memory_report.source_prior_art_packet_id
            != memory_packet.packet_id
        ):
            raise ValueError(
                "prior-art memory report/packet mismatch"
            )

        if (
            memory_packet.source_query_plan_id
            != memory_plan.plan_id
        ):
            raise ValueError(
                "prior-art memory packet/query-plan mismatch"
            )

        memory = build_prior_art_memory(
            memory_plan,
            memory_report,
        )

        packet, memory_matches = (
            augment_prior_art_packet_with_memory(
                current_plan=plan,
                current_packet=packet,
                memory=memory,
                memory_packet=memory_packet,
                matcher=PriorArtMemoryMatcher(
                    encoder
                ),
            )
        )

        _write(
            prefix.with_suffix(
                ".prior_art_memory.json"
            ),
            {
                "enabled": True,
                "historical_query_plan_id": (
                    memory_plan.plan_id
                ),
                "historical_report_id": (
                    memory_report.report_id
                ),
                "historical_prior_art_packet_id": (
                    memory_packet.packet_id
                ),
                "memory_entry_count": len(memory),
                "reexposed_work_ids_by_claim": {
                    claim_id: list(work_ids)
                    for claim_id, work_ids
                    in sorted(
                        memory_matches.items()
                    )
                },
            },
        )

    _write(prefix.with_suffix(".prior_art.json"), packet)

    ranker = PriorArtRanker(
        encoder,
        max_ranked_works_per_claim=policy.max_ranked_works_per_claim,
        domain_profile=domain_profile,
    )
    compiler = ClaimPriorArtCompiler(
        min_match_confidence=policy.min_match_confidence,
        direct_match_confidence=policy.direct_match_confidence,
        require_abstract_for_strong_match=policy.require_abstract_for_strong_match,
        require_abstract_for_partial_match=policy.require_abstract_for_partial_match,
        min_reaction_domain_for_conflict=policy.min_reaction_domain_for_conflict,
        min_catalyst_scope_for_conflict=policy.min_catalyst_scope_for_conflict,
        domain_profile=domain_profile,
    )
    assessor = ExternalNoveltyAssessor(
        decomposer=decomposer,
        ranker=ranker,
        review_backend=backend,
        policy=policy,
        compiler=compiler,
    )
    with prior_art_review_audit_scope(
        assessment_kind="alpha5_initial",
        source_portfolio_id=portfolio.portfolio_id,
        query_plan_id=plan.plan_id,
        prior_art_packet_id=packet.packet_id,
    ):
        report = assessor.assess(
            portfolio,
            plan,
            packet,
            lineage=lineage,
        )
    _write(report_path, report)

    if args.save_prompts:
        prompt_dir = prefix.parent / (prefix.name + ".prompts")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(backend.prompt_records, start=1):
            safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in row.name)
            target = prompt_dir / f"{index:02d}_{safe}.txt"
            target.write_text(
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

    print("External novelty assessment complete")
    print("Portfolio:", portfolio.portfolio_id)
    print("Domain profile:", domain_profile.profile_id)
    print("Prior-art packet:", packet.packet_id)
    print("Search providers:", ", ".join(packet.providers_requested))
    if provider_plan is not None:
        print("Provider mode:", provider_plan.mode)
        print("Provider plan:", provider_plan.plan_id)
        print("Provider plan artifact:", provider_plan_path)
    print("Unique works:", len(packet.works))
    print(
        "Canonicalization:",
        f"raw={packet.raw_work_count or len(packet.works)}",
        f"canonical={packet.canonical_work_count or len(packet.works)}",
        f"deduplicated={packet.deduplicated_work_count}",
        f"supplementary_collapsed={packet.supplementary_records_collapsed}",
    )
    success = sum(row.success for row in packet.executions)
    print("Successful provider-query executions:", success, "/", len(packet.executions))
    print("Status counts:", dict(Counter(row.status for row in report.cards)))
    print()
    for index, card in enumerate(report.cards, start=1):
        print(
            f"[{index}] {card.status} | works={card.coverage.unique_work_count} "
            f"abstracts={card.coverage.abstract_work_count} "
            f"absence_coverage={card.coverage.sufficient_for_absence_based_novelty}"
        )
        print("    ", card.title)
        for review in card.claim_reviews:
            print(
                f"     - {review.importance} {review.status}: "
                f"{review.claim_text[:160]}"
            )
            for match in review.matches[:2]:
                if match.relationship in {"UNRELATED", "INSUFFICIENT_METADATA"}:
                    continue
                print(
                    f"         {match.relationship} conf={match.confidence:.2f} "
                    f"rel={match.relevance_score:.2f} "
                    f"rxn={match.reaction_domain_relevance:.2f} "
                    f"scope={match.catalyst_scope_relevance:.2f} | "
                    f"{match.title[:120]}"
                )
        print("     interpretation:", card.interpretation)
        print()

    print("Saved query plan:", prefix.with_suffix(".claims_queries.json"))
    print("Saved prior art:", prefix.with_suffix(".prior_art.json"))
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
