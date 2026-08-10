from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dac_her.discovery_axis_contracts import DiscoveryAxisSynthesisReport
from dac_her.external_novelty import ExternalNoveltyAssessor
from dac_her.external_novelty_contracts import (
    ExternalNoveltyPolicy,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from dac_her.external_novelty_llm import (
    InstructorOpenAICompatibleExternalNoveltyBackend,
)
from dac_her.hypothesis_contracts import HypothesisPortfolio
from dac_her.literature_retrieval import (
    CrossrefProvider,
    LiteratureRetriever,
    SemanticScholarProvider,
    canonicalize_prior_art_packet,
)
from dac_her.node_mapping import DEFAULT_EMBED_MODEL, SentenceTransformerEncoder
from dac_her.novelty_claim_decomposition import (
    LiteratureQueryPlanner,
    NoveltyClaimDecomposer,
)
from dac_her.prior_art_matching import PriorArtRanker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess generated hypotheses against external prior-art metadata "
            "without promoting external literature into positive premises."
        )
    )
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--lineage", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--device", default=None)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--providers",
        default="semantic_scholar,crossref",
        help="Comma-separated: semantic_scholar,crossref",
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


def main() -> None:
    args = parse_args()
    portfolio = HypothesisPortfolio.model_validate_json(
        Path(args.portfolio).read_text(encoding="utf-8")
    )
    lineage = (
        DiscoveryAxisSynthesisReport.model_validate_json(
            Path(args.lineage).read_text(encoding="utf-8")
        )
        if args.lineage
        else None
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
        decompositions = [decomposer.decompose(row) for row in portfolio.hypotheses]
        plan = LiteratureQueryPlanner().build(portfolio, decompositions)

    prefix = Path(args.output_prefix)
    _write(prefix.with_suffix(".claims_queries.json"), plan)

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
        requested = [x.strip() for x in args.providers.split(",") if x.strip()]
        providers = []
        for name in requested:
            if name == "semantic_scholar":
                providers.append(SemanticScholarProvider())
            elif name == "crossref":
                providers.append(CrossrefProvider())
            else:
                raise ValueError(f"Unknown provider: {name}")
        packet = LiteratureRetriever(
            providers,
            results_per_query=args.results_per_query,
        ).retrieve(plan).packet
    _write(prefix.with_suffix(".prior_art.json"), packet)

    encoder = SentenceTransformerEncoder(args.embed_model, device=args.device)
    ranker = PriorArtRanker(
        encoder,
        max_ranked_works_per_claim=policy.max_ranked_works_per_claim,
    )
    assessor = ExternalNoveltyAssessor(
        decomposer=decomposer,
        ranker=ranker,
        review_backend=backend,
        policy=policy,
    )
    report = assessor.assess(
        portfolio,
        plan,
        packet,
        lineage=lineage,
    )
    _write(prefix.with_suffix(".report.json"), report)

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
    print("Prior-art packet:", packet.packet_id)
    print("Search providers:", ", ".join(packet.providers_requested))
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
    print("Saved report:", prefix.with_suffix(".report.json"))


if __name__ == "__main__":
    main()
