from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline_core.discovery.semantic_distinctiveness_llm import OpenRouterSemanticDistinctivenessBackend
from pipeline_core.discovery.question_axis_responsiveness_llm import (
    OpenRouterQuestionAxisResponsivenessBackend,
)
from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxisPlan, DiscoveryAxisSynthesisReport
from domains.registry import get_domain_profile
from pipeline_core.discovery.dual_hypothesis_context import DualHypothesisContext
from pipeline_core.discovery.external_novelty import ExternalNoveltyAssessor
from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.external_novelty_llm import InstructorOpenAICompatibleExternalNoveltyBackend
from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.hypothesis_llm import InstructorOpenAICompatibleHypothesisBackend
from pipeline_core.discovery.prior_art_retrieval import LiteratureRetriever
from pipeline_core.discovery.prior_art_provider_plan import (
    build_literature_providers,
    load_literature_provider_plan,
    require_standard_or_full_auto_plan,
    resolve_literature_provider_plan,
)
from pipeline_core.discovery.node_mapping import DEFAULT_EMBED_MODEL, NodeMapper, SentenceTransformerEncoder
from pipeline_core.discovery.novelty_claim_decomposition import NoveltyClaimDecomposer
from pipeline_core.discovery.novelty_gap_analysis import NoveltyGapAnalyzer
from pipeline_core.discovery.novelty_refinement_runtime import TargetedNoveltyRefinementRuntime
from pipeline_core.discovery.prior_art_matching import ClaimPriorArtCompiler, PriorArtRanker
from pipeline_core.discovery.targeted_novelty_retrieval import TargetedNoveltyRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run v2.8.0-alpha6 bounded targeted novelty refinement. "
            "External prior art remains exclusion/boundary evidence only."
        )
    )
    p.add_argument("--dual-context", required=True, type=Path)
    p.add_argument(
        "--domain-profile",
        required=True,
        help="Scientific domain profile for all targeted/final prior-art checks.",
    )
    p.add_argument("--axis-plan", required=True, type=Path)
    p.add_argument("--portfolio", required=True, type=Path)
    p.add_argument("--lineage", required=True, type=Path)
    p.add_argument("--external-report", required=True, type=Path)
    p.add_argument("--external-query-plan", required=True, type=Path)
    p.add_argument("--external-prior-art", required=True, type=Path)
    p.add_argument(
        "--scientific-novelty-gate",
        type=Path,
        default=None,
        help=(
            "Optional authoritative Alpha6 original-fallback gate "
            "compiled from scientific novelty action decisions."
        ),
    )
    p.add_argument(
        "--question-task-preservation-enforce",
        action="store_true",
        help=(
            "Require stable Question-to-fresh-reaxis task preservation "
            "before Alpha6 may accept a fresh-context re-axis."
        ),
    )
    p.add_argument(
        "--post-generation-scientific-novelty-enforce",
        action="store_true",
        help=(
            "Run authoritative two-pass semantic scientific novelty "
            "assessment over each fresh Alpha6 candidate before final "
            "acceptance."
        ),
    )
    p.add_argument("--model", default=os.getenv("OPENROUTER_AGENT_MODEL"))
    p.add_argument(
        "--critic-model",
        default=os.getenv("OPENROUTER_CRITIC_MODEL")
        or os.getenv("OPENROUTER_AGENT_MODEL"),
    )
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=(
            "Mechanism node index used for refinement grounding. "
            "Defaults to the historical data_dac corpus layout when omitted."
        ),
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--providers",
        default="auto",
        help=(
            "Provider set. Default 'auto' resolves OpenAlex+Crossref, "
            "plus Semantic Scholar only with SEMANTIC_SCHOLAR_API_KEY."
        ),
    )
    p.add_argument(
        "--provider-plan",
        default=None,
        type=Path,
        help=(
            "Optional frozen literature-provider plan JSON shared with "
            "the initial external-novelty stage."
        ),
    )
    p.add_argument("--results-per-query", type=int, default=12)
    p.add_argument("--target-claims", type=int, default=2)
    p.add_argument("--target-queries", type=int, default=3)
    p.add_argument("--output-prefix", required=True, type=Path)
    p.add_argument("--dry-run-gap-plan", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    domain_profile = get_domain_profile(args.domain_profile)
    dual = DualHypothesisContext.model_validate_json(
        args.dual_context.read_text(encoding="utf-8")
    )
    axis_plan = DiscoveryAxisPlan.model_validate_json(
        args.axis_plan.read_text(encoding="utf-8")
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        args.portfolio.read_text(encoding="utf-8")
    )
    if dual.domain_profile_id != domain_profile.profile_id:
        raise ValueError(
            "Dual-context/domain profile mismatch: "
            f"dual={dual.domain_profile_id!r}, "
            f"requested={domain_profile.profile_id!r}"
        )
    if portfolio.domain_profile_id != domain_profile.profile_id:
        raise ValueError(
            "Portfolio/domain profile mismatch: "
            f"portfolio={portfolio.domain_profile_id!r}, "
            f"requested={domain_profile.profile_id!r}"
        )
    lineage = DiscoveryAxisSynthesisReport.model_validate_json(
        args.lineage.read_text(encoding="utf-8")
    )
    external = ExternalNoveltyReport.model_validate_json(
        args.external_report.read_text(encoding="utf-8")
    )
    query_plan = LiteratureQueryPlan.model_validate_json(
        args.external_query_plan.read_text(encoding="utf-8")
    )
    prior_art = PriorArtPacket.model_validate_json(
        args.external_prior_art.read_text(encoding="utf-8")
    )

    gap_analyzer = NoveltyGapAnalyzer(
        max_target_claims=args.target_claims,
        queries_per_gap=args.target_queries,
        domain_profile=domain_profile,
    )
    gap_plan = gap_analyzer.build(portfolio, external, query_plan)
    _write(Path(str(args.output_prefix) + ".gap_plan.json"), gap_plan)
    print("NoveltyGapPlan built")
    print("Plan ID:", gap_plan.plan_id)
    for i, gap in enumerate(gap_plan.gaps, 1):
        print(f"[{i}] {gap.action} | {gap.source_external_status}")
        print("    differentiator:", gap.differentiator)
        for q in gap.targeted_queries:
            print(
                f"    query[{q.claim_id}|{q.query_role}]:",
                q.query_text,
            )

    if args.dry_run_gap_plan:
        return 0
    if not args.model or not args.critic_model:
        raise SystemExit("--model and --critic-model are required")

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
    _write(
        Path(str(args.output_prefix) + ".provider_plan.json"),
        provider_plan,
    )
    providers = build_literature_providers(provider_plan)
    retriever = LiteratureRetriever(
        providers,
        results_per_query=args.results_per_query,
    )
    targeted_retriever = TargetedNoveltyRetriever(retriever)

    encoder = SentenceTransformerEncoder(args.embed_model, device=args.device)
    external_backend = InstructorOpenAICompatibleExternalNoveltyBackend(
        model=args.critic_model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
    )
    policy = ExternalNoveltyPolicy()
    decomposer = NoveltyClaimDecomposer(
        external_backend,
        max_claims_per_hypothesis=policy.max_claims_per_hypothesis,
        max_queries_per_claim=policy.max_queries_per_claim,
    )
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
    external_assessor = ExternalNoveltyAssessor(
        decomposer=decomposer,
        ranker=ranker,
        review_backend=external_backend,
        policy=policy,
        compiler=compiler,
    )

    index_dir = args.index_dir or (
        PROJECT_ROOT
        / "data_dac"
        / "corpus"
        / dual.grounded_context.corpus_id
        / "mechanism"
        / "navigation"
        / "node_index"
    )
    mapper = NodeMapper.from_directory(index_dir, device=args.device)
    hypothesis_backend = InstructorOpenAICompatibleHypothesisBackend(
        model=args.model,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=0.0,
        parse_retries=1,
    )
    task_responsiveness_backend = (
        OpenRouterQuestionAxisResponsivenessBackend(
            model=args.critic_model,
            temperature=0.0,
            reasoning_effort="medium",
            telemetry_context={
                "stage":
                    "alpha6_fresh_reaxis_task_preservation",
            },
        )
        if args.question_task_preservation_enforce
        else None
    )

    post_generation_scientific_novelty_model = (
        args.critic_model
        or args.model
    )

    if (
        args.post_generation_scientific_novelty_enforce
        and not post_generation_scientific_novelty_model
    ):
        raise RuntimeError(
            "--post-generation-scientific-novelty-enforce "
            "requires --critic-model or --model"
        )

    post_generation_scientific_novelty_backend = (
        OpenRouterSemanticDistinctivenessBackend(
            model=post_generation_scientific_novelty_model,
            temperature=0.0,
            reasoning_effort="medium",
            telemetry_context={
                "stage":
                    "alpha6_post_generation_scientific_novelty",
                "production_authority":
                    True,
            },
        )
        if args.post_generation_scientific_novelty_enforce
        else None
    )

    runtime = TargetedNoveltyRefinementRuntime(
        hypothesis_backend=hypothesis_backend,
        external_assessor=external_assessor,
        targeted_retriever=targeted_retriever,
        mapper=mapper,
        gap_analyzer=gap_analyzer,
        task_responsiveness_backend=(
            task_responsiveness_backend
        ),
        post_generation_scientific_novelty_backend=(
            post_generation_scientific_novelty_backend
        ),
    )
    outcome = runtime.run(
        dual=dual,
        portfolio=portfolio,
        lineage=lineage,
        axis_plan=axis_plan,
        external_report=external,
        external_query_plan=query_plan,
        external_prior_art=prior_art,
        scientific_novelty_gate=(
            json.loads(
                args.scientific_novelty_gate.read_text(
                    encoding="utf-8"
                )
            )
            if args.scientific_novelty_gate
            else None
        ),
    )

    prefix = args.output_prefix
    _write(Path(str(prefix) + ".portfolio.json"), outcome.portfolio)
    _write(Path(str(prefix) + ".report.json"), outcome.report)
    _write(Path(str(prefix) + ".gap_plan.json"), outcome.gap_plan)

    detail_dir = Path(str(prefix) + ".external")
    for i, row in enumerate(outcome.targeted_external_artifacts, 1):
        stem = detail_dir / f"targeted_{i:02d}_{row.hypothesis_id.split(':')[-1]}"
        _write(Path(str(stem) + ".claims_queries.json"), row.query_plan)
        _write(Path(str(stem) + ".prior_art.json"), row.prior_art)
        _write(Path(str(stem) + ".report.json"), row.report)
    for i, row in enumerate(outcome.final_external_artifacts, 1):
        stem = detail_dir / f"final_{i:02d}_{row.hypothesis_id.split(':')[-1]}"
        _write(Path(str(stem) + ".claims_queries.json"), row.query_plan)
        _write(Path(str(stem) + ".prior_art.json"), row.prior_art)
        _write(Path(str(stem) + ".report.json"), row.report)

    print()
    print("Targeted novelty refinement complete")
    print("Domain profile:", domain_profile.profile_id)
    print("Provider mode:", provider_plan.mode)
    print("Provider plan:", provider_plan.plan_id)
    print("Final hypotheses:", len(outcome.portfolio.hypotheses))
    print("Accepted refinements:", outcome.report.accepted_refinement_count)
    print("Accepted fresh re-axes:", outcome.report.accepted_reaxis_count)
    print("Kept originals:", outcome.report.kept_original_count)
    print("Rejected/abstained:", outcome.report.rejected_count)
    for i, row in enumerate(outcome.report.attempts, 1):
        print(
            f"[{i}] {row.decision} "
            f"external={row.original_external_status}"
            f"->{row.targeted_external_status}"
            f"->{row.final_external_status}"
        )
        print(
            "    grounding=", row.grounding_preserved,
            "axis=", row.axis_fidelity_status,
            "internal=", row.internal_novelty_status,
        )
        if row.reason_codes:
            print("    reasons=", ", ".join(row.reason_codes))
        print("    ", row.interpretation)
    print("Saved portfolio:", Path(str(prefix) + ".portfolio.json"))
    print("Saved report:", Path(str(prefix) + ".report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
