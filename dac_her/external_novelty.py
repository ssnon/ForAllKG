from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxisSynthesisReport
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ExternalNoveltyCard,
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    HypothesisSearchCoverage,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio
from pipeline_core.discovery.novelty_claim_decomposition import NoveltyClaimDecomposer
from dac_her.prior_art_matching import (
    ClaimPriorArtCompiler,
    ClaimReviewBackend,
    PriorArtRanker,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


class ExternalNoveltyAssessor:
    """Search-bounded external prior-art assessor.

    External literature is used only as prior-art evidence. Nothing in this
    runtime changes HypothesisContext or premise_statement_ids.
    """

    def __init__(
        self,
        *,
        decomposer: NoveltyClaimDecomposer,
        ranker: PriorArtRanker,
        review_backend: ClaimReviewBackend,
        policy: ExternalNoveltyPolicy | None = None,
        compiler: ClaimPriorArtCompiler,
    ) -> None:
        self.decomposer = decomposer
        self.ranker = ranker
        self.review_backend = review_backend
        self.policy = policy or ExternalNoveltyPolicy()
        self.compiler = compiler

    def decompose_portfolio(self, portfolio: HypothesisPortfolio) -> list[Any]:
        return [self.decomposer.decompose(row) for row in portfolio.hypotheses]

    def _coverage(
        self,
        hypothesis: HypothesisCard,
        reviews: list[ClaimPriorArtReview],
        packet: PriorArtPacket,
        plan: LiteratureQueryPlan,
    ) -> HypothesisSearchCoverage:
        query_ids = {
            row.query_id
            for row in plan.queries
            if row.hypothesis_id == hypothesis.hypothesis_id
        }
        executions = [row for row in packet.executions if row.query_id in query_ids]
        successful = [row for row in executions if row.success]
        successful_query_ids = {row.query_id for row in successful}
        providers = {row.provider for row in successful}
        work_ids = {
            row.work_id
            for row in packet.works
            if bool(query_ids & set(row.retrieval_query_ids))
        }
        works = {row.work_id: row for row in packet.works}
        abstract_count = sum(bool(works[row].abstract) for row in work_ids)
        core = [row for row in reviews if row.importance == "core"] or reviews
        covered_core = sum(
            row.coverage.abstract_work_count
            >= self.policy.min_abstract_works_per_core_claim
            for row in core
        )
        sufficient = (
            len(successful_query_ids) >= self.policy.min_successful_queries_for_absence
            and len(work_ids) >= self.policy.min_unique_works_for_absence
            and abstract_count >= self.policy.min_abstract_works_for_absence
            and covered_core == len(core)
        )
        return HypothesisSearchCoverage(
            hypothesis_id=hypothesis.hypothesis_id,
            query_count=len(query_ids),
            successful_query_count=len(successful_query_ids),
            provider_success_count=len(providers),
            unique_work_count=len(work_ids),
            abstract_work_count=abstract_count,
            core_claim_count=len(core),
            core_claims_with_minimum_abstract_coverage=covered_core,
            sufficient_for_absence_based_novelty=sufficient,
        )

    def _status(
        self,
        reviews: list[ClaimPriorArtReview],
        coverage: HypothesisSearchCoverage,
    ) -> tuple[str, list[str], str]:
        core = [row for row in reviews if row.importance == "core"] or reviews
        statuses = [row.status for row in core]
        reasons: list[str] = []

        if not core:
            return (
                "INSUFFICIENT_SEARCH_EVIDENCE",
                ["no_core_novelty_claims"],
                "No core novelty claims were available for external prior-art assessment.",
            )

        if "CONFLICTING_PRIOR_ART" in statuses:
            reasons.append("core_claim_conflicting_prior_art")
            return (
                "CONFLICTING_PRIOR_ART",
                reasons,
                "At least one core novelty claim has high-confidence conflicting prior art in the reviewed search evidence.",
            )

        if all(row == "DIRECT_PRIOR_ART" for row in statuses):
            reasons.append("all_core_claims_have_direct_prior_art")
            return (
                "WELL_ESTABLISHED",
                reasons,
                "All core differentiating claims have direct prior-art matches in the reviewed evidence set.",
            )

        if all(row in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"} for row in statuses):
            reasons.append("core_claims_direct_or_partial_prior_art")
            return (
                "LITERATURE_SUPPORTED_EXTENSION",
                reasons,
                "The hypothesis is strongly adjacent to existing prior art: every core claim is directly or partially represented, but the full formulation is not uniformly reconstructed.",
            )

        if "INSUFFICIENT_METADATA" in statuses:
            reasons.append("core_claim_insufficient_metadata")
            return (
                "INSUFFICIENT_SEARCH_EVIDENCE",
                reasons,
                "At least one core claim could not be assessed from the retrieved metadata.",
            )

        if "TITLE_ONLY_NEIGHBORS" in statuses:
            reasons.append("core_claim_title_only_unresolved")
            return (
                "INSUFFICIENT_SEARCH_EVIDENCE",
                reasons,
                "At least one core claim is supported only by title-level neighboring evidence, so substantive relationship overlap remains unresolved.",
            )

        absence_dependent = any(
            row in {
                "NO_DIRECT_MATCH_FOUND",
                "COMPONENTS_ONLY",
            }
            for row in statuses
        )
        if absence_dependent and not coverage.sufficient_for_absence_based_novelty:
            reasons.append("insufficient_coverage_for_absence_based_status")
            return (
                "INSUFFICIENT_SEARCH_EVIDENCE",
                reasons,
                "The available search/abstract coverage is insufficient to interpret missing direct matches as evidence of external distinctness.",
            )

        relation_backed = any(
            row in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}
            for row in statuses
        )
        no_direct = any(row == "NO_DIRECT_MATCH_FOUND" for row in statuses)
        components_only = any(row == "COMPONENTS_ONLY" for row in statuses)

        if all(row == "NO_DIRECT_MATCH_FOUND" for row in statuses):
            reasons.append("no_direct_match_for_any_core_claim_under_minimum_coverage")
            return (
                "PLAUSIBLY_NOVEL",
                reasons,
                "No direct prior-art match was found for the core claims under the recorded minimum search coverage. This is search-bounded plausibility, not proof of literature-wide novelty.",
            )

        if relation_backed and (components_only or no_direct):
            reasons.append("known_relations_with_unmatched_composite_relation")
            return (
                "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                reasons,
                "The reviewed literature positively represents at least one core scientific relation, while at least one other core relation remains unmatched or only component-supported under minimum search coverage.",
            )

        if components_only and not relation_backed:
            reasons.append("known_components_without_relation_backed_core_claim")
            return (
                "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                reasons,
                "The reviewed literature establishes relevant components, variables, mechanisms, or contexts, but none of the core claimed relations is positively represented as direct or partial prior art under minimum search coverage.",
            )

        return (
            "INSUFFICIENT_SEARCH_EVIDENCE",
            ["unresolved_external_novelty_pattern"],
            "The claim-level prior-art pattern does not support a reliable external-novelty category under the current policy.",
        )

    def _validate_report_sources(
        self,
        portfolio: HypothesisPortfolio,
        plan: LiteratureQueryPlan,
        packet: PriorArtPacket,
    ) -> None:
        if plan.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("query plan source_portfolio_id mismatch")
        if packet.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("prior-art packet source_portfolio_id mismatch")
        if packet.source_query_plan_id != plan.plan_id:
            raise ValueError("prior-art packet source_query_plan_id mismatch")

    def compile_report_from_claim_reviews(
        self,
        portfolio: HypothesisPortfolio,
        plan: LiteratureQueryPlan,
        packet: PriorArtPacket,
        claim_reviews: Iterable[ClaimPriorArtReview],
        *,
        lineage: DiscoveryAxisSynthesisReport | None = None,
    ) -> ExternalNoveltyReport:
        """Compile a production ExternalNoveltyReport from frozen claim reviews.

        This seam is deterministic. It performs no retrieval, ranking, or LLM
        review. Claim-level review semantics and coverage remain authoritative;
        only hypothesis-level coverage/status/card/report assembly is recomputed
        from the supplied production artifacts.
        """
        self._validate_report_sources(portfolio, plan, packet)

        portfolio_ids = [row.hypothesis_id for row in portfolio.hypotheses]
        if len(portfolio_ids) != len(set(portfolio_ids)):
            raise ValueError("duplicate hypothesis_id in portfolio")
        portfolio_id_set = set(portfolio_ids)

        group_ids = [row.hypothesis_id for row in plan.claims]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("duplicate hypothesis claim group in query plan")
        if set(group_ids) != portfolio_id_set:
            missing = sorted(portfolio_id_set - set(group_ids))
            unexpected = sorted(set(group_ids) - portfolio_id_set)
            raise ValueError(
                "query-plan hypothesis claim groups do not match portfolio: "
                f"missing={missing}, unexpected={unexpected}"
            )

        planned_by_id: dict[str, object] = {}
        planned_order_by_hypothesis: dict[str, list[str]] = {}
        for group in plan.claims:
            ordered_ids: list[str] = []
            for claim in group.claims:
                if claim.hypothesis_id != group.hypothesis_id:
                    raise ValueError(
                        "query-plan claim hypothesis_id mismatch: "
                        f"claim={claim.claim_id}, claim_hypothesis={claim.hypothesis_id}, "
                        f"group_hypothesis={group.hypothesis_id}"
                    )
                if claim.claim_id in planned_by_id:
                    raise ValueError(f"duplicate planned claim_id: {claim.claim_id}")
                planned_by_id[claim.claim_id] = claim
                ordered_ids.append(claim.claim_id)
            planned_order_by_hypothesis[group.hypothesis_id] = ordered_ids

        reviews = list(claim_reviews)
        review_ids = [row.claim_id for row in reviews]
        if len(review_ids) != len(set(review_ids)):
            duplicates = sorted(
                claim_id for claim_id in set(review_ids)
                if review_ids.count(claim_id) > 1
            )
            raise ValueError(f"duplicate claim review IDs: {duplicates}")

        planned_ids = set(planned_by_id)
        supplied_ids = set(review_ids)
        if supplied_ids != planned_ids:
            missing = sorted(planned_ids - supplied_ids)
            unexpected = sorted(supplied_ids - planned_ids)
            raise ValueError(
                "claim review set does not exactly match query plan: "
                f"missing={missing}, unexpected={unexpected}"
            )

        review_by_id = {row.claim_id: row for row in reviews}
        for claim_id, planned in planned_by_id.items():
            review = review_by_id[claim_id]
            if review.hypothesis_id != planned.hypothesis_id:
                raise ValueError(
                    f"claim review hypothesis_id drift for {claim_id}: "
                    f"expected {planned.hypothesis_id}, got {review.hypothesis_id}"
                )
            if review.claim_text != planned.text:
                raise ValueError(f"claim review text drift for {claim_id}")
            if review.importance != planned.importance:
                raise ValueError(
                    f"claim review importance drift for {claim_id}: "
                    f"expected {planned.importance}, got {review.importance}"
                )
            if review.coverage.claim_id != review.claim_id:
                raise ValueError(
                    f"claim review coverage claim_id mismatch for {claim_id}: "
                    f"coverage={review.coverage.claim_id}"
                )

        lineages = {
            row.hypothesis_id: row
            for row in (lineage.lineages if lineage is not None else [])
        }
        cards: list[ExternalNoveltyCard] = []
        for hypothesis in portfolio.hypotheses:
            ordered_claim_ids = planned_order_by_hypothesis[hypothesis.hypothesis_id]
            rows = [review_by_id[claim_id] for claim_id in ordered_claim_ids]
            coverage = self._coverage(hypothesis, rows, packet, plan)
            status, reasons, interpretation = self._status(rows, coverage)

            strongest: list[tuple[float, str]] = []
            for review in rows:
                for match in review.matches:
                    if match.relationship in {
                        "DIRECT_PRIOR_ART",
                        "PARTIAL_PRIOR_ART",
                        "CONFLICTING_PRIOR_ART",
                    }:
                        strongest.append(
                            (
                                match.confidence * match.relevance_score,
                                match.work_id,
                            )
                        )
            strongest_ids = [
                row[1]
                for row in sorted(strongest, reverse=True)[:5]
            ]

            contextual_conflict_ids: list[str] = []
            seen_contextual: set[str] = set()
            for review in rows:
                for match in review.matches:
                    if (
                        match.relationship == "CONTEXTUAL_CONFLICT"
                        and match.work_id not in seen_contextual
                    ):
                        seen_contextual.add(match.work_id)
                        contextual_conflict_ids.append(match.work_id)

            lineage_row = lineages.get(hypothesis.hypothesis_id)
            limitations = [
                "Assessment is bounded by the recorded providers, queries, returned metadata, and ranking limits; it is not an exhaustive literature review.",
                "Most relationship judgments use title/abstract metadata rather than full text; title-only neighbors are not counted as partial/direct prior art in v1.1.",
                "Conflicting prior art must pass reaction-domain and catalyst/site-scope gates; out-of-scope counterexamples are retained only as contextual conflicts.",
                "Failure to retrieve a direct match is not proof that no prior art exists.",
                "External prior-art records are prior-art evidence only and are not eligible positive premises unless separately ingested through the scientific grounding pipeline.",
            ]
            cards.append(
                ExternalNoveltyCard(
                    hypothesis_id=hypothesis.hypothesis_id,
                    title=hypothesis.title,
                    status=status,
                    claim_reviews=rows,
                    coverage=coverage,
                    strongest_prior_art_work_ids=strongest_ids,
                    contextual_conflict_work_ids=contextual_conflict_ids[:5],
                    discovery_axis_id=(lineage_row.axis_id if lineage_row else None),
                    discovery_inspiration_id=(
                        lineage_row.inspiration_id if lineage_row else None
                    ),
                    reason_codes=reasons,
                    interpretation=interpretation,
                    search_limitations=limitations,
                )
            )

        counts = Counter(row.status for row in cards)
        report_id = _stable_id(
            "external_novelty_report",
            portfolio.portfolio_id,
            packet.packet_id,
            *[f"{row.hypothesis_id}:{row.status}" for row in cards],
        )
        body = {
            "schema_version": "external-novelty-report-v1",
            "report_id": report_id,
            "source_portfolio_id": portfolio.portfolio_id,
            "source_prior_art_packet_id": packet.packet_id,
            "searched_at_utc": packet.searched_at_utc,
            "cards": [row.model_dump(mode="json") for row in cards],
            "status_counts": dict(sorted(counts.items())),
            "policy": self.policy.model_dump(mode="json"),
            "external_novelty_claim_scope": (
                "search-bounded_prior-art_assessment_not_literature-wide_proof"
            ),
            "epistemic_usage": "prior_art_only_not_positive_premise",
        }
        return ExternalNoveltyReport(**body, report_sha256=_sha256_json(body))

    def assess(
        self,
        portfolio: HypothesisPortfolio,
        plan: LiteratureQueryPlan,
        packet: PriorArtPacket,
        *,
        lineage: DiscoveryAxisSynthesisReport | None = None,
    ) -> ExternalNoveltyReport:
        self._validate_report_sources(portfolio, plan, packet)

        claim_rows = {
            row.hypothesis_id: row
            for row in plan.claims
        }
        reviews: list[ClaimPriorArtReview] = []
        for hypothesis in portfolio.hypotheses:
            decomposition = claim_rows.get(hypothesis.hypothesis_id)
            if decomposition is None:
                raise ValueError(
                    f"query plan lacks claims for hypothesis {hypothesis.hypothesis_id}"
                )
            for claim in decomposition.claims:
                candidates = self.ranker.rank(claim, packet, plan)
                work_index = {row.work_id: row for row in packet.works}
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
                draft = self.review_backend.review_claim(claim, review_input)
                reviews.append(
                    self.compiler.compile(
                        claim,
                        candidates,
                        draft,
                        packet,
                        plan,
                    )
                )

        return self.compile_report_from_claim_reviews(
            portfolio,
            plan,
            packet,
            reviews,
            lineage=lineage,
        )
