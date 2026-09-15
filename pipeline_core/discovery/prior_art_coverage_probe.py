from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    ExternalNoveltyPolicy,
    HypothesisSearchCoverage,
    LiteratureQueryPlan,
    NoveltyClaim,
    PriorArtPacket,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisPortfolio,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PriorArtRankerProtocol(Protocol):
    def rank(
        self,
        claim: NoveltyClaim,
        packet: PriorArtPacket,
        plan: LiteratureQueryPlan,
    ) -> ClaimPriorArtCandidateSet: ...


class PreReviewClaimCoverage(StrictModel):
    schema_version: Literal[
        "pre-review-claim-coverage-v1"
    ] = "pre-review-claim-coverage-v1"

    hypothesis_id: str
    claim_id: str
    importance: Literal["core", "supporting"]

    query_count: int = Field(ge=0)
    successful_query_count: int = Field(ge=0)

    ranked_work_count: int = Field(ge=0)
    abstract_work_count: int = Field(ge=0)

    minimum_abstract_work_count: int = Field(ge=0)
    passes_minimum_abstract_coverage: bool

    ranked_work_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_claim_coverage(
        self,
    ) -> "PreReviewClaimCoverage":
        if len(self.ranked_work_ids) != len(set(self.ranked_work_ids)):
            raise ValueError(
                "duplicate ranked work ID in pre-review claim coverage"
            )
        if self.ranked_work_count != len(self.ranked_work_ids):
            raise ValueError(
                "ranked_work_count does not match ranked_work_ids"
            )
        if self.abstract_work_count > self.ranked_work_count:
            raise ValueError(
                "abstract_work_count cannot exceed ranked_work_count"
            )
        expected = (
            self.abstract_work_count
            >= self.minimum_abstract_work_count
        )
        if self.passes_minimum_abstract_coverage != expected:
            raise ValueError(
                "claim minimum-abstract coverage flag drift"
            )
        return self


class PreReviewHypothesisCoverage(StrictModel):
    schema_version: Literal[
        "pre-review-hypothesis-coverage-v1"
    ] = "pre-review-hypothesis-coverage-v1"

    hypothesis_id: str

    coverage: HypothesisSearchCoverage

    claim_coverages: list[
        PreReviewClaimCoverage
    ] = Field(default_factory=list)

    core_claim_ids: list[str] = Field(default_factory=list)

    gate_decision: Literal[
        "KEEP_FOR_CLAIM_REVIEW",
        "EVIDENCE_REQUIRED",
    ]
    reason_codes: list[str] = Field(default_factory=list)

    novelty_authority: Literal[False] = False
    claim_review_llm_performed: Literal[False] = False
    semantic_distinctiveness_performed: Literal[False] = False
    n10_review_performed: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_hypothesis_coverage(
        self,
    ) -> "PreReviewHypothesisCoverage":
        claim_ids = [
            row.claim_id
            for row in self.claim_coverages
        ]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(
                "duplicate claim ID in pre-review hypothesis coverage"
            )

        expected_decision = (
            "KEEP_FOR_CLAIM_REVIEW"
            if self.coverage.sufficient_for_absence_based_novelty
            else "EVIDENCE_REQUIRED"
        )
        if self.gate_decision != expected_decision:
            raise ValueError(
                "pre-review gate decision does not match coverage"
            )

        known_ids = set(claim_ids)
        if not set(self.core_claim_ids).issubset(known_ids):
            raise ValueError(
                "core_claim_ids reference unknown claim coverage"
            )

        return self


class PreReviewCoverageReport(StrictModel):
    schema_version: Literal[
        "pre-review-coverage-report-v1"
    ] = "pre-review-coverage-report-v1"

    source_portfolio_id: str
    source_query_plan_id: str
    source_query_plan_sha256: str
    source_prior_art_packet_id: str
    source_prior_art_packet_sha256: str

    policy: ExternalNoveltyPolicy

    hypotheses: list[
        PreReviewHypothesisCoverage
    ] = Field(default_factory=list)

    keep_for_claim_review_count: int = Field(ge=0)
    evidence_required_count: int = Field(ge=0)

    claim_review_llm_performed: Literal[False] = False
    prior_art_relationship_classification_performed: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(
        self,
    ) -> "PreReviewCoverageReport":
        ids = [
            row.hypothesis_id
            for row in self.hypotheses
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate hypothesis ID in pre-review coverage report"
            )

        keep = sum(
            row.gate_decision == "KEEP_FOR_CLAIM_REVIEW"
            for row in self.hypotheses
        )
        evidence = sum(
            row.gate_decision == "EVIDENCE_REQUIRED"
            for row in self.hypotheses
        )
        if keep != self.keep_for_claim_review_count:
            raise ValueError(
                "keep_for_claim_review_count mismatch"
            )
        if evidence != self.evidence_required_count:
            raise ValueError(
                "evidence_required_count mismatch"
            )

        return self


class PriorArtPreReviewCoverageProbe:
    """
    Deterministic evidence-sufficiency gate before claim-review LLM calls.

    It uses only:
      - frozen query plan,
      - retrieved prior-art packet and query executions,
      - deterministic/ranker candidate sets,
      - ExternalNoveltyPolicy coverage thresholds.

    It does NOT classify prior-art relationships, infer novelty, run semantic
    distinctiveness, run N10, or change production selection.
    """

    def __init__(
        self,
        ranker: PriorArtRankerProtocol,
        *,
        policy: ExternalNoveltyPolicy | None = None,
    ) -> None:
        self.ranker = ranker
        self.policy = policy or ExternalNoveltyPolicy()

    def _validate_sources(
        self,
        portfolio: HypothesisPortfolio,
        plan: LiteratureQueryPlan,
        packet: PriorArtPacket,
    ) -> None:
        if plan.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError(
                "pre-review probe query-plan portfolio mismatch"
            )
        if packet.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError(
                "pre-review probe prior-art portfolio mismatch"
            )
        if packet.source_query_plan_id != plan.plan_id:
            raise ValueError(
                "pre-review probe prior-art/query-plan mismatch"
            )

        portfolio_ids = [
            row.hypothesis_id
            for row in portfolio.hypotheses
        ]
        if len(portfolio_ids) != len(set(portfolio_ids)):
            raise ValueError(
                "duplicate hypothesis ID in source portfolio"
            )

        group_ids = [
            row.hypothesis_id
            for row in plan.claims
        ]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(
                "duplicate hypothesis claim group in query plan"
            )

        if set(group_ids) != set(portfolio_ids):
            raise ValueError(
                "query-plan hypothesis groups do not match portfolio"
            )

        packet_work_ids = [
            row.work_id
            for row in packet.works
        ]
        if len(packet_work_ids) != len(set(packet_work_ids)):
            raise ValueError(
                "duplicate work ID in prior-art packet"
            )

    def _claim_coverage(
        self,
        *,
        claim: NoveltyClaim,
        packet: PriorArtPacket,
        plan: LiteratureQueryPlan,
    ) -> PreReviewClaimCoverage:
        candidates = self.ranker.rank(
            claim,
            packet,
            plan,
        )

        if candidates.hypothesis_id != claim.hypothesis_id:
            raise ValueError(
                "ranked candidate-set hypothesis_id mismatch"
            )
        if candidates.claim_id != claim.claim_id:
            raise ValueError(
                "ranked candidate-set claim_id mismatch"
            )

        ranked_ids = [
            row.work_id
            for row in candidates.ranked_works
        ]
        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError(
                "duplicate ranked work ID in candidate set"
            )

        works = {
            row.work_id: row
            for row in packet.works
        }
        unknown_ids = sorted(
            set(ranked_ids) - set(works)
        )
        if unknown_ids:
            raise ValueError(
                "ranked candidate set references unknown prior-art "
                f"work IDs: {unknown_ids}"
            )

        claim_query_ids = {
            row.query_id
            for row in plan.queries
            if row.claim_id == claim.claim_id
        }
        successful_query_ids = {
            row.query_id
            for row in packet.executions
            if (
                row.success
                and row.query_id in claim_query_ids
            )
        }

        abstract_count = sum(
            bool(works[work_id].abstract)
            for work_id in ranked_ids
        )

        return PreReviewClaimCoverage(
            hypothesis_id=claim.hypothesis_id,
            claim_id=claim.claim_id,
            importance=claim.importance,
            query_count=len(claim_query_ids),
            successful_query_count=len(
                successful_query_ids
            ),
            ranked_work_count=len(ranked_ids),
            abstract_work_count=abstract_count,
            minimum_abstract_work_count=(
                self.policy
                .min_abstract_works_per_core_claim
            ),
            passes_minimum_abstract_coverage=(
                abstract_count
                >= self.policy
                .min_abstract_works_per_core_claim
            ),
            ranked_work_ids=ranked_ids,
        )

    def _hypothesis_coverage(
        self,
        *,
        hypothesis_id: str,
        claim_coverages: list[
            PreReviewClaimCoverage
        ],
        packet: PriorArtPacket,
        plan: LiteratureQueryPlan,
    ) -> PreReviewHypothesisCoverage:
        query_ids = {
            row.query_id
            for row in plan.queries
            if row.hypothesis_id == hypothesis_id
        }
        successful = [
            row
            for row in packet.executions
            if (
                row.success
                and row.query_id in query_ids
            )
        ]
        successful_query_ids = {
            row.query_id
            for row in successful
        }
        providers = {
            row.provider
            for row in successful
        }

        work_ids = {
            row.work_id
            for row in packet.works
            if bool(
                query_ids
                & set(row.retrieval_query_ids)
            )
        }
        works = {
            row.work_id: row
            for row in packet.works
        }
        abstract_count = sum(
            bool(works[work_id].abstract)
            for work_id in work_ids
        )

        core = [
            row
            for row in claim_coverages
            if row.importance == "core"
        ] or list(claim_coverages)

        covered_core = sum(
            row.abstract_work_count
            >= self.policy
            .min_abstract_works_per_core_claim
            for row in core
        )

        sufficient = (
            len(successful_query_ids)
            >= self.policy
            .min_successful_queries_for_absence
            and len(work_ids)
            >= self.policy
            .min_unique_works_for_absence
            and abstract_count
            >= self.policy
            .min_abstract_works_for_absence
            and bool(core)
            and covered_core == len(core)
        )

        reason_codes: list[str] = []
        if (
            len(successful_query_ids)
            < self.policy
            .min_successful_queries_for_absence
        ):
            reason_codes.append(
                "successful_query_count_below_minimum"
            )
        if (
            len(work_ids)
            < self.policy
            .min_unique_works_for_absence
        ):
            reason_codes.append(
                "unique_work_count_below_minimum"
            )
        if (
            abstract_count
            < self.policy
            .min_abstract_works_for_absence
        ):
            reason_codes.append(
                "abstract_work_count_below_minimum"
            )
        if not core:
            reason_codes.append(
                "no_claims_available_for_coverage_gate"
            )
        elif covered_core != len(core):
            reason_codes.append(
                "one_or_more_core_claims_below_abstract_minimum"
            )

        coverage = HypothesisSearchCoverage(
            hypothesis_id=hypothesis_id,
            query_count=len(query_ids),
            successful_query_count=len(
                successful_query_ids
            ),
            provider_success_count=len(providers),
            unique_work_count=len(work_ids),
            abstract_work_count=abstract_count,
            core_claim_count=len(core),
            core_claims_with_minimum_abstract_coverage=(
                covered_core
            ),
            sufficient_for_absence_based_novelty=(
                sufficient
            ),
        )

        return PreReviewHypothesisCoverage(
            hypothesis_id=hypothesis_id,
            coverage=coverage,
            claim_coverages=claim_coverages,
            core_claim_ids=[
                row.claim_id
                for row in core
            ],
            gate_decision=(
                "KEEP_FOR_CLAIM_REVIEW"
                if sufficient
                else "EVIDENCE_REQUIRED"
            ),
            reason_codes=reason_codes,
        )

    def build(
        self,
        portfolio: HypothesisPortfolio,
        plan: LiteratureQueryPlan,
        packet: PriorArtPacket,
    ) -> PreReviewCoverageReport:
        self._validate_sources(
            portfolio,
            plan,
            packet,
        )

        group_by_hypothesis = {
            row.hypothesis_id: row
            for row in plan.claims
        }

        hypothesis_rows: list[
            PreReviewHypothesisCoverage
        ] = []

        for hypothesis in portfolio.hypotheses:
            group = group_by_hypothesis[
                hypothesis.hypothesis_id
            ]

            claim_ids = [
                row.claim_id
                for row in group.claims
            ]
            if len(claim_ids) != len(set(claim_ids)):
                raise ValueError(
                    "duplicate claim ID in hypothesis query-plan group"
                )

            for claim in group.claims:
                if (
                    claim.hypothesis_id
                    != hypothesis.hypothesis_id
                ):
                    raise ValueError(
                        "query-plan claim hypothesis_id mismatch"
                    )

            claim_coverages = [
                self._claim_coverage(
                    claim=claim,
                    packet=packet,
                    plan=plan,
                )
                for claim in group.claims
            ]

            hypothesis_rows.append(
                self._hypothesis_coverage(
                    hypothesis_id=(
                        hypothesis.hypothesis_id
                    ),
                    claim_coverages=(
                        claim_coverages
                    ),
                    packet=packet,
                    plan=plan,
                )
            )

        return PreReviewCoverageReport(
            source_portfolio_id=portfolio.portfolio_id,
            source_query_plan_id=plan.plan_id,
            source_query_plan_sha256=plan.plan_sha256,
            source_prior_art_packet_id=packet.packet_id,
            source_prior_art_packet_sha256=(
                packet.packet_sha256
            ),
            policy=self.policy,
            hypotheses=hypothesis_rows,
            keep_for_claim_review_count=sum(
                row.gate_decision
                == "KEEP_FOR_CLAIM_REVIEW"
                for row in hypothesis_rows
            ),
            evidence_required_count=sum(
                row.gate_decision
                == "EVIDENCE_REQUIRED"
                for row in hypothesis_rows
            ),
        )
