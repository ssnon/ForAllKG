from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

from pipeline_core.discovery.discovery_axis_contracts import DiscoveryAxisSynthesisReport
from pipeline_core.discovery.diagnostic_prior_art_review import (
    DiagnosticClaimPriorArtReview,
    compile_diagnostic_prior_art_review,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ExternalNoveltyCard,
    ExternalNoveltyPolicy,
    ExternalNoveltyReport,
    HypothesisSearchCoverage,
    HypothesisNoveltyDepthProfile,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio
from pipeline_core.discovery.novelty_claim_decomposition import NoveltyClaimDecomposer
from pipeline_core.discovery.prior_art_matching import (
    ClaimPriorArtCompiler,
    ClaimReviewBackend,
    PriorArtRanker,
)
from pipeline_core.discovery.prior_art_review_audit import (
    prior_art_review_audit_scope,
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


def _lower_order_gap_annotation(
    reviews: list[ClaimPriorArtReview],
    coverage,
    claims_by_id: dict[str, Any],
    *,
    gap_absence_sufficient: bool | None = None,
    diagnostic_lower_order_signals_by_claim: (
        dict[str, list[str]] | None
    ) = None,
) -> tuple[
    str,
    list[str],
    list[str],
    list[str],
]:
    """Summarize reviewed lower-order evidence without changing novelty status.

    LOWER_ORDER_RELATION_PRIOR_ART already asserts that a reviewed record
    establishes a nontrivial lower-order subrelation of the full claim.
    This helper preserves that reviewed evidence independently of whether
    the claim is structurally eligible for a higher-order gap annotation.

    HIGHER_ORDER_RELATIONAL_GAP is search-bounded and is emitted only when:
      * a core claim has abstract-backed lower-order prior art,
      * the planned claim is an explicit novelty-bearing composite,
      * it has exact higher-order provenance and explicit component topology,
      * that full claim remains COMPONENTS_ONLY, and
      * hypothesis-level absence coverage is sufficient.

    The annotation is descriptive only. It does not alter _status(),
    rejection, re-axis, or refinement policy.
    """

    supported_core_claim_ids: list[str] = []
    gap_claim_ids: list[str] = []

    core_work_ids: list[str] = []
    seen_core_work_ids: set[str] = set()

    for review in reviews:
        if review.importance != "core":
            continue

        if (
            diagnostic_lower_order_signals_by_claim
            is None
        ):
            lower_work_ids = [
                match.work_id
                for match in review.matches
                if (
                    match.relationship
                    == "LOWER_ORDER_RELATION_PRIOR_ART"
                )
            ]
        else:
            lower_work_ids = list(
                diagnostic_lower_order_signals_by_claim.get(
                    review.claim_id,
                    [],
                )
            )

        if not lower_work_ids:
            continue

        supported_core_claim_ids.append(
            review.claim_id
        )

        for work_id in lower_work_ids:
            if work_id in seen_core_work_ids:
                continue

            seen_core_work_ids.add(
                work_id
            )

            core_work_ids.append(
                work_id
            )

        claim = claims_by_id.get(
            review.claim_id
        )

        structurally_eligible = bool(
            claim is not None
            and getattr(
                claim,
                "importance",
                None,
            )
            == "core"
            and getattr(
                claim,
                "kind",
                None,
            )
            == "composite"
            and getattr(
                claim,
                "higher_order_relation_basis",
                None,
            )
            and getattr(
                claim,
                "higher_order_component_claim_ids",
                None,
            )
            and getattr(
                claim,
                "novelty_selection_role",
                None,
            )
            == "NOVELTY_BEARING"
        )

        if (
            structurally_eligible
            and review.status
            == "COMPONENTS_ONLY"
        ):
            gap_claim_ids.append(
                review.claim_id
            )

    if gap_absence_sufficient is None:
        gap_absence_sufficient = (
            coverage.sufficient_for_absence_based_novelty
        )

    if (
        gap_claim_ids
        and gap_absence_sufficient
    ):
        relational_gap_kind = (
            "HIGHER_ORDER_RELATIONAL_GAP"
        )
    else:
        relational_gap_kind = "NONE"

    return (
        relational_gap_kind,
        supported_core_claim_ids,
        gap_claim_ids,
        core_work_ids,
    )


_RELATION_BACKED_CLAIM_STATUSES = frozenset({
    "DIRECT_PRIOR_ART",
    "PARTIAL_PRIOR_ART",
})

_GAP_LIKE_CLAIM_STATUSES = frozenset({
    "COMPONENTS_ONLY",
    "NO_DIRECT_MATCH_FOUND",
})


def _novelty_depth_profile(
    *,
    hypothesis_id: str,
    reviews: list[ClaimPriorArtReview],
    claims_by_id: dict[str, Any],
) -> HypothesisNoveltyDepthProfile:
    core_reviews = [
        review
        for review in reviews
        if review.importance == "core"
    ] or list(reviews)

    relation_backed_core = [
        review
        for review in core_reviews
        if review.status in _RELATION_BACKED_CLAIM_STATUSES
    ]

    core_count = len(core_reviews)
    known_fraction = (
        len(relation_backed_core) / core_count
        if core_count
        else 0.0
    )

    role_binding_complete = True
    novelty_bearing_reviews: list[ClaimPriorArtReview] = []

    for review in core_reviews:
        claim = claims_by_id.get(review.claim_id)

        if claim is None or claim.hypothesis_id != hypothesis_id:
            raise ValueError(
                "S25b novelty-depth profile claim provenance drift: "
                + review.claim_id
            )

        role = getattr(claim, "novelty_selection_role", None)

        if role is None:
            role_binding_complete = False
            continue

        if role == "NOVELTY_BEARING":
            novelty_bearing_reviews.append(review)

    novelty_bearing_ids = [
        review.claim_id
        for review in novelty_bearing_reviews
    ]

    relation_backed_ids = [
        review.claim_id
        for review in novelty_bearing_reviews
        if review.status in _RELATION_BACKED_CLAIM_STATUSES
    ]

    gap_like_ids = [
        review.claim_id
        for review in novelty_bearing_reviews
        if review.status in _GAP_LIKE_CLAIM_STATUSES
    ]

    conflicting_ids = [
        review.claim_id
        for review in novelty_bearing_reviews
        if review.status == "CONFLICTING_PRIOR_ART"
    ]

    classified_ids = (
        set(relation_backed_ids)
        | set(gap_like_ids)
        | set(conflicting_ids)
    )

    unresolved_ids = [
        claim_id
        for claim_id in novelty_bearing_ids
        if claim_id not in classified_ids
    ]

    novelty_count = len(novelty_bearing_ids)

    relation_backed_fraction = (
        len(relation_backed_ids) / novelty_count
        if novelty_count
        else 0.0
    )

    if not role_binding_complete:
        gap_centrality = "UNRESOLVED"
    elif novelty_count == 0:
        gap_centrality = "NONE"
    elif novelty_count == 1:
        gap_centrality = "CENTRAL"
    else:
        gap_centrality = "DISTRIBUTED"

    if novelty_count == 0:
        prior_art_state = "NONE"
    elif conflicting_ids:
        prior_art_state = "CONFLICTING"
    elif unresolved_ids:
        prior_art_state = "UNRESOLVED"
    elif len(relation_backed_ids) == novelty_count:
        prior_art_state = "ALL_RELATION_BACKED"
    elif len(gap_like_ids) == novelty_count:
        prior_art_state = "ALL_GAP_LIKE"
    else:
        prior_art_state = "MIXED"

    reason_codes: list[str] = [
        "s25b_diagnostic_only",
        "s25b_existing_novelty_selection_roles_only",
    ]

    if not role_binding_complete:
        reason_codes.append(
            "s25b_incomplete_core_role_binding"
        )

    if novelty_count == 0:
        reason_codes.append(
            "s25b_no_explicit_novelty_bearing_core_claim"
        )
    elif novelty_count == 1:
        reason_codes.append(
            "s25b_single_explicit_novelty_bearing_core_claim"
        )
    else:
        reason_codes.append(
            "s25b_multiple_explicit_novelty_bearing_core_claims"
        )

    if relation_backed_ids:
        reason_codes.append(
            "s25b_novelty_bearing_relation_backed_prior_art_present"
        )

    if gap_like_ids:
        reason_codes.append(
            "s25b_novelty_bearing_gap_like_claim_present"
        )

    return HypothesisNoveltyDepthProfile(
        hypothesis_id=hypothesis_id,
        role_binding_complete=role_binding_complete,
        core_claim_count=core_count,
        relation_backed_core_claim_count=len(relation_backed_core),
        known_core_relation_fraction=float(known_fraction),
        novelty_bearing_claim_ids=novelty_bearing_ids,
        novelty_bearing_claim_count=novelty_count,
        novelty_bearing_relation_backed_claim_ids=relation_backed_ids,
        novelty_bearing_gap_like_claim_ids=gap_like_ids,
        novelty_bearing_conflicting_claim_ids=conflicting_ids,
        novelty_bearing_unresolved_claim_ids=unresolved_ids,
        novelty_bearing_relation_backed_fraction=float(
            relation_backed_fraction
        ),
        gap_centrality=gap_centrality,
        novelty_bearing_prior_art_state=prior_art_state,
        reason_codes=sorted(set(reason_codes)),
    )


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

    def _relational_gap_absence_sufficient(
        self,
        reviews: list[ClaimPriorArtReview],
        coverage: HypothesisSearchCoverage,
        claims_by_id: dict[str, Any],
    ) -> bool:
        # This authority is only for HIGHER_ORDER_RELATIONAL_GAP.
        # Global hypothesis status keeps the existing all-core coverage
        # contract. If any core claim lacks an explicit novelty-selection
        # role, fail closed to the existing all-core coverage result.

        core = [
            row
            for row in reviews
            if row.importance == "core"
        ] or reviews

        role_bound: list[
            tuple[ClaimPriorArtReview, str]
        ] = []

        for review in core:
            claim = claims_by_id.get(
                review.claim_id
            )

            role = (
                getattr(
                    claim,
                    "novelty_selection_role",
                    None,
                )
                if claim is not None
                else None
            )

            if role is None:
                return (
                    coverage
                    .sufficient_for_absence_based_novelty
                )

            role_bound.append(
                (
                    review,
                    role,
                )
            )

        novelty_bearing = [
            review
            for review, role in role_bound
            if role == "NOVELTY_BEARING"
        ]

        if not novelty_bearing:
            return False

        covered_novelty_bearing = sum(
            review.coverage.abstract_work_count
            >= self.policy.min_abstract_works_per_core_claim
            for review in novelty_bearing
        )

        return bool(
            coverage.successful_query_count
            >= self.policy.min_successful_queries_for_absence
            and coverage.unique_work_count
            >= self.policy.min_unique_works_for_absence
            and coverage.abstract_work_count
            >= self.policy.min_abstract_works_for_absence
            and covered_novelty_bearing
            == len(
                novelty_bearing
            )
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

        relation_backed_count = sum(
            row in {"DIRECT_PRIOR_ART", "PARTIAL_PRIOR_ART"}
            for row in statuses
        )
        relation_backed = relation_backed_count > 0
        no_direct = any(row == "NO_DIRECT_MATCH_FOUND" for row in statuses)
        components_only = any(row == "COMPONENTS_ONLY" for row in statuses)

        # A strict majority of positively relation-backed core claims is
        # sufficient to classify the hypothesis as literature-supported
        # even when a remaining claim is only component-supported.
        #
        # This classification does not infer novelty from a missing match;
        # therefore it must not depend on absence-coverage sufficiency.
        #
        # NO_DIRECT_MATCH_FOUND remains absence-dependent and is deliberately
        # excluded from this early positive-evidence branch.
        if (
            relation_backed
            and components_only
            and not no_direct
            and relation_backed_count * 2 > len(statuses)
        ):
            reasons.append(
                "majority_core_relations_have_direct_or_partial_prior_art"
            )
            return (
                "LITERATURE_SUPPORTED_EXTENSION",
                reasons,
                "A strict majority of the core differentiating relations are directly or partially represented in the reviewed literature, while the remaining core claims are component-supported rather than externally unmatched.",
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

        if all(row == "NO_DIRECT_MATCH_FOUND" for row in statuses):
            reasons.append("no_direct_match_for_any_core_claim_under_minimum_coverage")
            return (
                "PLAUSIBLY_NOVEL",
                reasons,
                "No direct prior-art match was found for the core claims under the recorded minimum search coverage. This is search-bounded plausibility, not proof of literature-wide novelty.",
            )

        if relation_backed and no_direct:
            reasons.append("known_relations_with_unmatched_composite_relation")
            return (
                "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                reasons,
                "The reviewed literature positively represents at least one core scientific relation, while at least one other core relation has no direct match under minimum search coverage.",
            )

        if relation_backed and components_only:
            strict_majority_relation_backed = (
                relation_backed_count * 2 > len(statuses)
            )
            if strict_majority_relation_backed:
                reasons.append(
                    "majority_core_relations_have_direct_or_partial_prior_art"
                )
                return (
                    "LITERATURE_SUPPORTED_EXTENSION",
                    reasons,
                    "A strict majority of the core differentiating relations are directly or partially represented in the reviewed literature, while the remaining core claims are component-supported rather than externally unmatched.",
                )

            reasons.append("known_relations_with_component_supported_gap")
            return (
                "NEW_COMBINATION_OF_KNOWN_EFFECTS",
                reasons,
                "The reviewed literature positively represents some core scientific relations, but a majority of the core differentiating relations are not directly or partially represented and remain component-supported.",
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

    def _diagnostic_lower_order_signal_map(
        self,
        diagnostic_reviews: (
            Iterable[DiagnosticClaimPriorArtReview]
            | None
        ),
        claims_by_id: dict[str, Any],
        *,
        diagnostic_plan: LiteratureQueryPlan | None,
        diagnostic_packet: PriorArtPacket | None,
    ) -> dict[str, list[str]] | None:
        if diagnostic_reviews is None:
            if (
                diagnostic_plan is not None
                or diagnostic_packet is not None
            ):
                raise ValueError(
                    "diagnostic plan/packet supplied without "
                    "diagnostic reviews"
                )
            return None

        rows = list(
            diagnostic_reviews
        )

        if not rows:
            if (
                diagnostic_plan is not None
                and diagnostic_packet is None
            ) or (
                diagnostic_plan is None
                and diagnostic_packet is not None
            ):
                raise ValueError(
                    "diagnostic plan and packet must be "
                    "supplied together"
                )
            return {}

        if (
            diagnostic_plan is None
            or diagnostic_packet is None
        ):
            raise ValueError(
                "non-empty diagnostic reviews require "
                "diagnostic plan and packet"
            )

        if (
            diagnostic_packet.source_portfolio_id
            != diagnostic_plan.source_portfolio_id
        ):
            raise ValueError(
                "diagnostic plan/packet source_portfolio_id "
                "mismatch"
            )

        if (
            diagnostic_packet.source_query_plan_id
            != diagnostic_plan.plan_id
        ):
            raise ValueError(
                "diagnostic plan/packet query-plan provenance "
                "mismatch"
            )

        if any(
            row.query_kind != "claim_diagnostic"
            for row in diagnostic_plan.queries
        ):
            raise ValueError(
                "diagnostic plan contains non-diagnostic query"
            )

        claim_ids = [
            row.claim_id
            for row in rows
        ]

        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(
                "duplicate diagnostic claim review IDs"
            )

        packet_works = {
            row.work_id: row
            for row in diagnostic_packet.works
        }
        projected: dict[str, list[str]] = {}

        for review in rows:
            planned = claims_by_id.get(
                review.claim_id
            )

            if planned is None:
                raise ValueError(
                    "diagnostic review references unplanned "
                    f"claim_id: {review.claim_id}"
                )

            if review.hypothesis_id != planned.hypothesis_id:
                raise ValueError(
                    "diagnostic review hypothesis_id drift "
                    f"for {review.claim_id}"
                )

            if review.claim_text != planned.text:
                raise ValueError(
                    "diagnostic review text drift "
                    f"for {review.claim_id}"
                )

            if (
                review.diagnostic_query_kind
                != planned.diagnostic_query_kind
            ):
                raise ValueError(
                    "diagnostic review kind drift "
                    f"for {review.claim_id}"
                )

            expected_query = str(
                planned.diagnostic_execution_query
                or ""
            ).strip()

            if (
                review.diagnostic_execution_query
                != expected_query
            ):
                raise ValueError(
                    "diagnostic review execution-query drift "
                    f"for {review.claim_id}"
                )

            signal_ids = list(
                review.signal_work_ids
            )

            if len(signal_ids) != len(set(signal_ids)):
                raise ValueError(
                    "duplicate diagnostic signal work IDs "
                    f"for {review.claim_id}"
                )

            match_by_id = {
                match.work_id:
                    match
                for match in review.matches
            }

            for work_id in signal_ids:
                match = match_by_id.get(
                    work_id
                )

                if match is None:
                    raise ValueError(
                        "diagnostic signal lacks compiled match "
                        f"for {review.claim_id}: {work_id}"
                    )

                if (
                    review.diagnostic_query_kind
                    == "LOWER_ORDER_RELATION"
                    and match.relationship
                    != "LOWER_ORDER_RELATION_PRIOR_ART"
                ):
                    raise ValueError(
                        "lower-order diagnostic signal has "
                        "wrong relationship for "
                        f"{review.claim_id}: {work_id}"
                    )

                work = packet_works.get(
                    work_id
                )

                if (
                    work is None
                    or not work.abstract
                    or not match.abstract_available
                ):
                    raise ValueError(
                        "diagnostic signal lacks abstract-backed "
                        "packet evidence for "
                        f"{review.claim_id}: {work_id}"
                    )

            if (
                review.diagnostic_query_kind
                == "LOWER_ORDER_RELATION"
            ):
                projected[
                    review.claim_id
                ] = signal_ids

        return projected

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
        diagnostic_reviews: (
            Iterable[DiagnosticClaimPriorArtReview]
            | None
        ) = None,
        diagnostic_plan: LiteratureQueryPlan | None = None,
        diagnostic_packet: PriorArtPacket | None = None,
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

        diagnostic_lower_order_signals_by_claim = (
            self._diagnostic_lower_order_signal_map(
                diagnostic_reviews,
                planned_by_id,
                diagnostic_plan=diagnostic_plan,
                diagnostic_packet=diagnostic_packet,
            )
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
            novelty_depth_profile = _novelty_depth_profile(
                hypothesis_id=hypothesis.hypothesis_id,
                reviews=rows,
                claims_by_id=planned_by_id,
            )
            gap_absence_sufficient = (
                self._relational_gap_absence_sufficient(
                    rows,
                    coverage,
                    planned_by_id,
                )
            )

            (
                relational_gap_kind,
                lower_order_supported_core_claim_ids,
                higher_order_relational_gap_claim_ids,
                lower_order_core_prior_art_work_ids,
            ) = _lower_order_gap_annotation(
                rows,
                coverage,
                planned_by_id,
                gap_absence_sufficient=(
                    gap_absence_sufficient
                ),
                diagnostic_lower_order_signals_by_claim=(
                    diagnostic_lower_order_signals_by_claim
                ),
            )

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
            lower_order_ids: list[str] = []
            directional_counterevidence_ids: list[str] = []

            seen_contextual: set[str] = set()
            seen_lower_order: set[str] = set()
            seen_directional_counterevidence: set[str] = set()
            for review in rows:
                for match in review.matches:
                    if (
                        match.relationship == "CONTEXTUAL_CONFLICT"
                        and match.work_id not in seen_contextual
                    ):
                        seen_contextual.add(match.work_id)
                        contextual_conflict_ids.append(match.work_id)

                    if (
                        diagnostic_lower_order_signals_by_claim
                        is None
                        and match.relationship
                        == "LOWER_ORDER_RELATION_PRIOR_ART"
                        and match.work_id not in seen_lower_order
                    ):
                        seen_lower_order.add(match.work_id)
                        lower_order_ids.append(match.work_id)

                    if (
                        match.relationship
                        == "DIRECTIONAL_COUNTEREVIDENCE"
                        and match.work_id
                        not in seen_directional_counterevidence
                    ):
                        seen_directional_counterevidence.add(
                            match.work_id
                        )
                        directional_counterevidence_ids.append(
                            match.work_id
                        )

            if (
                diagnostic_lower_order_signals_by_claim
                is not None
            ):
                for claim_id in ordered_claim_ids:
                    for work_id in (
                        diagnostic_lower_order_signals_by_claim.get(
                            claim_id,
                            [],
                        )
                    ):
                        if work_id in seen_lower_order:
                            continue

                        seen_lower_order.add(
                            work_id
                        )
                        lower_order_ids.append(
                            work_id
                        )

            if lower_order_ids:
                reasons.append(
                    "lower_order_relation_prior_art_present"
                )

            if directional_counterevidence_ids:
                reasons.append(
                    "directional_counterevidence_present"
                )

            if (
                relational_gap_kind
                == "HIGHER_ORDER_RELATIONAL_GAP"
            ):
                reasons.append(
                    "higher_order_relational_gap_present"
                )

            reasons = sorted(set(reasons))

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
                    novelty_depth_profile=novelty_depth_profile,
                    strongest_prior_art_work_ids=strongest_ids,
                    contextual_conflict_work_ids=contextual_conflict_ids[:5],
                    lower_order_prior_art_work_ids=lower_order_ids[:5],
                    lower_order_supported_core_claim_ids=(
                        lower_order_supported_core_claim_ids
                    ),
                    higher_order_relational_gap_claim_ids=(
                        higher_order_relational_gap_claim_ids
                    ),
                    lower_order_core_prior_art_work_ids=(
                        lower_order_core_prior_art_work_ids
                    ),
                    lower_order_core_unique_work_count=(
                        len(lower_order_core_prior_art_work_ids)
                    ),
                    relational_gap_kind=relational_gap_kind,
                    directional_counterevidence_work_ids=(
                        directional_counterevidence_ids[:5]
                    ),
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

    def review_diagnostic_prior_art(
        self,
        plan: LiteratureQueryPlan,
        packet: PriorArtPacket,
    ) -> list[DiagnosticClaimPriorArtReview]:
        # Run only the bounded diagnostic review lane.
        # This lane cannot set ordinary full-claim status. Its compiled
        # signals are consumed only by diagnostic-aware report assembly.

        if (
            packet.source_portfolio_id
            != plan.source_portfolio_id
        ):
            raise ValueError(
                "diagnostic plan/packet "
                "source_portfolio_id mismatch"
            )

        if (
            packet.source_query_plan_id
            != plan.plan_id
        ):
            raise ValueError(
                "diagnostic plan/packet "
                "query-plan provenance mismatch"
            )

        if any(
            row.query_kind
            != "claim_diagnostic"
            for row in plan.queries
        ):
            raise ValueError(
                "diagnostic review accepts "
                "claim_diagnostic queries only"
            )

        query_claim_ids = {
            row.claim_id
            for row in plan.queries
            if row.claim_id
        }

        planned_claims = {
            claim.claim_id: claim
            for group in plan.claims
            for claim in group.claims
        }

        unexpected = sorted(
            query_claim_ids
            - set(
                planned_claims
            )
        )

        if unexpected:
            raise ValueError(
                "diagnostic query references "
                "unplanned claim IDs: "
                f"{unexpected}"
            )

        claims = [
            planned_claims[
                claim_id
            ]
            for claim_id in sorted(
                query_claim_ids
            )
        ]

        work_index = {
            row.work_id: row
            for row in packet.works
        }

        reviews: list[
            DiagnosticClaimPriorArtReview
        ] = []

        with prior_art_review_audit_scope(
            assessment_kind=(
                "diagnostic_prior_art_review"
            ),
            source_portfolio_id=(
                plan.source_portfolio_id
            ),
            query_plan_id=(
                plan.plan_id
            ),
            prior_art_packet_id=(
                packet.packet_id
            ),
        ):
            for claim in claims:
                candidates = self.ranker.rank(
                    claim,
                    packet,
                    plan,
                )

                review_input = []

                for ranked in (
                    candidates.ranked_works
                ):
                    work = work_index[
                        ranked.work_id
                    ]

                    review_input.append(
                        {
                            "work_id":
                                work.work_id,
                            "title":
                                work.title,
                            "year":
                                work.year,
                            "doi":
                                work.doi,
                            "abstract":
                                work.abstract,
                            "semantic_similarity":
                                ranked.semantic_similarity,
                            "lexical_coverage":
                                ranked.lexical_coverage,
                            "reaction_domain_relevance":
                                ranked.reaction_domain_relevance,
                            "catalyst_scope_relevance":
                                ranked.catalyst_scope_relevance,
                            "relevance_score":
                                ranked.relevance_score,
                        }
                    )

                draft = (
                    self.review_backend
                    .review_diagnostic_claim(
                        claim,
                        review_input,
                    )
                )

                reviews.append(
                    compile_diagnostic_prior_art_review(
                        claim=claim,
                        candidates=candidates,
                        draft=draft,
                        packet=packet,
                    )
                )

        return reviews

    def _resolve_diagnostic_reviews_for_assess(
        self,
        *,
        diagnostic_plan: (
            LiteratureQueryPlan | None
        ),
        diagnostic_packet: (
            PriorArtPacket | None
        ),
        diagnostic_reviews: (
            Iterable[
                DiagnosticClaimPriorArtReview
            ]
            | None
        ),
    ) -> list[
        DiagnosticClaimPriorArtReview
    ]:
        # Production resolution is fail-closed:
        # no diagnostic artifacts means no lower-order gap authority.

        if diagnostic_reviews is not None:
            return list(
                diagnostic_reviews
            )

        if (
            diagnostic_plan is None
            and diagnostic_packet is None
        ):
            return []

        if (
            diagnostic_plan is None
            or diagnostic_packet is None
        ):
            raise ValueError(
                "diagnostic plan and packet must "
                "be supplied together"
            )

        return (
            self.review_diagnostic_prior_art(
                diagnostic_plan,
                diagnostic_packet,
            )
        )

    def assess(
        self,
        portfolio: HypothesisPortfolio,
        plan: LiteratureQueryPlan,
        packet: PriorArtPacket,
        *,
        diagnostic_plan: (
            LiteratureQueryPlan | None
        ) = None,
        diagnostic_packet: (
            PriorArtPacket | None
        ) = None,
        diagnostic_reviews: (
            Iterable[
                DiagnosticClaimPriorArtReview
            ]
            | None
        ) = None,
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

        resolved_diagnostic_reviews = (
            self._resolve_diagnostic_reviews_for_assess(
                diagnostic_plan=diagnostic_plan,
                diagnostic_packet=diagnostic_packet,
                diagnostic_reviews=(
                    diagnostic_reviews
                ),
            )
        )

        return self.compile_report_from_claim_reviews(
            portfolio,
            plan,
            packet,
            reviews,
            diagnostic_reviews=(
                resolved_diagnostic_reviews
            ),
            diagnostic_plan=(
                diagnostic_plan
            ),
            diagnostic_packet=(
                diagnostic_packet
            ),
            lineage=lineage,
        )
