from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Iterable

from pipeline_core.discovery.diagnostic_prior_art_report import (
    DiagnosticPriorArtReviewReport,
)
from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtReview,
    ExternalNoveltyCard,
    ExternalNoveltyReport,
    LiteratureQueryPlan,
    PriorArtPacket,
)
from pipeline_core.discovery.scientific_distinctiveness_contracts import (
    ScientificDistinctivenessClaimSignal,
    ScientificDistinctivenessReport,
    ScientificDistinctivenessReview,
    ScientificDistinctivenessSemanticDimensions,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


def _ordered_unique(
    values: Iterable[str],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        value = str(value)

        if not value or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def _fraction(
    count: int,
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0

    return count / denominator


def _match_work_ids(
    review: ClaimPriorArtReview,
    relationship: str,
) -> list[str]:
    return _ordered_unique(
        match.work_id
        for match in review.matches
        if match.relationship == relationship
    )


def _validate_sources(
    report: ExternalNoveltyReport,
    plan: LiteratureQueryPlan,
    packet: PriorArtPacket,
) -> None:
    portfolio_ids = {
        report.source_portfolio_id,
        plan.source_portfolio_id,
        packet.source_portfolio_id,
    }

    if len(portfolio_ids) != 1:
        raise ValueError(
            "Scientific-distinctiveness source portfolio "
            "provenance mismatch: "
            f"{sorted(portfolio_ids)}"
        )

    if (
        report.source_prior_art_packet_id
        != packet.packet_id
    ):
        raise ValueError(
            "External novelty report source_prior_art_packet_id "
            "does not match supplied prior-art packet"
        )

    if (
        packet.source_query_plan_id
        != plan.plan_id
    ):
        raise ValueError(
            "Prior-art packet source_query_plan_id does not "
            "match supplied literature query plan"
        )

    report_ids = [
        card.hypothesis_id
        for card in report.cards
    ]

    if len(report_ids) != len(set(report_ids)):
        raise ValueError(
            "Duplicate hypothesis_id in external novelty report"
        )

    plan_ids = [
        group.hypothesis_id
        for group in plan.claims
    ]

    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError(
            "Duplicate hypothesis claim group in query plan"
        )

    if set(report_ids) != set(plan_ids):
        raise ValueError(
            "External novelty cards and query-plan hypothesis "
            "groups do not match exactly"
        )

    work_ids = [
        work.work_id
        for work in packet.works
    ]

    if len(work_ids) != len(set(work_ids)):
        raise ValueError(
            "Duplicate work_id in prior-art packet"
        )


def _validated_diagnostic_lower_order_signal_map(
    *,
    report: ExternalNoveltyReport,
    main_plan: LiteratureQueryPlan,
    diagnostic_plan: LiteratureQueryPlan | None,
    diagnostic_packet: PriorArtPacket | None,
    diagnostic_review_report: DiagnosticPriorArtReviewReport | None,
) -> dict[str, list[str]] | None:
    # None means no diagnostic bundle: preserve historical main-only behavior.
    bundle = (
        diagnostic_plan,
        diagnostic_packet,
        diagnostic_review_report,
    )
    if not any(value is not None for value in bundle):
        return None
    if not all(value is not None for value in bundle):
        raise ValueError(
            "Scientific distinctiveness diagnostic prior-art provenance "
            "requires query plan, prior-art packet, and review report together"
        )

    assert diagnostic_plan is not None
    assert diagnostic_packet is not None
    assert diagnostic_review_report is not None

    source_portfolio_ids = {
        report.source_portfolio_id,
        main_plan.source_portfolio_id,
        diagnostic_plan.source_portfolio_id,
        diagnostic_packet.source_portfolio_id,
        diagnostic_review_report.source_portfolio_id,
    }
    if len(source_portfolio_ids) != 1:
        raise ValueError(
            "Diagnostic prior-art source portfolio provenance mismatch: "
            f"{sorted(source_portfolio_ids)}"
        )

    if diagnostic_packet.source_query_plan_id != diagnostic_plan.plan_id:
        raise ValueError(
            "Diagnostic prior-art packet source_query_plan_id mismatch"
        )

    if (
        diagnostic_review_report.source_diagnostic_query_plan_id
        != diagnostic_plan.plan_id
        or diagnostic_review_report.source_diagnostic_query_plan_sha256
        != diagnostic_plan.plan_sha256
    ):
        raise ValueError(
            "Diagnostic review report / query-plan provenance mismatch"
        )

    if (
        diagnostic_review_report.source_diagnostic_prior_art_packet_id
        != diagnostic_packet.packet_id
        or diagnostic_review_report.source_diagnostic_prior_art_packet_sha256
        != diagnostic_packet.packet_sha256
    ):
        raise ValueError(
            "Diagnostic review report / prior-art packet provenance mismatch"
        )

    if any(
        row.query_kind != "claim_diagnostic"
        for row in diagnostic_plan.queries
    ):
        raise ValueError(
            "Diagnostic query plan contains non-diagnostic query"
        )

    main_claims = {
        claim.claim_id: claim
        for group in main_plan.claims
        for claim in group.claims
    }
    diagnostic_claims = {
        claim.claim_id: claim
        for group in diagnostic_plan.claims
        for claim in group.claims
    }
    if set(diagnostic_claims) != set(main_claims):
        raise ValueError(
            "Diagnostic query-plan claim set does not match main query plan"
        )

    for claim_id, claim in main_claims.items():
        diagnostic_claim = diagnostic_claims[claim_id]
        if (
            diagnostic_claim.hypothesis_id != claim.hypothesis_id
            or diagnostic_claim.text != claim.text
            or diagnostic_claim.importance != claim.importance
        ):
            raise ValueError(
                "Diagnostic query-plan claim provenance drift: "
                f"{claim_id}"
            )

    rows = list(diagnostic_review_report.reviews)
    review_claim_ids = [row.claim_id for row in rows]
    if len(review_claim_ids) != len(set(review_claim_ids)):
        raise ValueError("Duplicate diagnostic claim review IDs")

    packet_works = {
        row.work_id: row
        for row in diagnostic_packet.works
    }
    if len(packet_works) != len(diagnostic_packet.works):
        raise ValueError(
            "Duplicate work_id in diagnostic prior-art packet"
        )

    projected: dict[str, list[str]] = {}

    for review in rows:
        planned = main_claims.get(review.claim_id)
        if planned is None:
            raise ValueError(
                "Diagnostic review references unplanned claim_id: "
                f"{review.claim_id}"
            )
        if review.hypothesis_id != planned.hypothesis_id:
            raise ValueError(
                "Diagnostic review hypothesis_id drift for "
                f"{review.claim_id}"
            )
        if review.claim_text != planned.text:
            raise ValueError(
                "Diagnostic review text drift for "
                f"{review.claim_id}"
            )
        if (
            review.diagnostic_query_kind
            != planned.diagnostic_query_kind
        ):
            raise ValueError(
                "Diagnostic review kind drift for "
                f"{review.claim_id}"
            )

        expected_query = str(
            planned.diagnostic_execution_query or ""
        ).strip()
        if review.diagnostic_execution_query != expected_query:
            raise ValueError(
                "Diagnostic review execution-query drift for "
                f"{review.claim_id}"
            )

        signal_ids = list(review.signal_work_ids)
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError(
                "Duplicate diagnostic signal work IDs for "
                f"{review.claim_id}"
            )

        match_by_id = {
            match.work_id: match
            for match in review.matches
        }
        for work_id in signal_ids:
            match = match_by_id.get(work_id)
            if match is None:
                raise ValueError(
                    "Diagnostic signal lacks compiled match for "
                    f"{review.claim_id}: {work_id}"
                )
            if (
                review.diagnostic_query_kind
                == "LOWER_ORDER_RELATION"
                and match.relationship
                != "LOWER_ORDER_RELATION_PRIOR_ART"
            ):
                raise ValueError(
                    "Lower-order diagnostic signal has wrong relationship for "
                    f"{review.claim_id}: {work_id}"
                )
            work = packet_works.get(work_id)
            if (
                work is None
                or not work.abstract
                or not match.abstract_available
            ):
                raise ValueError(
                    "Diagnostic signal lacks abstract-backed packet evidence "
                    f"for {review.claim_id}: {work_id}"
                )

        if review.diagnostic_query_kind == "LOWER_ORDER_RELATION":
            projected[review.claim_id] = signal_ids

    return projected


def _validate_card(
    card: ExternalNoveltyCard,
    planned_group,
    *,
    packet_work_ids: set[str],
    diagnostic_lower_order_signals_by_claim: (
        dict[str, list[str]] | None
    ) = None,
) -> dict[str, object]:

    planned = {
        claim.claim_id:
            claim
        for claim in planned_group.claims
    }

    if len(planned) != len(planned_group.claims):
        raise ValueError(
            "Duplicate claim_id in query-plan hypothesis group: "
            f"{card.hypothesis_id}"
        )

    reviews = {
        review.claim_id:
            review
        for review in card.claim_reviews
    }

    if len(reviews) != len(card.claim_reviews):
        raise ValueError(
            "Duplicate claim review ID in external novelty card: "
            f"{card.hypothesis_id}"
        )

    if set(planned) != set(reviews):
        raise ValueError(
            "Claim-review set does not exactly match query-plan "
            f"claims for hypothesis {card.hypothesis_id}"
        )

    referenced_work_ids: list[str] = []
    referenced_diagnostic_work_ids: list[str] = []

    computed_lower_order_work_ids: list[str] = []
    computed_core_lower_order_work_ids: list[str] = []
    computed_core_lower_order_claim_ids: list[str] = []
    computed_directional_work_ids: list[str] = []

    for claim_id, claim in planned.items():
        review = reviews[claim_id]

        if review.hypothesis_id != card.hypothesis_id:
            raise ValueError(
                "Claim review hypothesis_id drift: "
                f"{claim_id}"
            )

        if claim.hypothesis_id != card.hypothesis_id:
            raise ValueError(
                "Planned claim hypothesis_id drift: "
                f"{claim_id}"
            )

        if review.claim_text != claim.text:
            raise ValueError(
                "Claim review text drift: "
                f"{claim_id}"
            )

        if review.importance != claim.importance:
            raise ValueError(
                "Claim review importance drift: "
                f"{claim_id}"
            )

        if review.coverage.claim_id != claim_id:
            raise ValueError(
                "Claim review coverage ID mismatch: "
                f"{claim_id}"
            )

        for match in review.matches:
            if match.work_id not in packet_work_ids:
                raise ValueError(
                    "Claim review references unknown prior-art "
                    f"work_id {match.work_id!r}"
                )

            referenced_work_ids.append(
                match.work_id
            )

            if (
                diagnostic_lower_order_signals_by_claim is None
                and match.relationship
                == "LOWER_ORDER_RELATION_PRIOR_ART"
            ):
                computed_lower_order_work_ids.append(
                    match.work_id
                )
                if review.importance == "core":
                    computed_core_lower_order_work_ids.append(
                        match.work_id
                    )

            if (
                match.relationship
                == "DIRECTIONAL_COUNTEREVIDENCE"
            ):
                computed_directional_work_ids.append(
                    match.work_id
                )

        if diagnostic_lower_order_signals_by_claim is None:
            if (
                review.importance == "core"
                and any(
                    match.relationship
                    == "LOWER_ORDER_RELATION_PRIOR_ART"
                    for match in review.matches
                )
            ):
                computed_core_lower_order_claim_ids.append(
                    claim_id
                )
        else:
            diagnostic_ids = list(
                diagnostic_lower_order_signals_by_claim.get(
                    claim_id,
                    [],
                )
            )
            computed_lower_order_work_ids.extend(
                diagnostic_ids
            )
            referenced_diagnostic_work_ids.extend(
                diagnostic_ids
            )
            if review.importance == "core" and diagnostic_ids:
                computed_core_lower_order_work_ids.extend(
                    diagnostic_ids
                )
                computed_core_lower_order_claim_ids.append(
                    claim_id
                )

    # Card-level references must still resolve to the supplied packet.
    # However, several card fields below are redundant aggregates derived
    # from claim-level reviews. Historical reports may contain stale
    # aggregate snapshots after the production aggregation policy evolved.
    #
    # Claim-level reviews remain authoritative for this diagnostic. Drift in
    # redundant aggregates is recorded explicitly instead of silently
    # accepted or treated as a primary-provenance failure.
    main_packet_card_work_fields = (
        card.strongest_prior_art_work_ids,
        card.contextual_conflict_work_ids,
        card.directional_counterevidence_work_ids,
    )

    for values in main_packet_card_work_fields:
        for work_id in values:
            if work_id not in packet_work_ids:
                raise ValueError(
                    "External novelty card references unknown "
                    f"prior-art work_id {work_id!r}"
                )
            referenced_work_ids.append(work_id)

    validated_diagnostic_lower_order_ids = {
        work_id
        for values in (
            diagnostic_lower_order_signals_by_claim or {}
        ).values()
        for work_id in values
    }

    for values in (
        card.lower_order_prior_art_work_ids,
        card.lower_order_core_prior_art_work_ids,
    ):
        for work_id in values:
            if work_id in packet_work_ids:
                referenced_work_ids.append(work_id)
            elif work_id in validated_diagnostic_lower_order_ids:
                referenced_diagnostic_work_ids.append(work_id)
            else:
                raise ValueError(
                    "External novelty card references unknown "
                    f"prior-art work_id {work_id!r}"
                )

    aggregate_warnings: list[str] = []

    computed_lower_order_work_ids = _ordered_unique(
        computed_lower_order_work_ids
    )
    computed_core_lower_order_work_ids = _ordered_unique(
        computed_core_lower_order_work_ids
    )
    computed_core_lower_order_claim_ids = _ordered_unique(
        computed_core_lower_order_claim_ids
    )
    computed_directional_work_ids = _ordered_unique(
        computed_directional_work_ids
    )

    computed_lower = set(
        computed_lower_order_work_ids
    )

    if not set(
        card.lower_order_prior_art_work_ids
    ).issubset(computed_lower):
        aggregate_warnings.append(
            "card_lower_order_prior_art_work_ids_not_supported"
        )

    computed_core_lower = set(
        computed_core_lower_order_work_ids
    )

    if set(
        card.lower_order_core_prior_art_work_ids
    ) != computed_core_lower:
        aggregate_warnings.append(
            "card_lower_order_core_prior_art_work_ids_drift"
        )

    if (
        card.lower_order_core_unique_work_count
        != len(computed_core_lower)
    ):
        aggregate_warnings.append(
            "card_lower_order_core_unique_work_count_drift"
        )

    if set(
        card.lower_order_supported_core_claim_ids
    ) != set(
        computed_core_lower_order_claim_ids
    ):
        aggregate_warnings.append(
            "card_lower_order_supported_core_claim_ids_drift"
        )

    computed_gap_claim_ids = [
        claim_id
        for claim_id in computed_core_lower_order_claim_ids
        if reviews[claim_id].status == "COMPONENTS_ONLY"
    ]

    if (
        computed_gap_claim_ids
        and card.coverage.sufficient_for_absence_based_novelty
    ):
        computed_relational_gap_kind = (
            "HIGHER_ORDER_RELATIONAL_GAP"
        )
    else:
        computed_relational_gap_kind = "NONE"

    if set(
        card.higher_order_relational_gap_claim_ids
    ) != set(
        computed_gap_claim_ids
        if computed_relational_gap_kind
        == "HIGHER_ORDER_RELATIONAL_GAP"
        else []
    ):
        aggregate_warnings.append(
            "card_higher_order_relational_gap_claim_ids_drift"
        )

    if (
        card.relational_gap_kind
        != computed_relational_gap_kind
    ):
        aggregate_warnings.append(
            "card_relational_gap_kind_drift"
        )

    if not set(
        card.directional_counterevidence_work_ids
    ).issubset(
        set(
            computed_directional_work_ids
        )
    ):
        aggregate_warnings.append(
            "card_directional_counterevidence_work_ids_drift"
        )

    return {
        "planned":
            planned,

        "reviews":
            reviews,

        "referenced_work_ids":
            _ordered_unique(
                referenced_work_ids
            ),

        "referenced_diagnostic_work_ids":
            _ordered_unique(
                referenced_diagnostic_work_ids
            ),

        "full_directional_work_ids":
            computed_directional_work_ids,

        "computed_core_lower_order_work_ids":
            computed_core_lower_order_work_ids,

        "computed_core_lower_order_claim_ids":
            computed_core_lower_order_claim_ids,

        "computed_higher_order_gap_claim_ids":
            (
                computed_gap_claim_ids
                if computed_relational_gap_kind
                == "HIGHER_ORDER_RELATIONAL_GAP"
                else []
            ),

        "computed_relational_gap_kind":
            computed_relational_gap_kind,

        "source_aggregate_warnings":
            sorted(
                set(
                    aggregate_warnings
                )
            ),
    }


def _evidence_pattern(
    card: ExternalNoveltyCard,
    core_signals: list[
        ScientificDistinctivenessClaimSignal
    ],
    *,
    relational_gap_kind: str,
) -> str:

    if not core_signals:
        return (
            "NO_REVIEWABLE_CORE_CLAIMS"
        )

    core_count = len(
        core_signals
    )

    direct = sum(
        signal.prior_art_status
        == "DIRECT_PRIOR_ART"
        for signal in core_signals
    )

    relation_backed = sum(
        signal.prior_art_status
        in {
            "DIRECT_PRIOR_ART",
            "PARTIAL_PRIOR_ART",
        }
        for signal in core_signals
    )

    lower_order = sum(
        bool(
            signal.lower_order_prior_art_work_ids
        )
        for signal in core_signals
    )

    no_direct = sum(
        signal.prior_art_status
        == "NO_DIRECT_MATCH_FOUND"
        for signal in core_signals
    )

    if direct == core_count:
        return (
            "DIRECT_PRIOR_ART_SATURATED"
        )

    if relation_backed == core_count:
        return (
            "RELATION_BACKED_SATURATED"
        )

    if (
        relational_gap_kind
        == "HIGHER_ORDER_RELATIONAL_GAP"
    ):
        return (
            "HIGHER_ORDER_RELATIONAL_GAP_WITH_LOWER_ORDER_PRIOR_ART"
        )

    if not (
        card.coverage
        .sufficient_for_absence_based_novelty
    ):
        return (
            "SEARCH_COVERAGE_LIMITED"
        )

    if lower_order > 0:
        return (
            "LOWER_ORDER_PRIOR_ART_PRESENT"
        )

    if no_direct == core_count:
        return (
            "SEARCH_BOUNDED_UNMATCHED"
        )

    return "MIXED_PRIOR_ART"


def _interpretation(
    pattern: str,
) -> str:

    if (
        pattern
        == "DIRECT_PRIOR_ART_SATURATED"
    ):
        return (
            "All reviewable core claims are classified as "
            "DIRECT_PRIOR_ART in the frozen external-novelty "
            "review. This is a prior-art structure diagnostic, "
            "not a scientific-importance judgment."
        )

    if (
        pattern
        == "RELATION_BACKED_SATURATED"
    ):
        return (
            "All reviewable core claims are classified as "
            "DIRECT_PRIOR_ART or PARTIAL_PRIOR_ART. The frozen "
            "evidence therefore shows strong relation-level "
            "adjacency, without deciding scientific obviousness."
        )

    if (
        pattern
        == (
            "HIGHER_ORDER_RELATIONAL_GAP_WITH_"
            "LOWER_ORDER_PRIOR_ART"
        )
    ):
        return (
            "The frozen external-novelty evidence records "
            "lower-order prior art for at least one core claim "
            "while retaining a search-bounded higher-order "
            "relational gap. This does not determine whether the "
            "higher-order combination is scientifically obvious "
            "or non-obvious."
        )

    if (
        pattern
        == "LOWER_ORDER_PRIOR_ART_PRESENT"
    ):
        return (
            "Reviewed lower-order relation prior art is present "
            "for at least one core claim, but the external report "
            "does not establish the higher-order relational-gap "
            "condition. Semantic non-obviousness remains unassessed."
        )

    if (
        pattern
        == "SEARCH_BOUNDED_UNMATCHED"
    ):
        return (
            "No reviewable core claim has a direct prior-art "
            "match under the recorded sufficient search coverage. "
            "This remains search-bounded absence evidence, not "
            "proof of scientific non-obviousness."
        )

    if (
        pattern
        == "SEARCH_COVERAGE_LIMITED"
    ):
        return (
            "The frozen search coverage is insufficient for "
            "absence-based distinctiveness interpretation. "
            "Positive prior-art matches remain descriptive, but "
            "missing matches must not be promoted to novelty."
        )

    if (
        pattern
        == "NO_REVIEWABLE_CORE_CLAIMS"
    ):
        return (
            "No reviewable core prior-art claims are available "
            "for deterministic distinctiveness diagnostics."
        )

    return (
        "The frozen prior-art review contains a mixed evidence "
        "pattern. v1 records that structure without converting it "
        "into a scientific non-obviousness verdict."
    )


class ScientificDistinctivenessAnalyzer:
    """Build a provenance-locked diagnostic from frozen novelty artifacts.

    This analyzer performs:
      - no literature retrieval,
      - no embedding/ranking,
      - no LLM review,
      - no hypothesis action or selection.

    It converts already-reviewed external prior-art structure into
    explicit deterministic signals and leaves semantic non-obviousness
    dimensions unassessed.
    """

    def build(
        self,
        report: ExternalNoveltyReport,
        plan: LiteratureQueryPlan,
        packet: PriorArtPacket,
        *,
        diagnostic_plan: LiteratureQueryPlan | None = None,
        diagnostic_packet: PriorArtPacket | None = None,
        diagnostic_review_report: (
            DiagnosticPriorArtReviewReport | None
        ) = None,
    ) -> ScientificDistinctivenessReport:

        _validate_sources(
            report,
            plan,
            packet,
        )

        diagnostic_lower_order_signals_by_claim = (
            _validated_diagnostic_lower_order_signal_map(
                report=report,
                main_plan=plan,
                diagnostic_plan=diagnostic_plan,
                diagnostic_packet=diagnostic_packet,
                diagnostic_review_report=diagnostic_review_report,
            )
        )

        plan_by_hypothesis = {
            group.hypothesis_id:
                group
            for group in plan.claims
        }

        packet_work_ids = {
            work.work_id
            for work in packet.works
        }

        reviews_out: list[
            ScientificDistinctivenessReview
        ] = []

        for card in report.cards:

            group = plan_by_hypothesis[
                card.hypothesis_id
            ]

            validation = _validate_card(
                card,
                group,
                packet_work_ids=packet_work_ids,
                diagnostic_lower_order_signals_by_claim=(
                    diagnostic_lower_order_signals_by_claim
                ),
            )

            planned = validation[
                "planned"
            ]

            review_by_id = validation[
                "reviews"
            ]

            claim_signals: list[
                ScientificDistinctivenessClaimSignal
            ] = []

            for claim in group.claims:

                review = review_by_id[
                    claim.claim_id
                ]

                relationship_counts = Counter(
                    match.relationship
                    for match in review.matches
                )

                claim_signals.append(
                    ScientificDistinctivenessClaimSignal(
                        hypothesis_id=(
                            card.hypothesis_id
                        ),
                        claim_id=(
                            claim.claim_id
                        ),
                        claim_kind=(
                            claim.kind
                        ),
                        importance=(
                            review.importance
                        ),
                        claim_text=(
                            review.claim_text
                        ),
                        prior_art_status=(
                            review.status
                        ),
                        query_count=(
                            review.coverage
                            .query_count
                        ),
                        successful_query_count=(
                            review.coverage
                            .successful_query_count
                        ),
                        unique_work_count=(
                            review.coverage
                            .unique_work_count
                        ),
                        abstract_work_count=(
                            review.coverage
                            .abstract_work_count
                        ),
                        reviewed_work_count=(
                            review.coverage
                            .reviewed_work_count
                        ),
                        relationship_counts=dict(
                            sorted(
                                relationship_counts
                                .items()
                            )
                        ),
                        direct_prior_art_work_ids=(
                            _match_work_ids(
                                review,
                                "DIRECT_PRIOR_ART",
                            )
                        ),
                        partial_prior_art_work_ids=(
                            _match_work_ids(
                                review,
                                "PARTIAL_PRIOR_ART",
                            )
                        ),
                        lower_order_prior_art_work_ids=(
                            list(
                                diagnostic_lower_order_signals_by_claim.get(
                                    claim.claim_id,
                                    [],
                                )
                            )
                            if diagnostic_lower_order_signals_by_claim
                            is not None
                            else _match_work_ids(
                                review,
                                (
                                    "LOWER_ORDER_"
                                    "RELATION_PRIOR_ART"
                                ),
                            )
                        ),
                        directional_counterevidence_work_ids=(
                            _match_work_ids(
                                review,
                                "DIRECTIONAL_COUNTEREVIDENCE",
                            )
                        ),
                        contextual_conflict_work_ids=(
                            _match_work_ids(
                                review,
                                "CONTEXTUAL_CONFLICT",
                            )
                        ),
                        conflicting_prior_art_work_ids=(
                            _match_work_ids(
                                review,
                                "CONFLICTING_PRIOR_ART",
                            )
                        ),
                        reason_codes=(
                            list(
                                review.reason_codes
                            )
                        ),
                    )
                )

            explicit_core = [
                signal
                for signal in claim_signals
                if signal.importance
                == "core"
            ]

            # Mirror the external novelty assessor's fallback semantics.
            core_signals = (
                explicit_core
                or claim_signals
            )

            core_count = len(
                core_signals
            )

            direct_count = sum(
                signal.prior_art_status
                == "DIRECT_PRIOR_ART"
                for signal in core_signals
            )

            relation_backed_count = sum(
                signal.prior_art_status
                in {
                    "DIRECT_PRIOR_ART",
                    "PARTIAL_PRIOR_ART",
                }
                for signal in core_signals
            )

            component_count = sum(
                signal.prior_art_status
                == "COMPONENTS_ONLY"
                for signal in core_signals
            )

            no_direct_count = sum(
                signal.prior_art_status
                == "NO_DIRECT_MATCH_FOUND"
                for signal in core_signals
            )

            lower_order_count = sum(
                bool(
                    signal
                    .lower_order_prior_art_work_ids
                )
                for signal in core_signals
            )

            full_directional_work_ids = (
                validation[
                    "full_directional_work_ids"
                ]
            )

            computed_core_lower_order_work_ids = (
                validation[
                    "computed_core_lower_order_work_ids"
                ]
            )

            computed_higher_order_gap_claim_ids = (
                validation[
                    "computed_higher_order_gap_claim_ids"
                ]
            )

            computed_relational_gap_kind = (
                validation[
                    "computed_relational_gap_kind"
                ]
            )

            source_aggregate_warnings = (
                validation[
                    "source_aggregate_warnings"
                ]
            )

            pattern = _evidence_pattern(
                card,
                core_signals,
                relational_gap_kind=(
                    computed_relational_gap_kind
                ),
            )

            reason_codes = [
                (
                    "existing_external_novelty_"
                    "evidence_reused"
                ),
                (
                    "semantic_non_obviousness_"
                    "dimensions_unassessed"
                ),
                (
                    "no_new_literature_retrieval"
                ),
                (
                    "no_new_model_review"
                ),
                (
                    f"evidence_pattern:{pattern}"
                ),
            ]

            if (
                computed_relational_gap_kind
                == (
                    "HIGHER_ORDER_"
                    "RELATIONAL_GAP"
                )
            ):
                reason_codes.append(
                    "higher_order_relational_gap_present"
                )

            if lower_order_count:
                reason_codes.append(
                    "lower_order_relation_prior_art_present"
                )

            if full_directional_work_ids:
                reason_codes.append(
                    "directional_counterevidence_present"
                )

            reviews_out.append(
                ScientificDistinctivenessReview(
                    hypothesis_id=(
                        card.hypothesis_id
                    ),
                    title=(
                        card.title
                    ),
                    external_novelty_status=(
                        card.status
                    ),
                    evidence_pattern=(
                        pattern
                    ),
                    claim_count=(
                        len(
                            claim_signals
                        )
                    ),
                    core_claim_count=(
                        core_count
                    ),
                    direct_prior_art_core_claim_count=(
                        direct_count
                    ),
                    relation_backed_core_claim_count=(
                        relation_backed_count
                    ),
                    component_supported_core_claim_count=(
                        component_count
                    ),
                    no_direct_match_core_claim_count=(
                        no_direct_count
                    ),
                    lower_order_supported_core_claim_count=(
                        lower_order_count
                    ),
                    direct_prior_art_core_fraction=(
                        _fraction(
                            direct_count,
                            core_count,
                        )
                    ),
                    relation_backed_core_fraction=(
                        _fraction(
                            relation_backed_count,
                            core_count,
                        )
                    ),
                    component_supported_core_fraction=(
                        _fraction(
                            component_count,
                            core_count,
                        )
                    ),
                    no_direct_match_core_fraction=(
                        _fraction(
                            no_direct_count,
                            core_count,
                        )
                    ),
                    lower_order_supported_core_fraction=(
                        _fraction(
                            lower_order_count,
                            core_count,
                        )
                    ),
                    higher_order_relational_gap_claim_count=(
                        len(
                            computed_higher_order_gap_claim_ids
                        )
                    ),
                    lower_order_core_unique_work_count=(
                        len(
                            computed_core_lower_order_work_ids
                        )
                    ),
                    directional_counterevidence_unique_work_count=(
                        len(
                            full_directional_work_ids
                        )
                    ),
                    search_coverage_sufficient=(
                        card.coverage
                        .sufficient_for_absence_based_novelty
                    ),
                    search_unique_work_count=(
                        card.coverage
                        .unique_work_count
                    ),
                    search_abstract_work_count=(
                        card.coverage
                        .abstract_work_count
                    ),
                    claim_signals=(
                        claim_signals
                    ),
                    semantic_dimensions=(
                        ScientificDistinctivenessSemanticDimensions()
                    ),
                    source_claim_ids=[
                        claim.claim_id
                        for claim
                        in group.claims
                    ],
                    referenced_prior_art_work_ids=(
                        validation[
                            "referenced_work_ids"
                        ]
                    ),
                    referenced_diagnostic_prior_art_work_ids=(
                        validation[
                            "referenced_diagnostic_work_ids"
                        ]
                    ),
                    source_aggregate_warnings=(
                        source_aggregate_warnings
                    ),
                    reason_codes=sorted(
                        set(
                            reason_codes
                        )
                    ),
                    interpretation=(
                        _interpretation(
                            pattern
                        )
                    ),
                )
            )

        counts = Counter(
            row.evidence_pattern
            for row in reviews_out
        )

        report_id = _stable_id(
            "scientific_distinctiveness_report",
            report.report_id,
            plan.plan_id,
            packet.packet_id,
            *(
                [diagnostic_review_report.report_id]
                if diagnostic_review_report is not None
                else []
            ),
            *[
                (
                    f"{row.hypothesis_id}:"
                    f"{row.evidence_pattern}"
                )
                for row in reviews_out
            ],
        )

        body = {
            "schema_version":
                "scientific-distinctiveness-report-v1",

            "report_id":
                report_id,

            "source_portfolio_id":
                report.source_portfolio_id,

            "source_external_novelty_report_id":
                report.report_id,

            "source_external_novelty_report_sha256":
                report.report_sha256,

            "source_query_plan_id":
                plan.plan_id,

            "source_query_plan_sha256":
                plan.plan_sha256,

            "source_prior_art_packet_id":
                packet.packet_id,

            "source_prior_art_packet_sha256":
                packet.packet_sha256,

            **(
                {
                    "source_diagnostic_query_plan_id":
                        diagnostic_plan.plan_id,
                    "source_diagnostic_query_plan_sha256":
                        diagnostic_plan.plan_sha256,
                    "source_diagnostic_prior_art_packet_id":
                        diagnostic_packet.packet_id,
                    "source_diagnostic_prior_art_packet_sha256":
                        diagnostic_packet.packet_sha256,
                    "source_diagnostic_review_report_id":
                        diagnostic_review_report.report_id,
                    "source_diagnostic_review_report_sha256":
                        diagnostic_review_report.report_sha256,
                }
                if (
                    diagnostic_plan is not None
                    and diagnostic_packet is not None
                    and diagnostic_review_report is not None
                )
                else {}
            ),

            "source_searched_at_utc":
                report.searched_at_utc,

            "reviews": [
                row.model_dump(
                    mode="json"
                )
                for row in reviews_out
            ],

            "evidence_pattern_counts":
                dict(
                    sorted(
                        counts.items()
                    )
                ),

            "source_aggregate_warning_count":
                sum(
                    len(
                        row.source_aggregate_warnings
                    )
                    for row in reviews_out
                ),

            "diagnostic_scope":
                "existing_external_prior_art_evidence_only",

            "semantic_dimensions_assessed":
                False,

            "retrieval_performed":
                False,

            "model_review_performed":
                False,

            "action_policy_applied":
                False,

            "scientific_selection_changed":
                False,

            "epistemic_usage": (
                "diagnostic_only_existing_prior_art_"
                "not_positive_premise"
            ),
        }

        return ScientificDistinctivenessReport(
            **body,
            report_sha256=_sha256_json(
                body
            ),
        )
