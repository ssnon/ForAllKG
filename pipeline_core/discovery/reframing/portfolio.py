from __future__ import annotations

from pipeline_core.discovery.reframing.critic_contracts import (
    CRITIC_DIMENSIONS,
    ScientificReframeCriticReport,
    ScientificReframeCriticReview,
)
from pipeline_core.discovery.reframing.portfolio_contracts import (
    PortfolioVectorEntry,
    ReframePortfolioAssessment,
    ReframePortfolioRouteSignals,
    ReframePortfolioSlot,
    ScientificReframeShadowPortfolio,
    stable_shadow_portfolio_id,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)


def _rating_map(review: ScientificReframeCriticReview | None) -> dict[str, int | None]:
    if review is None:
        return {dimension: None for dimension in CRITIC_DIMENSIONS}
    return {row.dimension: row.rating for row in review.dimensions}


def _vector_complete(review: ScientificReframeCriticReview | None) -> bool:
    if review is None:
        return False
    return all(row.rating is not None for row in review.dimensions)


def _pareto_evaluable(review: ScientificReframeCriticReview | None) -> bool:
    if review is None or not _vector_complete(review):
        return False
    audit = review.structural_audit
    return bool(
        audit.operator_shape_valid
        and audit.all_premise_ids_grounded
        and audit.all_gap_ids_grounded
    )


def _dominates(
    left: ScientificReframeCriticReview,
    right: ScientificReframeCriticReview,
) -> bool:
    """Pareto dominance over the full 10-D critic vector.

    All dimensions are oriented so larger is better. Missing ratings are never
    imputed; callers must restrict comparisons to evaluable reviews.
    """

    left_ratings = _rating_map(left)
    right_ratings = _rating_map(right)
    pairs = [
        (left_ratings[dimension], right_ratings[dimension])
        for dimension in CRITIC_DIMENSIONS
    ]
    if any(a is None or b is None for a, b in pairs):
        return False
    return all(int(a) >= int(b) for a, b in pairs) and any(
        int(a) > int(b) for a, b in pairs
    )


def _risk_flags(review: ScientificReframeCriticReview | None) -> list[str]:
    if review is None:
        return ["incomplete_critic_vector"]

    flags: list[str] = []
    audit = review.structural_audit
    if not _vector_complete(review):
        flags.append("incomplete_critic_vector")
    if not audit.operator_shape_valid:
        flags.append("invalid_operator_shape")
    if not audit.all_premise_ids_grounded:
        flags.append("ungrounded_premise_reference")
    if not audit.all_gap_ids_grounded:
        flags.append("ungrounded_gap_reference")

    ratings = _rating_map(review)
    threshold_flags = (
        ("operator_validity", "low_operator_validity"),
        ("premise_fidelity", "low_premise_fidelity"),
        ("over_specificity", "low_over_specificity_safety"),
        ("reframe_depth", "low_reframe_depth"),
    )
    for dimension, flag in threshold_flags:
        value = ratings.get(dimension)
        if value is not None and value <= 1:
            flags.append(flag)
    return flags


def _route_candidate(
    candidate: ScientificReframeCandidate,
    review: ScientificReframeCriticReview | None,
) -> tuple[str, str | None, ReframePortfolioRouteSignals, list[str], list[str]]:
    flags = _risk_flags(review)
    reasons: list[str] = []

    if review is None or not _pareto_evaluable(review):
        reasons.append(
            "Candidate lacks a complete, structurally grounded critic vector; "
            "it is preserved outside A/B/C/D routing until review is complete."
        )
        return (
            "not_evaluable",
            None,
            ReframePortfolioRouteSignals(),
            flags,
            reasons,
        )

    ratings = _rating_map(review)
    deep_reframe = bool(
        ratings["operator_validity"] is not None
        and ratings["operator_validity"] >= 2
        and ratings["reframe_depth"] is not None
        and ratings["reframe_depth"] >= 2
    )
    high_risk = bool(
        ratings["premise_fidelity"] is not None
        and ratings["premise_fidelity"] <= 1
        or ratings["over_specificity"] is not None
        and ratings["over_specificity"] <= 1
        or not review.structural_audit.operator_shape_valid
        or not review.structural_audit.all_premise_ids_grounded
        or not review.structural_audit.all_gap_ids_grounded
    )

    if high_risk:
        reasons.append(
            "Deep-reframe signal is retained, but low premise fidelity or "
            "over-specificity safety routes the candidate to the high-risk "
            "exploratory slot without production authority."
        )
        return (
            "high_risk_exploratory",
            "D_HIGH_RISK_EXPLORATORY",
            ReframePortfolioRouteSignals(
                structural_followup_signal=deep_reframe,
                measurement_regime_signal=False,
                high_risk_signal=True,
            ),
            flags,
            reasons,
        )

    if candidate.operator_id == "REGIME_BOUNDARY":
        reasons.append(
            "A structurally grounded REGIME_BOUNDARY candidate with no high-risk "
            "routing flag is preserved in the measurement/regime challenge slot."
        )
        return (
            "measurement_regime_challenge",
            "C_MEASUREMENT_REGIME_CHALLENGE",
            ReframePortfolioRouteSignals(
                structural_followup_signal=deep_reframe,
                measurement_regime_signal=True,
                high_risk_signal=False,
            ),
            flags,
            reasons,
        )

    reasons.append(
        "A structurally grounded non-regime reframe with no high-risk routing "
        "flag is preserved in the structural reframe slot."
    )
    return (
        "structural_reframe",
        "B_STRUCTURAL_REFRAME",
        ReframePortfolioRouteSignals(
            structural_followup_signal=deep_reframe,
            measurement_regime_signal=False,
            high_risk_signal=False,
        ),
        flags,
        reasons,
    )


def build_scientific_reframe_shadow_portfolio(
    *,
    shadow: ScientificReframingShadowReport,
    critic: ScientificReframeCriticReport,
) -> ScientificReframeShadowPortfolio:
    if critic.source_shadow_report_id != shadow.report_id:
        raise ValueError("critic/shadow report lineage mismatch")
    for field in ("source_task_id", "source_context_id", "source_context_sha256"):
        if getattr(critic, field) != getattr(shadow, field):
            raise ValueError(f"critic/shadow {field} lineage mismatch")

    candidates = {row.reframe_id: row for row in shadow.candidates}
    reviews = {row.candidate_id: row for row in critic.reviews}
    unknown_reviews = sorted(set(reviews) - set(candidates))
    if unknown_reviews:
        raise ValueError(
            "critic report references candidates absent from shadow report: "
            + ", ".join(unknown_reviews)
        )

    evaluable_reviews = {
        candidate_id: review
        for candidate_id, review in reviews.items()
        if _pareto_evaluable(review)
    }
    dominated_by: dict[str, list[str]] = {candidate_id: [] for candidate_id in candidates}
    for right_id, right_review in evaluable_reviews.items():
        for left_id, left_review in evaluable_reviews.items():
            if left_id == right_id:
                continue
            if _dominates(left_review, right_review):
                dominated_by[right_id].append(left_id)
        dominated_by[right_id].sort()

    assessments: list[ReframePortfolioAssessment] = []
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        review = reviews.get(candidate_id)
        ratings = _rating_map(review)
        vector = [
            PortfolioVectorEntry(
                dimension=dimension,
                rating=ratings[dimension],
            )
            for dimension in CRITIC_DIMENSIONS
        ]
        complete = _vector_complete(review)
        if not _pareto_evaluable(review):
            pareto_status = "not_evaluable"
        elif dominated_by[candidate_id]:
            pareto_status = "dominated"
        else:
            pareto_status = "non_dominated"

        route_kind, slot_id, signals, flags, reasons = _route_candidate(
            candidate,
            review,
        )
        assessments.append(
            ReframePortfolioAssessment(
                candidate_id=candidate_id,
                operator_id=candidate.operator_id,
                review_id=review.review_id if review is not None else None,
                vector=vector,
                vector_complete=complete,
                pareto_status=pareto_status,
                dominated_by_candidate_ids=dominated_by[candidate_id],
                route_kind=route_kind,
                assigned_slot_id=slot_id,
                risk_flags=flags,
                route_signals=signals,
                routing_reasons=reasons,
            )
        )

    by_slot: dict[str, list[str]] = {
        "A_RELATIONAL_EVIDENCE_NEAR": [],
        "B_STRUCTURAL_REFRAME": [],
        "C_MEASUREMENT_REGIME_CHALLENGE": [],
        "D_HIGH_RISK_EXPLORATORY": [],
    }
    for row in assessments:
        if row.assigned_slot_id is not None:
            by_slot[row.assigned_slot_id].append(row.candidate_id)
    for values in by_slot.values():
        values.sort()

    slots = [
        ReframePortfolioSlot(
            slot_id="A_RELATIONAL_EVIDENCE_NEAR",
            label="Evidence-near relational",
            purpose=(
                "Reserved for the existing relational discovery lane; the scientific "
                "reframing lane does not manufacture a relational candidate to fill quota."
            ),
            candidate_ids=[],
            empty_reason="Relational lane integration is intentionally deferred.",
        ),
        ReframePortfolioSlot(
            slot_id="B_STRUCTURAL_REFRAME",
            label="Structural reframe",
            purpose="Grounded model-structure changes without a high-risk routing flag.",
            candidate_ids=by_slot["B_STRUCTURAL_REFRAME"],
            empty_reason=(
                None
                if by_slot["B_STRUCTURAL_REFRAME"]
                else "No reviewed candidate met the structural-reframe routing conditions."
            ),
        ),
        ReframePortfolioSlot(
            slot_id="C_MEASUREMENT_REGIME_CHALLENGE",
            label="Measurement/regime challenge",
            purpose=(
                "Grounded regime or measurement challenges retained separately from "
                "general structural reframes."
            ),
            candidate_ids=by_slot["C_MEASUREMENT_REGIME_CHALLENGE"],
            empty_reason=(
                None
                if by_slot["C_MEASUREMENT_REGIME_CHALLENGE"]
                else "No reviewed low-risk regime/measurement challenge is available."
            ),
        ),
        ReframePortfolioSlot(
            slot_id="D_HIGH_RISK_EXPLORATORY",
            label="High-risk exploratory",
            purpose=(
                "Deep or structurally interesting reframes whose evidence fidelity or "
                "specificity warrants explicit high-risk separation."
            ),
            candidate_ids=by_slot["D_HIGH_RISK_EXPLORATORY"],
            empty_reason=(
                None
                if by_slot["D_HIGH_RISK_EXPLORATORY"]
                else "No reviewed candidate triggered the high-risk exploratory route."
            ),
        ),
    ]

    non_dominated = sorted(
        row.candidate_id for row in assessments if row.pareto_status == "non_dominated"
    )
    dominated = sorted(
        row.candidate_id for row in assessments if row.pareto_status == "dominated"
    )
    not_evaluable = sorted(
        row.candidate_id for row in assessments if row.pareto_status == "not_evaluable"
    )

    portfolio_id = stable_shadow_portfolio_id(
        shadow_report_id=shadow.report_id,
        critic_report_id=critic.report_id,
        assessment_ids=[row.candidate_id for row in assessments],
    )
    return ScientificReframeShadowPortfolio(
        portfolio_id=portfolio_id,
        source_shadow_report_id=shadow.report_id,
        source_critic_report_id=critic.report_id,
        source_task_id=shadow.source_task_id,
        source_context_id=shadow.source_context_id,
        source_context_sha256=shadow.source_context_sha256,
        assessments=assessments,
        slots=slots,
        non_dominated_candidate_ids=non_dominated,
        dominated_candidate_ids=dominated,
        not_evaluable_candidate_ids=not_evaluable,
    )
