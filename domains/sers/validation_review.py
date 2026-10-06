from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio

from domains.sers.validation_orchestration_contracts import (
    SERSHypothesisValidationState,
    SERSRouteEvidenceBundle,
    SERSRouteEvidenceRecord,
    SERSRouteValidationState,
    SERSValidationOrchestrationBundle,
)
from domains.sers.validation_review_contracts import (
    SERSClaimScopedResearchFeedback,
    SERSHypothesisScientificReview,
    SERSRouteScientificAssessment,
    SERSValidationReviewBundle,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlan,
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


_REVIEWER_VERSION = "sers-multi-validator-scientific-review-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _independent_record(record: SERSRouteEvidenceRecord, card: HypothesisCard) -> bool:
    if record.review_readiness != "review_ready":
        return False
    if set(record.source_paper_ids) & set(card.source_paper_ids):
        return False
    if record.source_overlap_with_generation is True:
        return False
    if record.evidence_role == "experimental_result":
        return True
    if record.evidence_role != "independent_validation":
        return False
    if record.source_overlap_with_generation is False:
        return True
    return bool(record.source_paper_ids)


def _source_keys(record: SERSRouteEvidenceRecord) -> list[str]:
    if record.source_paper_ids:
        return [f"paper:{value}" for value in record.source_paper_ids]
    if record.source_ids:
        return [f"source:{value}" for value in record.source_ids]
    return [f"evidence:{record.evidence_id}"]


def _evidence_strength(records: list[SERSRouteEvidenceRecord]) -> tuple[str, int, int]:
    source_keys = sorted({key for row in records for key in _source_keys(row)})
    experimental_count = sum(row.evidence_role == "experimental_result" for row in records)
    independent_count = sum(row.evidence_role == "independent_validation" for row in records)

    if experimental_count and independent_count:
        strength = "mixed_source_types"
    elif experimental_count:
        strength = "experimental_result"
    elif len(source_keys) >= 2:
        strength = "multiple_independent_sources"
    elif len(source_keys) == 1:
        strength = "single_independent_source"
    else:
        strength = "none"
    return strength, len(source_keys), experimental_count


def _relation_verdict(records: list[SERSRouteEvidenceRecord]) -> tuple[str, list[str], str]:
    decisive = [
        row
        for row in records
        if row.relation_to_claim in {"supportive", "inconsistent", "mixed"}
    ]
    relations = {row.relation_to_claim for row in decisive}
    if "mixed" in relations or {"supportive", "inconsistent"} <= relations:
        return (
            "mixed",
            [row.evidence_id for row in decisive],
            "Independent review-ready evidence contains conflicting or explicitly mixed relations to the routed claim.",
        )
    if relations == {"supportive"}:
        return (
            "supportive",
            [row.evidence_id for row in decisive],
            "All decisive independent review-ready evidence is reported as supportive of the routed claim.",
        )
    if relations == {"inconsistent"}:
        return (
            "inconsistent",
            [row.evidence_id for row in decisive],
            "All decisive independent review-ready evidence is reported as inconsistent with the routed claim.",
        )
    return (
        "indeterminate",
        [],
        "Review-ready evidence exists, but it does not provide a decisive relation to the routed claim.",
    )


def _route_assessment(
    *,
    card: HypothesisCard,
    route: SERSValidationRoute,
    route_state: SERSRouteValidationState,
    records: list[SERSRouteEvidenceRecord],
) -> SERSRouteScientificAssessment:
    blockers: list[str] = []
    qualified = [row for row in records if _independent_record(row, card)]

    if route_state.state != "review_ready":
        blockers.extend(route_state.blockers)
        verdict = "not_reviewable"
        interpretation = (
            "The routed evidence package is not review-ready; preserve uncertainty "
            "and do not convert numerical/model/evidence-readiness limitations into "
            "scientific contradiction."
        )
        decisive_ids: list[str] = []
        strength, source_count, experimental_count = _evidence_strength(qualified)
    elif route.route_kind == "classical_em":
        # Current EM lineage intentionally separates readiness from scientific
        # interpretation.  A future explicit component-review artifact may feed
        # this branch; readiness alone is never silently treated as support.
        verdict = "indeterminate"
        blockers.append("classical_em_component_interpretation_missing")
        interpretation = (
            "Classical-EM evidence is review-ready in routing terms, but no explicit "
            "scientific component interpretation artifact was supplied."
        )
        decisive_ids = []
        strength, source_count, experimental_count = ("none", 0, 0)
    elif not qualified:
        verdict = "not_reviewable"
        blockers.append("independent_review_ready_evidence_missing")
        interpretation = (
            "The route was marked review-ready upstream, but no independently "
            "qualified evidence record remains after provenance checks."
        )
        decisive_ids = []
        strength, source_count, experimental_count = ("none", 0, 0)
    elif route.route_kind == "integrated_sers_experiment" and not any(
        row.evidence_role == "experimental_result" for row in qualified
    ):
        verdict = "indeterminate"
        blockers.append("hypothesis_specific_experimental_result_missing")
        interpretation = (
            "Literature or external evidence may contextualize the integrated SERS "
            "route, but a hypothesis-specific experimental result is required to "
            "resolve the routed integrated outcome."
        )
        decisive_ids = []
        strength, source_count, experimental_count = _evidence_strength(qualified)
    else:
        verdict, decisive_ids, interpretation = _relation_verdict(qualified)
        strength, source_count, experimental_count = _evidence_strength(qualified)

    return SERSRouteScientificAssessment(
        assessment_id=_stable_id(
            "sers_route_scientific_assessment",
            card.hypothesis_id,
            route.route_id,
            route_state.model_dump(mode="json"),
            [row.model_dump(mode="json") for row in records],
            _REVIEWER_VERSION,
        ),
        hypothesis_id=card.hypothesis_id,
        route_id=route.route_id,
        route_kind=route.route_kind,
        required_for_mechanism_assessment=route.required_for_mechanism_assessment,
        required_for_outcome_assessment=route.required_for_outcome_assessment,
        route_state=route_state.state,
        verdict=verdict,
        evidence_strength=strength,
        evidence_ids=[row.evidence_id for row in records],
        decisive_evidence_ids=decisive_ids,
        independent_source_count=source_count,
        experimental_result_count=experimental_count,
        blockers=_unique(blockers),
        interpretation=interpretation,
    )


def _aggregate_pre_experiment_mechanism(
    plan: SERSHypothesisValidationPlan,
    assessments: list[SERSRouteScientificAssessment],
) -> str:
    if not plan.mechanism_claim_present:
        return "not_claimed"
    rows = [
        row
        for row in assessments
        if row.required_for_mechanism_assessment
        and row.route_kind != "integrated_sers_experiment"
    ]
    if not rows:
        return "indeterminate"
    verdicts = {row.verdict for row in rows}
    if "inconsistent" in verdicts:
        return "inconsistent"
    if "mixed" in verdicts:
        return "mixed"
    if all(row.verdict == "supportive" for row in rows):
        return "supportive"
    if "not_reviewable" in verdicts:
        return "incomplete"
    return "indeterminate"


def _aggregate_outcome(
    assessments: list[SERSRouteScientificAssessment],
) -> str:
    integrated = [row for row in assessments if row.route_kind == "integrated_sers_experiment"]
    if integrated:
        row = integrated[0]
        if row.experimental_result_count == 0:
            return "untested"
        if row.verdict in {"supportive", "inconsistent", "mixed", "indeterminate"}:
            return row.verdict
        return "incomplete"

    rows = [row for row in assessments if row.required_for_outcome_assessment]
    if not rows:
        return "incomplete"
    verdicts = {row.verdict for row in rows}
    if "inconsistent" in verdicts:
        return "inconsistent"
    if "mixed" in verdicts:
        return "mixed"
    if all(row.verdict == "supportive" for row in rows):
        return "supportive"
    if "not_reviewable" in verdicts:
        return "incomplete"
    return "indeterminate"


def _feedback(
    *,
    card: HypothesisCard,
    assessment: SERSRouteScientificAssessment,
    kind: str,
    action: str,
    rationale: str,
) -> SERSClaimScopedResearchFeedback:
    return SERSClaimScopedResearchFeedback(
        feedback_id=_stable_id(
            "sers_claim_scoped_feedback",
            card.hypothesis_id,
            assessment.route_id,
            kind,
            action,
            rationale,
            _REVIEWER_VERSION,
        ),
        hypothesis_id=card.hypothesis_id,
        route_id=assessment.route_id,
        route_kind=assessment.route_kind,
        feedback_kind=kind,
        action=action,
        rationale=rationale,
        evidence_ids=list(assessment.decisive_evidence_ids or assessment.evidence_ids),
    )


def _decision(
    *,
    card: HypothesisCard,
    plan: SERSHypothesisValidationPlan,
    state: SERSHypothesisValidationState,
    route_states: dict[str, SERSRouteValidationState],
    assessments: list[SERSRouteScientificAssessment],
    pre_mechanism: str,
    outcome: str,
) -> tuple[str, str, str | None, str | None, str, str, list[SERSClaimScopedResearchFeedback]]:
    by_route = {row.route_id: row for row in assessments}
    feedback: list[SERSClaimScopedResearchFeedback] = []

    unresolved = [row for row in state.route_states if row.state == "requires_interpretation"]
    if unresolved:
        row = unresolved[0]
        return (
            "not_candidate",
            "resolve_route_interpretation",
            row.route_kind,
            row.route_id,
            "resolve_route_interpretation",
            "A routed claim is semantically unresolved; clarify it before scientific review or experiment planning.",
            feedback,
        )

    inconsistent = [
        row for row in assessments
        if row.required_for_mechanism_assessment
        and row.route_kind != "integrated_sers_experiment"
        and row.verdict == "inconsistent"
    ]
    if inconsistent:
        row = inconsistent[0]
        feedback.append(_feedback(
            card=card,
            assessment=row,
            kind="revise_component_claim",
            action=f"revisit_{row.route_kind}_subclaim",
            rationale=(
                "Independent routed evidence is inconsistent with this component. "
                "Revisit only the implicated subclaim/mechanism; do not reject unrelated "
                "parts of the hypothesis automatically."
            ),
        ))
        return (
            "not_candidate",
            "revise_component_claim",
            row.route_kind,
            row.route_id,
            f"revisit_{row.route_kind}_subclaim",
            "At least one required pre-experiment mechanism component is inconsistent with independent review-ready evidence.",
            feedback,
        )

    mixed = [
        row for row in assessments
        if row.required_for_mechanism_assessment
        and row.route_kind != "integrated_sers_experiment"
        and row.verdict == "mixed"
    ]
    if mixed:
        row = mixed[0]
        feedback.append(_feedback(
            card=card,
            assessment=row,
            kind="resolve_evidence_conflict",
            action=f"resolve_{row.route_kind}_evidence_conflict",
            rationale=(
                "Independent evidence disagrees on the routed component; resolve the "
                "source/context conflict before using the component as a screening gate."
            ),
        ))
        return (
            "not_candidate",
            "resolve_conflicting_component_evidence",
            row.route_kind,
            row.route_id,
            f"resolve_{row.route_kind}_evidence_conflict",
            "A required mechanism component has mixed independent evidence.",
            feedback,
        )

    if outcome == "inconsistent":
        integrated = next(
            (row for row in assessments if row.route_kind == "integrated_sers_experiment"),
            None,
        )
        if integrated is not None:
            feedback.append(_feedback(
                card=card,
                assessment=integrated,
                kind="reassess_mechanism_after_outcome",
                action="analyze_failed_integrated_sers_outcome",
                rationale=(
                    "The hypothesis-specific integrated SERS result is inconsistent. "
                    "Reassess EM, chemistry, fabrication, accessibility, and alternative "
                    "mechanisms rather than attributing failure to one route automatically."
                ),
            ))
        return (
            "experimental_result_available",
            "outcome_inconsistent_reassess_mechanism",
            integrated.route_kind if integrated is not None else None,
            integrated.route_id if integrated is not None else None,
            "analyze_failed_integrated_sers_outcome",
            "The integrated experimental outcome is inconsistent with the routed prediction; perform claim-scoped failure analysis.",
            feedback,
        )

    if outcome in {"supportive", "mixed", "indeterminate"}:
        integrated = next(
            (row for row in assessments if row.route_kind == "integrated_sers_experiment"),
            None,
        )
        if outcome == "supportive" and pre_mechanism not in {"inconsistent", "mixed"}:
            return (
                "experimental_result_available",
                "provisional_routed_support",
                None,
                None,
                "preserve_provisional_routed_support_for_human_scientific_review",
                "The integrated outcome is supportive and no reviewed pre-experiment mechanism component is inconsistent; retain only a provisional routed support state.",
                feedback,
            )
        return (
            "experimental_result_available",
            "review_integrated_experiment_outcome",
            integrated.route_kind if integrated is not None else None,
            integrated.route_id if integrated is not None else None,
            "review_integrated_sers_experiment_outcome",
            "An integrated experimental result exists but is mixed or indeterminate; human scientific interpretation is required.",
            feedback,
        )

    integrated_state = next(
        (row for row in state.route_states if row.route_kind == "integrated_sers_experiment"),
        None,
    )

    pre_assessments = [
        row
        for row in assessments
        if row.required_for_mechanism_assessment
        and row.route_kind != "integrated_sers_experiment"
    ]
    unresolved_pre = [row for row in pre_assessments if row.verdict != "supportive"]

    if not unresolved_pre and integrated_state is not None:
        feedback.append(SERSClaimScopedResearchFeedback(
            feedback_id=_stable_id(
                "sers_claim_scoped_feedback",
                card.hypothesis_id,
                integrated_state.route_id,
                "review_experiment_requirement",
                _REVIEWER_VERSION,
            ),
            hypothesis_id=card.hypothesis_id,
            route_id=integrated_state.route_id,
            route_kind=integrated_state.route_kind,
            feedback_kind="review_experiment_requirement",
            action="review_integrated_sers_experiment_requirement",
            rationale=(
                "All required pre-experiment mechanism components are provisionally "
                "supportive in the shadow review; the integrated outcome remains untested."
            ),
        ))
        return (
            "review_candidate",
            "candidate_for_integrated_experiment_review",
            integrated_state.route_kind,
            integrated_state.route_id,
            "review_integrated_sers_experiment_requirement",
            "All required pre-experiment mechanism components are provisionally supportive; the integrated SERS outcome should be considered for experiment review, not automatic execution.",
            feedback,
        )

    # Important deadlock escape: when the only remaining pre-experiment blocker
    # is an already-observed classical-EM branch whose expensive escalation has
    # explicitly been deferred, do not force the researcher back into deeper FDTD.
    # The integrated experiment can be reviewed as a direct information-gathering
    # option, but this is not equivalent to the EM component having passed.
    if integrated_state is not None and unresolved_pre:
        non_em_unresolved = [row for row in unresolved_pre if row.route_kind != "classical_em"]
        em_unresolved = [row for row in unresolved_pre if row.route_kind == "classical_em"]
        em_deferred = bool(em_unresolved) and all(
            route_states[row.route_id].high_cost_escalation_deferred
            and route_states[row.route_id].state == "evidence_available_not_reviewable"
            for row in em_unresolved
        )
        if not non_em_unresolved and em_deferred:
            em = em_unresolved[0]
            feedback.append(_feedback(
                card=card,
                assessment=em,
                kind="preserve_uncertainty",
                action="preserve_classical_em_uncertainty_without_high_cost_escalation",
                rationale=(
                    "Classical-EM evidence remains non-reviewable and must not be called "
                    "supportive or inconsistent. High-cost escalation is explicitly "
                    "deferred, so retain that uncertainty while considering whether the "
                    "integrated experiment is a better next information source."
                ),
            ))
            return (
                "direct_test_review_candidate",
                "consider_integrated_experiment_review",
                integrated_state.route_kind,
                integrated_state.route_id,
                "compare_integrated_experiment_value_against_deferred_em_escalation",
                "All non-EM pre-experiment mechanism components are review-ready, while the remaining classical-EM uncertainty is already observed and high-cost escalation is deferred. Consider the integrated experiment as a direct test without pretending the EM route passed.",
                feedback,
            )

    # Otherwise inherit the orchestrator's next unresolved research action.
    target_assessment = by_route.get(state.recommended_next_route_id or "")
    if target_assessment is not None:
        kind = (
            "collect_evidence"
            if target_assessment.route_state == "no_evidence"
            else "strengthen_evidence"
        )
        feedback.append(_feedback(
            card=card,
            assessment=target_assessment,
            kind=kind,
            action=state.recommended_next_action,
            rationale=state.rationale,
        ))
    decision = (
        "collect_missing_route_evidence"
        if target_assessment is not None and target_assessment.route_state == "no_evidence"
        else "strengthen_route_evidence"
    )
    return (
        "not_candidate",
        decision,
        state.recommended_next_route_kind,
        state.recommended_next_route_id,
        state.recommended_next_action,
        state.rationale,
        feedback,
    )


class SERSMultiValidatorScientificReviewer:
    """Produce provisional, claim-scoped scientific review from routed evidence.

    The reviewer is intentionally downstream of S4.0 orchestration. It can call a
    routed component supportive/inconsistent only when the route is review-ready
    and independent evidence survives provenance checks. Its outputs remain shadow
    review artifacts: no canonical hypothesis rejection, experiment execution, or
    self-feedback authority is created.
    """

    reviewer_version = _REVIEWER_VERSION

    def review(
        self,
        portfolio: HypothesisPortfolio,
        plans: SERSHypothesisValidationPlanBundle,
        orchestration: SERSValidationOrchestrationBundle,
        *,
        route_evidence: SERSRouteEvidenceBundle | None = None,
    ) -> SERSValidationReviewBundle:
        if portfolio.domain_profile_id != "sers_au_ag":
            raise ValueError("SERS scientific review requires sers_au_ag portfolio")
        if plans.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("validation-plan bundle/portfolio lineage mismatch")
        if orchestration.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("orchestration/portfolio lineage mismatch")
        if orchestration.source_validation_plan_bundle_id != plans.bundle_id:
            raise ValueError("orchestration/validation-plan lineage mismatch")
        if route_evidence is not None:
            source_bundle_id = orchestration.source_route_evidence_bundle_id
            if source_bundle_id is not None and source_bundle_id != route_evidence.bundle_id:
                raise ValueError("orchestration/route-evidence bundle lineage mismatch")

        card_by_id = {row.hypothesis_id: row for row in portfolio.hypotheses}
        plan_by_id = {row.hypothesis_id: row for row in plans.plans}
        state_by_id = {row.hypothesis_id: row for row in orchestration.states}
        evidence_by_route: dict[str, list[SERSRouteEvidenceRecord]] = {}
        if route_evidence is not None:
            for record in route_evidence.records:
                evidence_by_route.setdefault(record.route_id, []).append(record)

        reviews: list[SERSHypothesisScientificReview] = []
        for hypothesis_id, plan in plan_by_id.items():
            card = card_by_id.get(hypothesis_id)
            state = state_by_id.get(hypothesis_id)
            if card is None or state is None:
                raise ValueError(
                    f"missing portfolio/orchestration lineage for hypothesis {hypothesis_id}"
                )
            route_state_by_id = {row.route_id: row for row in state.route_states}
            assessments: list[SERSRouteScientificAssessment] = []

            for route in plan.routes:
                route_state = route_state_by_id.get(route.route_id)
                if route_state is None:
                    raise ValueError(
                        f"orchestration missing route state {route.route_id}"
                    )
                records = evidence_by_route.get(route.route_id, [])
                for record in records:
                    if record.hypothesis_id != hypothesis_id:
                        raise ValueError("route evidence/hypothesis lineage mismatch")
                    if record.route_kind != route.route_kind:
                        raise ValueError("route evidence/route-kind lineage mismatch")
                assessments.append(_route_assessment(
                    card=card,
                    route=route,
                    route_state=route_state,
                    records=records,
                ))

            pre_mechanism = _aggregate_pre_experiment_mechanism(plan, assessments)
            outcome = _aggregate_outcome(assessments)
            (
                experiment_state,
                research_decision,
                next_kind,
                next_route_id,
                next_action,
                rationale,
                feedback,
            ) = _decision(
                card=card,
                plan=plan,
                state=state,
                route_states=route_state_by_id,
                assessments=assessments,
                pre_mechanism=pre_mechanism,
                outcome=outcome,
            )

            reviews.append(SERSHypothesisScientificReview(
                review_id=_stable_id(
                    "sers_hypothesis_scientific_review",
                    hypothesis_id,
                    state.state_id,
                    *(row.assessment_id for row in assessments),
                    pre_mechanism,
                    outcome,
                    research_decision,
                    self.reviewer_version,
                ),
                hypothesis_id=hypothesis_id,
                source_validation_state_id=state.state_id,
                route_assessments=assessments,
                pre_experiment_mechanism_verdict=pre_mechanism,
                outcome_verdict=outcome,
                experiment_review_state=experiment_state,
                research_decision=research_decision,
                recommended_next_route_kind=next_kind,
                recommended_next_route_id=next_route_id,
                recommended_next_action=next_action,
                rationale=rationale,
                feedback=feedback,
            ))

        return SERSValidationReviewBundle(
            bundle_id=_stable_id(
                "sers_validation_review_bundle",
                portfolio.portfolio_id,
                plans.bundle_id,
                orchestration.bundle_id,
                route_evidence.bundle_id if route_evidence is not None else None,
                *(row.review_id for row in reviews),
                self.reviewer_version,
            ),
            source_portfolio_id=portfolio.portfolio_id,
            source_validation_plan_bundle_id=plans.bundle_id,
            source_orchestration_bundle_id=orchestration.bundle_id,
            source_route_evidence_bundle_id=(
                route_evidence.bundle_id if route_evidence is not None else None
            ),
            reviews=reviews,
            review_count=len(reviews),
        )
