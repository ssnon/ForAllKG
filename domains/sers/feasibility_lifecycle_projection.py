from __future__ import annotations

import hashlib

from pipeline_core.discovery.candidate_contracts import CandidateDecisionPortfolio

from domains.sers.feasibility_lifecycle_projection_contracts import (
    SERSCanonicalLifecycleProjection,
    SERSCanonicalLifecycleProjectionBundle,
)
from domains.sers.integrated_experiment_cycle_contracts import (
    SERSIntegratedExperimentFollowupBundle,
    SERSIntegratedExperimentReviewPlanBundle,
)
from domains.sers.validation_review_contracts import SERSValidationReviewBundle


_PROJECTION_VERSION = "sers-canonical-lifecycle-projection-v0"


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


class SERSCanonicalLifecycleProjector:
    """Project SERS shadow lifecycle state beside canonical feasibility output.

    This is deliberately not an authority bridge.  Canonical candidate decisions
    remain unchanged, while the projection makes route-level SERS research state,
    experiment information priority, and claim-scoped feedback visible to the
    same hypothesis lineage.
    """

    projection_version = _PROJECTION_VERSION

    def project(
        self,
        candidate_decisions: CandidateDecisionPortfolio,
        scientific_review: SERSValidationReviewBundle,
        *,
        experiment_plans: SERSIntegratedExperimentReviewPlanBundle | None = None,
        followups: SERSIntegratedExperimentFollowupBundle | None = None,
    ) -> SERSCanonicalLifecycleProjectionBundle:
        decisions = {row.hypothesis_id: row for row in candidate_decisions.cards}
        reviews = {row.hypothesis_id: row for row in scientific_review.reviews}
        if set(decisions) != set(reviews):
            missing_review = sorted(set(decisions) - set(reviews))
            missing_decision = sorted(set(reviews) - set(decisions))
            raise ValueError(
                "canonical/lifecycle hypothesis coverage mismatch; "
                f"missing_review={missing_review}; missing_decision={missing_decision}"
            )

        plan_by_hypothesis = {}
        if experiment_plans is not None:
            for row in experiment_plans.plans:
                if row.hypothesis_id in plan_by_hypothesis:
                    raise ValueError(
                        "multiple integrated experiment review plans for one hypothesis"
                    )
                plan_by_hypothesis[row.hypothesis_id] = row
            unknown = sorted(set(plan_by_hypothesis) - set(reviews))
            if unknown:
                raise ValueError(
                    "experiment plan references hypothesis absent from scientific review: "
                    + ", ".join(unknown)
                )

        followup_by_hypothesis = {}
        if followups is not None:
            for row in followups.followups:
                if row.hypothesis_id in followup_by_hypothesis:
                    raise ValueError(
                        "multiple integrated experiment followups for one hypothesis"
                    )
                followup_by_hypothesis[row.hypothesis_id] = row
            unknown = sorted(set(followup_by_hypothesis) - set(reviews))
            if unknown:
                raise ValueError(
                    "followup references hypothesis absent from scientific review: "
                    + ", ".join(unknown)
                )

        projections: list[SERSCanonicalLifecycleProjection] = []
        for hypothesis_id in sorted(reviews):
            decision = decisions[hypothesis_id]
            review = reviews[hypothesis_id]
            plan = plan_by_hypothesis.get(hypothesis_id)
            followup = followup_by_hypothesis.get(hypothesis_id)

            preserved_uncertainties: list[str] = []
            unresolved_route_kinds: list[str] = []
            for route in review.route_assessments:
                if route.verdict == "not_reviewable" or route.blockers:
                    unresolved_route_kinds.append(route.route_kind)
                    preserved_uncertainties.extend(
                        f"{route.route_kind}:{blocker}" for blocker in route.blockers
                    )
            if plan is not None:
                preserved_uncertainties.extend(plan.preserved_uncertainties)

            claim_actions = [row.action for row in review.feedback]
            followup_actions = (
                [row.action for row in followup.feedback_candidates]
                if followup is not None
                else []
            )

            conflict = decision.final_disposition in {
                "rejected_physical",
                "rejected_experimental",
            }
            if conflict:
                interpretation = (
                    "The canonical candidate decision is a rejection while the SERS "
                    "lifecycle projection remains authority-neutral. This is an authority "
                    "conflict for human/policy review; the projection does not override it."
                )
            elif followup is not None:
                interpretation = (
                    "Canonical feasibility and the SERS multi-validator lifecycle share "
                    "hypothesis lineage. Experimental followup is visible as claim-scoped "
                    "research guidance only; no canonical feedback or candidate authority "
                    "is created."
                )
            elif plan is not None:
                interpretation = (
                    "The canonical feasibility decision remains unchanged while the SERS "
                    "lifecycle records the integrated experiment as an information-gain "
                    "candidate and preserves unresolved mechanism/model uncertainty."
                )
            else:
                interpretation = (
                    "The canonical feasibility decision remains unchanged while the SERS "
                    "lifecycle supplies route-level research state and next-action context."
                )

            projections.append(
                SERSCanonicalLifecycleProjection(
                    projection_id=_stable_id(
                        "sers_canonical_lifecycle_projection",
                        decision.decision_id,
                        review.review_id,
                        plan.plan_id if plan is not None else "",
                        followup.followup_id if followup is not None else "",
                        self.projection_version,
                    ),
                    hypothesis_id=hypothesis_id,
                    source_candidate_decision_id=decision.decision_id,
                    source_scientific_review_id=review.review_id,
                    source_experiment_plan_id=(
                        plan.plan_id if plan is not None else None
                    ),
                    source_followup_id=(
                        followup.followup_id if followup is not None else None
                    ),
                    canonical_final_disposition=decision.final_disposition,
                    sers_research_decision=review.research_decision,
                    pre_experiment_mechanism_verdict=(
                        review.pre_experiment_mechanism_verdict
                    ),
                    outcome_verdict=review.outcome_verdict,
                    experiment_review_state=review.experiment_review_state,
                    experiment_plan_status=(
                        plan.plan_status if plan is not None else None
                    ),
                    experiment_information_priority=(
                        plan.information_priority if plan is not None else None
                    ),
                    preserved_uncertainties=_unique(preserved_uncertainties),
                    claim_scoped_feedback_actions=_unique(claim_actions),
                    followup_feedback_actions=_unique(followup_actions),
                    unresolved_route_kinds=_unique(unresolved_route_kinds),
                    interpretation=interpretation,
                    authority_conflict_detected=conflict,
                )
            )

        return SERSCanonicalLifecycleProjectionBundle(
            bundle_id=_stable_id(
                "sers_canonical_lifecycle_projection_bundle",
                candidate_decisions.decision_portfolio_id,
                scientific_review.bundle_id,
                experiment_plans.bundle_id if experiment_plans is not None else "",
                followups.bundle_id if followups is not None else "",
                *(row.projection_id for row in projections),
                self.projection_version,
            ),
            source_candidate_decision_portfolio_id=(
                candidate_decisions.decision_portfolio_id
            ),
            source_scientific_review_bundle_id=scientific_review.bundle_id,
            source_experiment_plan_bundle_id=(
                experiment_plans.bundle_id if experiment_plans is not None else None
            ),
            source_followup_bundle_id=(
                followups.bundle_id if followups is not None else None
            ),
            projections=projections,
            projection_count=len(projections),
        )
