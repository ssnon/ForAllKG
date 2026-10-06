from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio

from domains.sers.integrated_experiment_cycle_contracts import (
    SERSExperimentComparisonDesign,
    SERSFailureAttributionCandidate,
    SERSForAllKGFeedbackCandidate,
    SERSIntegratedExperimentFollowup,
    SERSIntegratedExperimentFollowupBundle,
    SERSIntegratedExperimentResultBundle,
    SERSIntegratedExperimentResultRecord,
    SERSIntegratedExperimentResultSubmissionBundle,
    SERSIntegratedExperimentReviewPlan,
    SERSIntegratedExperimentReviewPlanBundle,
)
from domains.sers.validation_orchestration_contracts import (
    SERSIntegratedExperimentRequirement,
    SERSRouteEvidenceBundle,
    SERSRouteEvidenceRecord,
    SERSValidationOrchestrationBundle,
)
from domains.sers.validation_review_contracts import (
    SERSHypothesisScientificReview,
    SERSRouteScientificAssessment,
    SERSValidationReviewBundle,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


_PLANNER_VERSION = "sers-integrated-experiment-review-planner-v0"
_RESULT_COMPILER_VERSION = "sers-integrated-experiment-result-compiler-v0"
_FOLLOWUP_VERSION = "sers-integrated-experiment-followup-v0"


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(_canonical(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:20]}"


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _integrated_route(plan) -> SERSValidationRoute | None:
    rows = [row for row in plan.routes if row.route_kind == "integrated_sers_experiment"]
    if len(rows) > 1:
        raise ValueError(
            f"hypothesis {plan.hypothesis_id} has multiple integrated SERS routes; "
            "v0 requires one unambiguous integrated route"
        )
    return rows[0] if rows else None


def _requirement_for_route(
    orchestration: SERSValidationOrchestrationBundle,
    hypothesis_id: str,
    route_id: str,
) -> SERSIntegratedExperimentRequirement:
    rows = [
        row
        for row in orchestration.experiment_requirements
        if row.hypothesis_id == hypothesis_id
        and row.source_validation_route_id == route_id
    ]
    if len(rows) != 1:
        raise ValueError(
            "expected exactly one integrated experiment requirement for "
            f"hypothesis={hypothesis_id}, route={route_id}; found {len(rows)}"
        )
    return rows[0]


def _prediction_statements(card: HypothesisCard) -> list[str]:
    return [
        f"{row.observable}: expected_direction={row.expected_direction}; rationale={row.rationale}"
        for row in card.predicted_observations
    ]


def _falsification_statements(card: HypothesisCard) -> list[str]:
    return [
        f"{row.observable}: {row.falsifying_outcome}"
        for row in card.falsification_criteria
    ]


def _preserved_uncertainties(review: SERSHypothesisScientificReview) -> list[str]:
    rows: list[str] = []
    for assessment in review.route_assessments:
        if assessment.route_kind == "integrated_sers_experiment":
            continue
        if assessment.verdict in {"not_reviewable", "indeterminate", "mixed"}:
            if assessment.blockers:
                rows.extend(
                    f"{assessment.route_kind}:{blocker}"
                    for blocker in assessment.blockers
                )
            else:
                rows.append(
                    f"{assessment.route_kind}:scientific_verdict={assessment.verdict}"
                )
    return _unique(rows)


class SERSIntegratedExperimentReviewPlanner:
    """Turn an S4.1 experiment-review candidate into a non-procedural test design.

    This planner never creates a fabrication recipe or executable wet-lab protocol.
    It preserves the routed hypothesis, prediction/falsifier lineage, comparison
    requirements, and unresolved protocol dimensions so a human or future approved
    experimental planner can decide whether/how to execute the test.
    """

    planner_version = _PLANNER_VERSION

    def plan(
        self,
        portfolio: HypothesisPortfolio,
        validation_plans: SERSHypothesisValidationPlanBundle,
        orchestration: SERSValidationOrchestrationBundle,
        scientific_review: SERSValidationReviewBundle,
    ) -> SERSIntegratedExperimentReviewPlanBundle:
        if portfolio.domain_profile_id != "sers_au_ag":
            raise ValueError("integrated SERS experiment planning requires sers_au_ag")
        if validation_plans.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("validation-plan/portfolio lineage mismatch")
        if orchestration.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("orchestration/portfolio lineage mismatch")
        if scientific_review.source_portfolio_id != portfolio.portfolio_id:
            raise ValueError("scientific-review/portfolio lineage mismatch")
        if scientific_review.source_validation_plan_bundle_id != validation_plans.bundle_id:
            raise ValueError("scientific-review/validation-plan lineage mismatch")
        if scientific_review.source_orchestration_bundle_id != orchestration.bundle_id:
            raise ValueError("scientific-review/orchestration lineage mismatch")

        cards = {row.hypothesis_id: row for row in portfolio.hypotheses}
        plans = {row.hypothesis_id: row for row in validation_plans.plans}
        reviews = {row.hypothesis_id: row for row in scientific_review.reviews}
        states = {row.hypothesis_id: row for row in orchestration.states}

        outputs: list[SERSIntegratedExperimentReviewPlan] = []
        for hypothesis_id, plan in plans.items():
            route = _integrated_route(plan)
            if route is None:
                continue
            card = cards.get(hypothesis_id)
            review = reviews.get(hypothesis_id)
            state = states.get(hypothesis_id)
            if card is None or review is None or state is None:
                raise ValueError(
                    f"missing card/review/orchestration lineage for {hypothesis_id}"
                )

            requirement = _requirement_for_route(
                orchestration, hypothesis_id, route.route_id
            )

            blockers: list[str] = []
            if review.experiment_review_state not in {
                "review_candidate",
                "direct_test_review_candidate",
            }:
                plan_status = "not_candidate"
                priority = "not_prioritized"
                blockers.append(
                    f"scientific_review_state:{review.experiment_review_state}"
                )
            elif review.experiment_review_state == "direct_test_review_candidate":
                plan_status = "direct_test_review_candidate"
                priority = "preferred_over_deferred_high_cost_em_escalation"
            else:
                plan_status = "review_candidate"
                priority = "candidate_after_human_review"

            if plan_status == "direct_test_review_candidate":
                rationale = (
                    "The integrated experiment is an information-gain candidate because "
                    "non-EM mechanism evidence has been reviewed while unresolved classical-EM "
                    "evidence is explicitly preserved and high-cost numerical escalation is deferred."
                )
            elif plan_status == "review_candidate":
                rationale = (
                    "The integrated outcome is ready for human experiment-design review; execution "
                    "still requires explicit protocol, replication, instrumentation, and acceptance decisions."
                )
            else:
                rationale = (
                    "The current scientific review does not nominate this hypothesis for integrated "
                    "experiment design review. Preserve the requirement without promoting execution."
                )

            outputs.append(SERSIntegratedExperimentReviewPlan(
                plan_id=_stable_id(
                    "sers_integrated_experiment_review_plan",
                    hypothesis_id,
                    review.review_id,
                    state.state_id,
                    requirement.requirement_id,
                    plan_status,
                    self.planner_version,
                ),
                hypothesis_id=hypothesis_id,
                source_scientific_review_id=review.review_id,
                source_validation_state_id=state.state_id,
                source_experiment_requirement_id=requirement.requirement_id,
                source_validation_route_id=route.route_id,
                plan_status=plan_status,
                information_priority=priority,
                comparison_design=SERSExperimentComparisonDesign(
                    focal_hypothesis_statement=card.hypothesis_statement,
                    target_observables=list(requirement.route_target_observables),
                    prediction_statements=_prediction_statements(card),
                    falsification_statements=_falsification_statements(card),
                    comparison_requirements=list(requirement.comparison_requirements),
                    control_principles=list(requirement.control_principles),
                    unresolved_protocol_dimensions=list(
                        requirement.unresolved_protocol_dimensions
                    ),
                ),
                result_interpretation_rules=[
                    "supportive only when a reviewed experimental outcome matches the predeclared routed prediction under an adequate controlled comparison",
                    "inconsistent only when a reviewed experimental outcome satisfies a predeclared falsification relation for the routed integrated outcome",
                    "mixed when decisive integrated measurements disagree across valid comparisons or result units",
                    "indeterminate when data quality, controls, protocol deviations, or scope prevent a directional claim assessment",
                ],
                preserved_uncertainties=_preserved_uncertainties(review),
                blockers=blockers,
                rationale=rationale,
            ))

        return SERSIntegratedExperimentReviewPlanBundle(
            bundle_id=_stable_id(
                "sers_integrated_experiment_review_plan_bundle",
                portfolio.portfolio_id,
                validation_plans.bundle_id,
                orchestration.bundle_id,
                scientific_review.bundle_id,
                *(row.plan_id for row in outputs),
                self.planner_version,
            ),
            source_portfolio_id=portfolio.portfolio_id,
            source_validation_plan_bundle_id=validation_plans.bundle_id,
            source_orchestration_bundle_id=orchestration.bundle_id,
            source_scientific_review_bundle_id=scientific_review.bundle_id,
            plans=outputs,
            plan_count=len(outputs),
        )


class SERSIntegratedExperimentResultCompiler:
    """Normalize reviewed experimental outcomes and merge them into route evidence.

    The compiler does not infer relation_to_claim from raw measurements.  That
    relation must be supplied after a human/validator assessment against the
    predeclared experiment-review plan.  This prevents the runtime from fabricating
    a scientific interpretation from incomplete laboratory metadata.
    """

    compiler_version = _RESULT_COMPILER_VERSION

    def compile(
        self,
        experiment_plans: SERSIntegratedExperimentReviewPlanBundle,
        submissions: SERSIntegratedExperimentResultSubmissionBundle,
        *,
        existing_route_evidence: SERSRouteEvidenceBundle | None = None,
    ) -> tuple[SERSIntegratedExperimentResultBundle, SERSRouteEvidenceBundle]:
        plan_by_id = {row.plan_id: row for row in experiment_plans.plans}
        if len(plan_by_id) != len(experiment_plans.plans):
            raise ValueError("duplicate integrated experiment plan_id")

        existing_records = (
            list(existing_route_evidence.records)
            if existing_route_evidence is not None
            else []
        )
        existing_ids = {row.evidence_id for row in existing_records}
        new_records: list[SERSRouteEvidenceRecord] = []
        results: list[SERSIntegratedExperimentResultRecord] = []

        for submission in submissions.submissions:
            plan = plan_by_id.get(submission.experiment_plan_id)
            if plan is None:
                raise ValueError(
                    f"experimental result references unknown experiment plan {submission.experiment_plan_id}"
                )
            if submission.hypothesis_id != plan.hypothesis_id:
                raise ValueError("experimental result/plan hypothesis lineage mismatch")
            if submission.experiment_requirement_id != plan.source_experiment_requirement_id:
                raise ValueError("experimental result/requirement lineage mismatch")
            if plan.plan_status == "not_candidate":
                raise ValueError(
                    "experimental result intake through this cycle requires an experiment-review candidate; "
                    "record out-of-band experiments through a separate provenance path"
                )

            readiness = {
                "review_ready": "review_ready",
                "limited": "screened",
                "unresolved": "raw",
            }[submission.result_quality]

            route_evidence_id = _stable_id(
                "sers_route_evidence",
                submission.submission_id,
                plan.source_validation_route_id,
                "experimental_result",
                self.compiler_version,
            )
            if route_evidence_id in existing_ids:
                raise ValueError(f"duplicate route evidence id {route_evidence_id}")

            record = SERSRouteEvidenceRecord(
                evidence_id=route_evidence_id,
                hypothesis_id=submission.hypothesis_id,
                route_id=plan.source_validation_route_id,
                route_kind="integrated_sers_experiment",
                evidence_role="experimental_result",
                review_readiness=readiness,
                relation_to_claim=submission.relation_to_claim,
                target_observables=list(submission.measured_observables),
                source_ids=list(submission.source_ids),
                source_paper_ids=[],
                source_overlap_with_generation=False,
                limitations=_unique([
                    *submission.limitations,
                    f"controls_status={submission.controls_status}",
                    f"protocol_deviation_status={submission.protocol_deviation_status}",
                ]),
                provenance_notes=_unique([
                    *submission.provenance_notes,
                    f"experiment_plan_id={plan.plan_id}",
                    f"experiment_requirement_id={plan.source_experiment_requirement_id}",
                    f"assessment_basis={submission.assessment_basis}",
                    f"measurement_summary={submission.measurement_summary}",
                    f"comparison_summary={submission.comparison_summary}",
                ]),
            )
            new_records.append(record)
            existing_ids.add(route_evidence_id)

            results.append(SERSIntegratedExperimentResultRecord(
                result_id=_stable_id(
                    "sers_integrated_experiment_result",
                    submission.submission_id,
                    route_evidence_id,
                    self.compiler_version,
                ),
                source_submission_id=submission.submission_id,
                hypothesis_id=submission.hypothesis_id,
                source_experiment_plan_id=plan.plan_id,
                source_experiment_requirement_id=plan.source_experiment_requirement_id,
                source_validation_route_id=plan.source_validation_route_id,
                relation_to_claim=submission.relation_to_claim,
                result_quality=submission.result_quality,
                assessment_basis=submission.assessment_basis,
                measurement_summary=submission.measurement_summary,
                measured_observables=list(submission.measured_observables),
                comparison_summary=submission.comparison_summary,
                controls_status=submission.controls_status,
                protocol_deviation_status=submission.protocol_deviation_status,
                source_ids=list(submission.source_ids),
                limitations=list(submission.limitations),
                provenance_notes=list(submission.provenance_notes),
                route_evidence_id=route_evidence_id,
                review_ready_for_scientific_aggregation=(
                    submission.result_quality == "review_ready"
                ),
            ))

        merged_records = existing_records + new_records
        merged_bundle = SERSRouteEvidenceBundle(
            bundle_id=_stable_id(
                "sers_route_evidence_bundle",
                existing_route_evidence.bundle_id if existing_route_evidence else None,
                submissions.bundle_id,
                *(row.evidence_id for row in merged_records),
                self.compiler_version,
            ),
            records=merged_records,
            record_count=len(merged_records),
        )
        result_bundle = SERSIntegratedExperimentResultBundle(
            bundle_id=_stable_id(
                "sers_integrated_experiment_result_bundle",
                experiment_plans.bundle_id,
                submissions.bundle_id,
                *(row.result_id for row in results),
                self.compiler_version,
            ),
            source_experiment_plan_bundle_id=experiment_plans.bundle_id,
            results=results,
            result_count=len(results),
            merged_route_evidence_bundle_id=merged_bundle.bundle_id,
        )
        return result_bundle, merged_bundle


def _assessment_by_kind(
    review: SERSHypothesisScientificReview,
    kind: str,
) -> SERSRouteScientificAssessment | None:
    rows = [row for row in review.route_assessments if row.route_kind == kind]
    if len(rows) > 1:
        raise ValueError(f"multiple route assessments for route_kind={kind}")
    return rows[0] if rows else None


def _attribution(
    *,
    hypothesis_id: str,
    category: str,
    status: str,
    rationale: str,
    next_test_question: str,
    route_kind: str | None = None,
    evidence_ids: list[str] | None = None,
) -> SERSFailureAttributionCandidate:
    return SERSFailureAttributionCandidate(
        attribution_id=_stable_id(
            "sers_failure_attribution",
            hypothesis_id,
            category,
            route_kind,
            rationale,
        ),
        hypothesis_id=hypothesis_id,
        category=category,
        status=status,
        route_kind=route_kind,
        rationale=rationale,
        evidence_ids=_unique(evidence_ids or []),
        next_test_question=next_test_question,
    )


def _feedback_candidate(
    *,
    hypothesis_id: str,
    review_id: str,
    result_ids: list[str],
    kind: str,
    action: str,
    question: str,
    rationale: str,
    route_kind: str | None = None,
) -> SERSForAllKGFeedbackCandidate:
    return SERSForAllKGFeedbackCandidate(
        feedback_candidate_id=_stable_id(
            "sers_forallkg_feedback_candidate",
            hypothesis_id,
            review_id,
            kind,
            route_kind,
            action,
        ),
        hypothesis_id=hypothesis_id,
        feedback_kind=kind,
        target_route_kind=route_kind,
        action=action,
        research_question=question,
        rationale=rationale,
        source_result_ids=_unique(result_ids),
        source_review_id=review_id,
    )


class SERSIntegratedExperimentFollowupAnalyzer:
    """Interpret an updated S4.1 review after integrated experimental evidence.

    Failure attribution is explicitly abductive and candidate-only.  The analyzer
    can propose what to investigate next, but it cannot identify a causal failure
    mechanism, reject the whole hypothesis, rewrite it, or feed canonical state.
    """

    analyzer_version = _FOLLOWUP_VERSION

    def analyze(
        self,
        updated_review: SERSValidationReviewBundle,
        experiment_results: SERSIntegratedExperimentResultBundle,
    ) -> SERSIntegratedExperimentFollowupBundle:
        result_by_hypothesis: dict[str, list[SERSIntegratedExperimentResultRecord]] = {}
        for row in experiment_results.results:
            result_by_hypothesis.setdefault(row.hypothesis_id, []).append(row)

        review_by_hypothesis = {row.hypothesis_id: row for row in updated_review.reviews}
        outputs: list[SERSIntegratedExperimentFollowup] = []

        for hypothesis_id, results in result_by_hypothesis.items():
            review = review_by_hypothesis.get(hypothesis_id)
            if review is None:
                raise ValueError(
                    f"updated scientific review missing experimental hypothesis {hypothesis_id}"
                )
            integrated = _assessment_by_kind(review, "integrated_sers_experiment")
            if integrated is None:
                raise ValueError("updated review missing integrated SERS route assessment")
            if review.experiment_review_state != "experimental_result_available":
                raise ValueError(
                    "followup analysis requires an updated S4.1 review that has consumed "
                    "the experimental result evidence"
                )
            if integrated.experimental_result_count < 1:
                raise ValueError(
                    "updated integrated route assessment does not report experimental results"
                )

            result_ids = [row.result_id for row in results]
            attributions: list[SERSFailureAttributionCandidate] = []
            feedback: list[SERSForAllKGFeedbackCandidate] = []
            failure_required = review.outcome_verdict in {
                "inconsistent",
                "mixed",
                "indeterminate",
            }

            if review.outcome_verdict == "inconsistent":
                em = _assessment_by_kind(review, "classical_em")
                if em is not None:
                    if em.verdict == "not_reviewable":
                        status = "requires_targeted_followup"
                        rationale = (
                            "The integrated outcome is inconsistent while the classical-EM component "
                            "remains non-reviewable. The EM mechanism/model is therefore unresolved: "
                            "it is neither cleared nor established as the cause of failure."
                        )
                    elif em.verdict == "inconsistent":
                        status = "candidate_explanation_only"
                        rationale = (
                            "Both the integrated outcome and routed classical-EM review are inconsistent; "
                            "the EM mechanism is a plausible failure contributor but causal attribution remains unproven."
                        )
                    else:
                        status = "contextually_deprioritized_not_excluded"
                        rationale = (
                            "Classical-EM evidence is not itself inconsistent, which reduces but does not eliminate "
                            "EM/model-context explanations for the failed integrated outcome."
                        )
                    attributions.append(_attribution(
                        hypothesis_id=hypothesis_id,
                        category="classical_em_model_or_mechanism",
                        status=status,
                        route_kind="classical_em",
                        rationale=rationale,
                        evidence_ids=list(em.evidence_ids),
                        next_test_question=(
                            "Does a model or measurement better matched to the realized substrate preserve the proposed EM mechanism?"
                        ),
                    ))
                    if em.verdict in {"not_reviewable", "inconsistent"}:
                        feedback.append(_feedback_candidate(
                            hypothesis_id=hypothesis_id,
                            review_id=review.review_id,
                            result_ids=result_ids,
                            kind="revisit_component_mechanism",
                            route_kind="classical_em",
                            action="reassess_classical_em_mechanism_against_realized_experiment",
                            question=(
                                "Does the electromagnetic mechanism survive when evaluated against the experimentally realized geometry and conditions?"
                            ),
                            rationale=rationale,
                        ))

                molecular = _assessment_by_kind(review, "molecular_spectroscopy")
                surface = _assessment_by_kind(review, "surface_chemistry")
                chem_rows = [row for row in (molecular, surface) if row is not None]
                if chem_rows:
                    chem_inconsistent = any(row.verdict == "inconsistent" for row in chem_rows)
                    chem_unresolved = any(
                        row.verdict in {"not_reviewable", "mixed", "indeterminate"}
                        for row in chem_rows
                    )
                    if chem_inconsistent or chem_unresolved:
                        status = "requires_targeted_followup"
                    else:
                        status = "contextually_deprioritized_not_excluded"
                    attributions.append(_attribution(
                        hypothesis_id=hypothesis_id,
                        category="molecular_or_surface_chemistry",
                        status=status,
                        route_kind=(
                            "surface_chemistry"
                            if surface is not None
                            else "molecular_spectroscopy"
                        ),
                        rationale=(
                            "Existing molecular/surface evidence does not by itself establish transferability to the "
                            "specific realized substrate, adsorption state, orientation, and measurement context."
                        ),
                        evidence_ids=[
                            evidence_id
                            for row in chem_rows
                            for evidence_id in row.evidence_ids
                        ],
                        next_test_question=(
                            "Do adsorption state, orientation, or surface-specific molecular response differ under the realized experimental conditions?"
                        ),
                    ))

                fabrication = _assessment_by_kind(review, "fabrication_process")
                if fabrication is not None:
                    fab_status = (
                        "requires_targeted_followup"
                        if fabrication.verdict in {"not_reviewable", "inconsistent", "mixed", "indeterminate"}
                        else "contextually_deprioritized_not_excluded"
                    )
                    fab_evidence = list(fabrication.evidence_ids)
                else:
                    fab_status = "candidate_explanation_only"
                    fab_evidence = []
                attributions.append(_attribution(
                    hypothesis_id=hypothesis_id,
                    category="fabrication_realization",
                    status=fab_status,
                    route_kind="fabrication_process" if fabrication is not None else None,
                    rationale=(
                        "The integrated experiment tests a realized substrate, so deviation between intended and realized morphology/composition "
                        "remains a possible explanation unless characterization excludes it."
                    ),
                    evidence_ids=fab_evidence,
                    next_test_question=(
                        "Did the fabricated substrate realize the morphology, composition, and variability assumed by the scientific comparison?"
                    ),
                ))
                feedback.append(_feedback_candidate(
                    hypothesis_id=hypothesis_id,
                    review_id=review.review_id,
                    result_ids=result_ids,
                    kind="inspect_fabrication_realization",
                    action="inspect_realized_substrate_against_design_assumptions",
                    question=(
                        "Did fabrication and characterization produce the substrate state that the proposed mechanism presupposed?"
                    ),
                    rationale=(
                        "An inconsistent integrated outcome cannot be assigned to mechanism failure until realized-substrate deviations are considered."
                    ),
                ))

                result_quality_issue = any(
                    row.result_quality != "review_ready"
                    or row.controls_status != "adequate"
                    or row.protocol_deviation_status != "none"
                    for row in results
                )
                if result_quality_issue:
                    attributions.append(_attribution(
                        hypothesis_id=hypothesis_id,
                        category="measurement_or_protocol_confound",
                        status="requires_targeted_followup",
                        rationale=(
                            "At least one experimental result carries limited controls, quality, or protocol-deviation metadata, "
                            "so measurement/process confounding remains active."
                        ),
                        next_test_question=(
                            "Does the integrated outcome persist under a reviewed repeat with adequate controls and resolved protocol deviations?"
                        ),
                    ))
                    feedback.append(_feedback_candidate(
                        hypothesis_id=hypothesis_id,
                        review_id=review.review_id,
                        result_ids=result_ids,
                        kind="repeat_or_refine_integrated_experiment",
                        action="repeat_or_refine_integrated_experiment_before_mechanism_rejection",
                        question=(
                            "Does the inconsistent outcome reproduce after resolving control, measurement, and protocol-quality limitations?"
                        ),
                        rationale=(
                            "Experimental quality limitations should be resolved before converting the outcome into a mechanistic conclusion."
                        ),
                    ))

                attributions.append(_attribution(
                    hypothesis_id=hypothesis_id,
                    category="interaction_nonadditivity_or_alternative_mechanism",
                    status="candidate_explanation_only",
                    rationale=(
                        "The integrated SERS outcome can differ from individually routed components because their interaction is non-additive "
                        "or because an unmodeled mechanism dominates in the realized system."
                    ),
                    next_test_question=(
                        "Which interaction or alternative mechanism could explain the integrated outcome without contradicting individually supported components?"
                    ),
                ))
                feedback.append(_feedback_candidate(
                    hypothesis_id=hypothesis_id,
                    review_id=review.review_id,
                    result_ids=result_ids,
                    kind="investigate_alternative_mechanism",
                    action="generate_targeted_followup_questions_for_failed_integrated_outcome",
                    question=(
                        "What alternative or interaction mechanism best explains the discrepancy between routed component evidence and the integrated experiment?"
                    ),
                    rationale=(
                        "A failed integrated prediction is evidence for revisiting the mechanistic decomposition, not automatic rejection of every routed component."
                    ),
                ))
                interpretation = (
                    "The integrated outcome is provisionally inconsistent. Failure attribution remains abductive: "
                    "candidate explanations are proposed for targeted follow-up, not asserted as causes."
                )

            elif review.outcome_verdict == "supportive":
                feedback.append(_feedback_candidate(
                    hypothesis_id=hypothesis_id,
                    review_id=review.review_id,
                    result_ids=result_ids,
                    kind="preserve_outcome_support_with_mechanism_uncertainty",
                    action="retain_supportive_integrated_outcome_without_upgrading_unresolved_components",
                    question=(
                        "Which unresolved mechanism components should be tested next to explain the supportive integrated outcome rather than merely predict it?"
                    ),
                    rationale=(
                        "A supportive integrated result strengthens the routed outcome but does not validate unresolved EM/chemical mechanisms or confirm the whole hypothesis."
                    ),
                ))
                interpretation = (
                    "The integrated outcome is provisionally supportive. Preserve any unresolved mechanism routes and do not promote the result to whole-hypothesis confirmation."
                )

            elif review.outcome_verdict in {"mixed", "indeterminate"}:
                feedback.append(_feedback_candidate(
                    hypothesis_id=hypothesis_id,
                    review_id=review.review_id,
                    result_ids=result_ids,
                    kind=(
                        "repeat_or_refine_integrated_experiment"
                        if review.outcome_verdict == "mixed"
                        else "preserve_indeterminate_outcome"
                    ),
                    action="resolve_integrated_outcome_ambiguity",
                    question=(
                        "What controlled repeat, stratification, or evidence review would resolve the ambiguous integrated SERS outcome?"
                    ),
                    rationale=(
                        "Mixed or indeterminate experimental evidence should remain unresolved until the source of heterogeneity or uncertainty is characterized."
                    ),
                ))
                attributions.append(_attribution(
                    hypothesis_id=hypothesis_id,
                    category="measurement_or_protocol_confound",
                    status="candidate_explanation_only",
                    rationale=(
                        "Experimental heterogeneity, controls, or protocol context may contribute to the mixed/indeterminate integrated outcome."
                    ),
                    next_test_question=(
                        "Can the ambiguous outcome be localized to measurement quality, sample heterogeneity, or comparison context?"
                    ),
                ))
                interpretation = (
                    "The integrated outcome remains mixed or indeterminate; preserve uncertainty and resolve experimental/contextual heterogeneity before mechanism revision."
                )
            else:
                failure_required = False
                interpretation = (
                    "The updated scientific review does not yet contain a decisive integrated outcome; no failure attribution is performed."
                )

            outputs.append(SERSIntegratedExperimentFollowup(
                followup_id=_stable_id(
                    "sers_integrated_experiment_followup",
                    hypothesis_id,
                    review.review_id,
                    *(result_ids),
                    review.outcome_verdict,
                    self.analyzer_version,
                ),
                hypothesis_id=hypothesis_id,
                source_scientific_review_id=review.review_id,
                source_experiment_result_ids=result_ids,
                outcome_verdict=review.outcome_verdict,
                mechanism_verdict=review.pre_experiment_mechanism_verdict,
                failure_analysis_required=failure_required,
                attribution_candidates=attributions,
                feedback_candidates=feedback,
                interpretation=interpretation,
            ))

        return SERSIntegratedExperimentFollowupBundle(
            bundle_id=_stable_id(
                "sers_integrated_experiment_followup_bundle",
                updated_review.bundle_id,
                experiment_results.bundle_id,
                *(row.followup_id for row in outputs),
                self.analyzer_version,
            ),
            source_scientific_review_bundle_id=updated_review.bundle_id,
            source_experiment_result_bundle_id=experiment_results.bundle_id,
            followups=outputs,
            followup_count=len(outputs),
        )
