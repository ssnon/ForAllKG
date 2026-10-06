from __future__ import annotations

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from domains.sers.integrated_experiment_cycle import (
    SERSIntegratedExperimentFollowupAnalyzer,
    SERSIntegratedExperimentResultCompiler,
    SERSIntegratedExperimentReviewPlanner,
)
from domains.sers.integrated_experiment_cycle_contracts import (
    SERSIntegratedExperimentResultSubmission,
    SERSIntegratedExperimentResultSubmissionBundle,
)
from domains.sers.validation_orchestration_contracts import (
    SERSHypothesisValidationState,
    SERSIntegratedExperimentRequirement,
    SERSRouteEvidenceBundle,
    SERSRouteEvidenceRecord,
    SERSRouteValidationState,
    SERSValidationOrchestrationBundle,
)
from domains.sers.validation_review_contracts import (
    SERSHypothesisScientificReview,
    SERSRouteScientificAssessment,
    SERSValidationReviewBundle,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlan,
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


def _portfolio():
    card = HypothesisCard(
        hypothesis_id="hypothesis:1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="Integrated SERS excitation optimum",
        hypothesis_statement=(
            "The integrated SERS optimum shifts relative to the substrate-only reference "
            "because molecular excitation dependence contributes to the measured response."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        inferential_bridge=(
            "Molecular excitation dependence and substrate electromagnetic response "
            "jointly contribute to the measured SERS optimum."
        ),
        predicted_observations=[PredictedObservation(
            observation_id="prediction:1",
            observable="excitation wavelength producing maximum SERS intensity",
            expected_direction="shift",
            rationale="molecular and substrate responses interact",
        )],
        falsification_criteria=[FalsificationCriterion(
            criterion_id="falsifier:1",
            observable="excitation wavelength producing maximum SERS intensity",
            falsifying_outcome="The controlled integrated optimum does not differ from the substrate-only reference.",
        )],
        assumptions=[],
        source_paper_ids=["paper:generation"],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )
    return HypothesisPortfolio(
        portfolio_id="portfolio:1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        hypotheses=[card],
    )


def _plans():
    em = SERSValidationRoute(
        route_id="route:em",
        route_kind="classical_em",
        statement="EM",
        rationale="fixture",
        target_observables=["substrate electromagnetic spectral response"],
        readiness="ready",
        required_for_mechanism_assessment=True,
    )
    mol = SERSValidationRoute(
        route_id="route:mol",
        route_kind="molecular_spectroscopy",
        statement="molecular",
        rationale="fixture",
        target_observables=["molecular excitation dependence"],
        readiness="deferred",
        required_for_mechanism_assessment=True,
    )
    exp = SERSValidationRoute(
        route_id="route:exp",
        route_kind="integrated_sers_experiment",
        statement="integrated",
        rationale="fixture",
        target_observables=["measured SERS performance under controlled comparison"],
        readiness="deferred",
        required_for_mechanism_assessment=True,
        required_for_outcome_assessment=True,
    )
    plan = SERSHypothesisValidationPlan(
        plan_id="plan:1",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_report_id="fdtd-app:1",
        hypothesis_id="hypothesis:1",
        fdtd_applicability="partial",
        routes=[em, mol, exp],
        mechanism_claim_present=True,
        mechanism_assessment="deferred_pending_routed_evidence",
        integrated_experiment_required=True,
    )
    return SERSHypothesisValidationPlanBundle(
        bundle_id="plans:1",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_bundle_id="fdtd-apps:1",
        plans=[plan],
    )


def _orchestration():
    requirement = SERSIntegratedExperimentRequirement(
        requirement_id="requirement:1",
        hypothesis_id="hypothesis:1",
        source_validation_plan_id="plan:1",
        source_validation_route_id="route:exp",
        route_target_observables=["measured SERS performance under controlled comparison"],
        source_prediction_ids=["prediction:1"],
        source_falsification_criterion_ids=["falsifier:1"],
        comparison_requirements=["define an explicit baseline/control"],
        control_principles=["hold non-focal acquisition conditions comparable"],
        unresolved_protocol_dimensions=[
            "fabrication_recipe",
            "sample_count_and_replication",
            "instrument_configuration",
            "statistical_acceptance_criterion",
        ],
    )
    states = [
        SERSRouteValidationState(
            route_id="route:em",
            route_kind="classical_em",
            required_for_mechanism_assessment=True,
            state="evidence_available_not_reviewable",
            blockers=["underresolved_geometry", "operator_concretized_model_unvalidated"],
            next_actions=["defer_classical_em_high_cost_escalation"],
            high_cost_escalation_deferred=True,
        ),
        SERSRouteValidationState(
            route_id="route:mol",
            route_kind="molecular_spectroscopy",
            required_for_mechanism_assessment=True,
            state="review_ready",
            evidence_ids=["evidence:mol1", "evidence:mol2"],
        ),
        SERSRouteValidationState(
            route_id="route:exp",
            route_kind="integrated_sers_experiment",
            required_for_mechanism_assessment=True,
            required_for_outcome_assessment=True,
            state="no_evidence",
            blockers=["route_evidence_missing"],
            experiment_requirement_id="requirement:1",
        ),
    ]
    state = SERSHypothesisValidationState(
        state_id="state:1",
        hypothesis_id="hypothesis:1",
        source_validation_plan_id="plan:1",
        route_states=states,
        mechanism_evidence_state="incomplete",
        outcome_evidence_state="incomplete",
        overall_research_state="mechanism_evidence_incomplete",
        recommended_next_route_kind="integrated_sers_experiment",
        recommended_next_route_id="route:exp",
        recommended_next_action="compare_integrated_experiment_value_against_deferred_em_escalation",
        rationale="fixture",
        experiment_requirement_ids=["requirement:1"],
    )
    return SERSValidationOrchestrationBundle(
        bundle_id="orchestration:1",
        source_portfolio_id="portfolio:1",
        source_validation_plan_bundle_id="plans:1",
        states=[state],
        state_count=1,
        experiment_requirements=[requirement],
        experiment_requirement_count=1,
    )


def _review(*, outcome="untested", experiment_state="direct_test_review_candidate", em_verdict="not_reviewable"):
    assessments = [
        SERSRouteScientificAssessment(
            assessment_id="assessment:em",
            hypothesis_id="hypothesis:1",
            route_id="route:em",
            route_kind="classical_em",
            required_for_mechanism_assessment=True,
            route_state="evidence_available_not_reviewable",
            verdict=em_verdict,
            evidence_strength="none" if em_verdict == "not_reviewable" else "single_independent_source",
            blockers=["underresolved_geometry"] if em_verdict == "not_reviewable" else [],
            interpretation="fixture",
        ),
        SERSRouteScientificAssessment(
            assessment_id="assessment:mol",
            hypothesis_id="hypothesis:1",
            route_id="route:mol",
            route_kind="molecular_spectroscopy",
            required_for_mechanism_assessment=True,
            route_state="review_ready",
            verdict="supportive",
            evidence_strength="multiple_independent_sources",
            evidence_ids=["evidence:mol1", "evidence:mol2"],
            decisive_evidence_ids=["evidence:mol1", "evidence:mol2"],
            independent_source_count=2,
            interpretation="fixture",
        ),
        SERSRouteScientificAssessment(
            assessment_id="assessment:exp",
            hypothesis_id="hypothesis:1",
            route_id="route:exp",
            route_kind="integrated_sers_experiment",
            required_for_mechanism_assessment=True,
            required_for_outcome_assessment=True,
            route_state="review_ready" if outcome != "untested" else "no_evidence",
            verdict=(outcome if outcome != "untested" else "not_reviewable"),
            evidence_strength="experimental_result" if outcome != "untested" else "none",
            evidence_ids=["evidence:exp"] if outcome != "untested" else [],
            decisive_evidence_ids=["evidence:exp"] if outcome != "untested" else [],
            experimental_result_count=1 if outcome != "untested" else 0,
            blockers=[] if outcome != "untested" else ["route_evidence_missing"],
            interpretation="fixture",
        ),
    ]
    review = SERSHypothesisScientificReview(
        review_id="review:1" if outcome == "untested" else f"review:{outcome}",
        hypothesis_id="hypothesis:1",
        source_validation_state_id="state:1",
        route_assessments=assessments,
        pre_experiment_mechanism_verdict="incomplete",
        outcome_verdict=outcome,
        experiment_review_state=experiment_state,
        research_decision=(
            "consider_integrated_experiment_review"
            if outcome == "untested"
            else (
                "outcome_inconsistent_reassess_mechanism"
                if outcome == "inconsistent"
                else "provisional_routed_support"
            )
        ),
        recommended_next_route_kind=("integrated_sers_experiment" if outcome == "untested" else None),
        recommended_next_route_id=("route:exp" if outcome == "untested" else None),
        recommended_next_action=(
            "compare_integrated_experiment_value_against_deferred_em_escalation"
            if outcome == "untested"
            else "review_integrated_experiment_outcome"
        ),
        rationale="fixture",
    )
    return SERSValidationReviewBundle(
        bundle_id="reviews:1" if outcome == "untested" else f"reviews:{outcome}",
        source_portfolio_id="portfolio:1",
        source_validation_plan_bundle_id="plans:1",
        source_orchestration_bundle_id="orchestration:1",
        reviews=[review],
        review_count=1,
    )


def _submission(plan_id, *, relation="inconsistent", quality="review_ready", controls="adequate", deviation="none"):
    row = SERSIntegratedExperimentResultSubmission(
        submission_id="submission:1",
        hypothesis_id="hypothesis:1",
        experiment_plan_id=plan_id,
        experiment_requirement_id="requirement:1",
        relation_to_claim=relation,
        result_quality=quality,
        assessment_basis="predeclared_criterion_match",
        measurement_summary="The controlled integrated SERS comparison was reviewed.",
        measured_observables=["excitation wavelength producing maximum SERS intensity"],
        comparison_summary="Compared the focal integrated outcome against the predeclared reference.",
        controls_status=controls,
        protocol_deviation_status=deviation,
        source_ids=["experiment_run:1"],
    )
    return SERSIntegratedExperimentResultSubmissionBundle(
        bundle_id="submissions:1",
        submissions=[row],
        submission_count=1,
    )


def test_direct_test_candidate_becomes_nonprocedural_experiment_review_plan():
    bundle = SERSIntegratedExperimentReviewPlanner().plan(
        _portfolio(), _plans(), _orchestration(), _review()
    )
    assert bundle.plan_count == 1
    row = bundle.plans[0]
    assert row.plan_status == "direct_test_review_candidate"
    assert row.information_priority == "preferred_over_deferred_high_cost_em_escalation"
    assert "fabrication_recipe" in row.comparison_design.unresolved_protocol_dimensions
    assert any(x.startswith("classical_em:") for x in row.preserved_uncertainties)
    assert row.protocol_generation_permitted is False
    assert row.experiment_execution_authority is False


def test_not_candidate_review_does_not_promote_experiment():
    bundle = SERSIntegratedExperimentReviewPlanner().plan(
        _portfolio(),
        _plans(),
        _orchestration(),
        _review(experiment_state="not_candidate"),
    )
    row = bundle.plans[0]
    assert row.plan_status == "not_candidate"
    assert row.information_priority == "not_prioritized"
    assert row.blockers == ["scientific_review_state:not_candidate"]


def test_result_compiler_merges_experimental_result_with_existing_molecular_evidence():
    plans = SERSIntegratedExperimentReviewPlanner().plan(
        _portfolio(), _plans(), _orchestration(), _review()
    )
    existing = SERSRouteEvidenceBundle(
        bundle_id="route_evidence:existing",
        records=[SERSRouteEvidenceRecord(
            evidence_id="evidence:mol1",
            hypothesis_id="hypothesis:1",
            route_id="route:mol",
            route_kind="molecular_spectroscopy",
            evidence_role="independent_validation",
            review_readiness="review_ready",
            relation_to_claim="supportive",
            source_ids=["DOI:1"],
            source_overlap_with_generation=False,
        )],
        record_count=1,
    )
    result, merged = SERSIntegratedExperimentResultCompiler().compile(
        plans,
        _submission(plans.plans[0].plan_id),
        existing_route_evidence=existing,
    )
    assert result.result_count == 1
    assert merged.record_count == 2
    exp = next(row for row in merged.records if row.route_kind == "integrated_sers_experiment")
    assert exp.evidence_role == "experimental_result"
    assert exp.review_readiness == "review_ready"
    assert exp.relation_to_claim == "inconsistent"


def test_review_ready_result_requires_adequate_or_partial_controls():
    plans = SERSIntegratedExperimentReviewPlanner().plan(
        _portfolio(), _plans(), _orchestration(), _review()
    )
    with pytest.raises(ValueError, match="adequate or partial controls"):
        _submission(plans.plans[0].plan_id, controls="inadequate")


def test_result_lineage_mismatch_fails_closed():
    plans = SERSIntegratedExperimentReviewPlanner().plan(
        _portfolio(), _plans(), _orchestration(), _review()
    )
    submissions = _submission(plans.plans[0].plan_id).model_copy(deep=True)
    submissions.submissions[0].experiment_requirement_id = "requirement:wrong"
    with pytest.raises(ValueError, match="requirement lineage mismatch"):
        SERSIntegratedExperimentResultCompiler().compile(plans, submissions)


def _result_bundle(relation="inconsistent"):
    plans = SERSIntegratedExperimentReviewPlanner().plan(
        _portfolio(), _plans(), _orchestration(), _review()
    )
    result, _ = SERSIntegratedExperimentResultCompiler().compile(
        plans, _submission(plans.plans[0].plan_id, relation=relation)
    )
    return result


def test_inconsistent_outcome_generates_abductive_failure_attribution_not_rejection():
    follow = SERSIntegratedExperimentFollowupAnalyzer().analyze(
        _review(outcome="inconsistent", experiment_state="experimental_result_available"),
        _result_bundle("inconsistent"),
    ).followups[0]
    categories = {row.category for row in follow.attribution_candidates}
    assert "classical_em_model_or_mechanism" in categories
    assert "fabrication_realization" in categories
    assert "interaction_nonadditivity_or_alternative_mechanism" in categories
    assert follow.failure_analysis_required is True
    assert follow.automatic_hypothesis_rejection_permitted is False
    assert all(row.causal_conclusion_permitted is False for row in follow.attribution_candidates)
    assert any(
        row.feedback_kind == "revisit_component_mechanism"
        for row in follow.feedback_candidates
    )


def test_supportive_outcome_preserves_mechanism_uncertainty_without_failure_attribution():
    follow = SERSIntegratedExperimentFollowupAnalyzer().analyze(
        _review(outcome="supportive", experiment_state="experimental_result_available"),
        _result_bundle("supportive"),
    ).followups[0]
    assert follow.failure_analysis_required is False
    assert follow.attribution_candidates == []
    assert follow.feedback_candidates[0].feedback_kind == (
        "preserve_outcome_support_with_mechanism_uncertainty"
    )
    assert follow.whole_hypothesis_confirmation_permitted is False
