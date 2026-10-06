from __future__ import annotations

from pipeline_core.discovery.candidate_contracts import (
    CandidateDecisionCard,
    CandidateDecisionPortfolio,
)
from pipeline_core.discovery.candidate_decision import CandidateDecisionEngine
from pipeline_core.discovery.feasibility.feasibility_contracts import (
    FeasibilityHypothesis,
    FeasibilityIntake,
    FeasibilityPrediction,
    FeasibilityFalsifier,
)

from domains.sers.feasibility import SERSFeasibilityAdapter
from domains.sers.feasibility_lifecycle_projection import (
    SERSCanonicalLifecycleProjector,
)
from domains.sers.integrated_experiment_cycle_contracts import (
    SERSExperimentComparisonDesign,
    SERSIntegratedExperimentReviewPlan,
    SERSIntegratedExperimentReviewPlanBundle,
)
from domains.sers.validation_review_contracts import (
    SERSClaimScopedResearchFeedback,
    SERSHypothesisScientificReview,
    SERSRouteScientificAssessment,
    SERSValidationReviewBundle,
)


def _intake() -> FeasibilityIntake:
    return FeasibilityIntake(
        intake_id="intake:sers",
        intake_sha256="isha",
        source_context_id="ctx",
        source_context_sha256="csha",
        source_portfolio_id="portfolio:sers",
        source_portfolio_sha256="psha",
        source_semantic_review_id="review:semantic",
        task_id="task",
        question="q",
        corpus_id="sers-corpus",
        hypotheses=[
            FeasibilityHypothesis(
                hypothesis_id="hypothesis:1",
                title="MB excitation optimum on Au@Ag core-shell nanorod",
                statement=(
                    "For methylene blue on an Au@Ag core-shell nanorod, the "
                    "excitation wavelength maximizing SERS intensity shifts from "
                    "the substrate-only 785 nm resonance because analyte-specific "
                    "wavelength dependence contributes to the response."
                ),
                hypothesis_type="context_dependency",
                inferential_bridge=(
                    "Au@Ag plasmonic substrate response and methylene-blue "
                    "wavelength dependence jointly shape the measured SERS optimum."
                ),
                source_paper_ids=["paper:generation:1", "paper:generation:2"],
                candidate_dependency="none",
                predictions=[
                    FeasibilityPrediction(
                        observation_id="prediction:1",
                        observable="excitation wavelength producing maximum SERS intensity",
                        expected_direction="shift",
                        rationale="molecular and substrate response interact",
                    )
                ],
                falsifiers=[
                    FeasibilityFalsifier(
                        criterion_id="falsifier:1",
                        observable="excitation wavelength producing maximum SERS intensity",
                        falsifying_outcome="The controlled optimum remains at 785 nm.",
                    )
                ],
                semantic_gate_status="eligible",
            )
        ],
    )


def _run_adapter():
    intake = _intake()
    adapter = SERSFeasibilityAdapter()
    scopes = adapter.compile_scopes(intake)
    specs = adapter.compile_specifications(intake, scopes)
    physics = adapter.run_physics(intake, scopes, specs)
    experimental = adapter.run_experimental(intake, physics, scopes, specs)
    decisions = CandidateDecisionEngine().decide(
        intake, scopes, specs, physics, experimental
    )
    return intake, scopes[0], specs[0], physics[0], experimental[0], decisions


def test_sers_adapter_compiles_multi_validator_scope_without_her_semantics():
    _, scope, spec, physics, experimental, decisions = _run_adapter()

    assert scope.scientific_domain == "sers_au_ag"
    assert scope.system_class == "au_ag_core_shell_nanorod_sers"
    assert scope.process == "surface_enhanced_raman_scattering"
    assert scope.catalyst_class == "unknown"
    assert scope.reaction == "unknown"
    assert scope.requires_candidate_concretization is True

    assert spec.validation_strategy == "multi_validator"
    assert set(spec.required_scientific_checks) >= {
        "classical_em",
        "molecular_spectroscopy",
        "integrated_sers_experiment",
    }
    assert spec.requires_candidate_concretization is True

    assert physics.disposition == "requires_computation"
    assert physics.blocking_checks == []
    assert all(row.status != "fail" for row in physics.checks)
    assert "classical_em" in physics.unresolved_checks

    assert experimental.disposition == "conditionally_plausible"
    assert experimental.required_electrochemical_tests == []
    assert "electrochemical_performance_testing" in experimental.not_applicable_capabilities
    assert experimental.performance_testability == "conditional"
    assert any("controlled SERS" in row for row in experimental.required_performance_tests)

    card = decisions.cards[0]
    assert card.final_disposition == "requires_validation_design"
    assert card.physics_disposition != "physically_implausible"
    assert card.final_disposition != "rejected_physical"


def _canonical_decisions() -> CandidateDecisionPortfolio:
    _, _, _, _, _, decisions = _run_adapter()
    return decisions


def _review_bundle() -> SERSValidationReviewBundle:
    review = SERSHypothesisScientificReview(
        review_id="review:sers:1",
        hypothesis_id="hypothesis:1",
        source_validation_state_id="state:1",
        route_assessments=[
            SERSRouteScientificAssessment(
                assessment_id="route_assessment:em",
                hypothesis_id="hypothesis:1",
                route_id="route:em",
                route_kind="classical_em",
                required_for_mechanism_assessment=True,
                route_state="evidence_available_not_reviewable",
                verdict="not_reviewable",
                evidence_strength="none",
                blockers=["underresolved_geometry", "operator_concretized_model_unvalidated"],
                interpretation="Preserve EM uncertainty.",
            ),
            SERSRouteScientificAssessment(
                assessment_id="route_assessment:molecular",
                hypothesis_id="hypothesis:1",
                route_id="route:molecular",
                route_kind="molecular_spectroscopy",
                required_for_mechanism_assessment=True,
                route_state="review_ready",
                verdict="supportive",
                evidence_strength="multiple_independent_sources",
                evidence_ids=["evidence:m1", "evidence:m2"],
                decisive_evidence_ids=["evidence:m1", "evidence:m2"],
                independent_source_count=2,
                interpretation="Independent molecular evidence is supportive.",
            ),
            SERSRouteScientificAssessment(
                assessment_id="route_assessment:experiment",
                hypothesis_id="hypothesis:1",
                route_id="route:experiment",
                route_kind="integrated_sers_experiment",
                required_for_outcome_assessment=True,
                route_state="no_evidence",
                verdict="not_reviewable",
                evidence_strength="none",
                blockers=["route_evidence_missing"],
                interpretation="Integrated outcome remains untested.",
            ),
        ],
        pre_experiment_mechanism_verdict="incomplete",
        outcome_verdict="untested",
        experiment_review_state="direct_test_review_candidate",
        research_decision="consider_integrated_experiment_review",
        recommended_next_route_kind="integrated_sers_experiment",
        recommended_next_route_id="route:experiment",
        recommended_next_action=(
            "compare_integrated_experiment_value_against_deferred_em_escalation"
        ),
        rationale="Integrated experiment is the next information-gain candidate.",
        feedback=[
            SERSClaimScopedResearchFeedback(
                feedback_id="feedback:em",
                hypothesis_id="hypothesis:1",
                route_id="route:em",
                route_kind="classical_em",
                feedback_kind="preserve_uncertainty",
                action="preserve_classical_em_uncertainty_without_high_cost_escalation",
                rationale="EM evidence is non-reviewable and high-cost escalation is deferred.",
            )
        ],
    )
    return SERSValidationReviewBundle(
        bundle_id="review_bundle:1",
        source_portfolio_id="portfolio:sers",
        source_validation_plan_bundle_id="validation_plans:1",
        source_orchestration_bundle_id="orchestration:1",
        reviews=[review],
        review_count=1,
    )


def _experiment_plan_bundle() -> SERSIntegratedExperimentReviewPlanBundle:
    plan = SERSIntegratedExperimentReviewPlan(
        plan_id="experiment_plan:1",
        hypothesis_id="hypothesis:1",
        source_scientific_review_id="review:sers:1",
        source_validation_state_id="state:1",
        source_experiment_requirement_id="experiment_requirement:1",
        source_validation_route_id="route:experiment",
        plan_status="direct_test_review_candidate",
        information_priority="preferred_over_deferred_high_cost_em_escalation",
        comparison_design=SERSExperimentComparisonDesign(
            focal_hypothesis_statement="Integrated SERS optimum shifts.",
            target_observables=["measured SERS performance under controlled comparison"],
            prediction_statements=["maximum SERS wavelength shifts"],
            falsification_statements=["maximum remains at 785 nm"],
            comparison_requirements=["explicit substrate-only baseline"],
            control_principles=["hold non-focal conditions comparable"],
            unresolved_protocol_dimensions=["sample_count_and_replication"],
        ),
        result_interpretation_rules=["predeclared criterion only"],
        preserved_uncertainties=["classical_em:underresolved_geometry"],
        rationale="Experiment is preferred over deferred high-cost EM escalation.",
    )
    return SERSIntegratedExperimentReviewPlanBundle(
        bundle_id="experiment_plan_bundle:1",
        source_portfolio_id="portfolio:sers",
        source_validation_plan_bundle_id="validation_plans:1",
        source_orchestration_bundle_id="orchestration:1",
        source_scientific_review_bundle_id="review_bundle:1",
        plans=[plan],
        plan_count=1,
    )


def test_projection_joins_canonical_feasibility_and_sers_lifecycle_without_authority_change():
    result = SERSCanonicalLifecycleProjector().project(
        _canonical_decisions(),
        _review_bundle(),
        experiment_plans=_experiment_plan_bundle(),
    )
    row = result.projections[0]

    assert row.canonical_final_disposition == "requires_validation_design"
    assert row.sers_research_decision == "consider_integrated_experiment_review"
    assert row.experiment_plan_status == "direct_test_review_candidate"
    assert row.experiment_information_priority == (
        "preferred_over_deferred_high_cost_em_escalation"
    )
    assert "classical_em" in row.unresolved_route_kinds
    assert "classical_em:underresolved_geometry" in row.preserved_uncertainties
    assert row.claim_scoped_feedback_actions == [
        "preserve_classical_em_uncertainty_without_high_cost_escalation"
    ]
    assert row.authority_conflict_detected is False
    assert row.canonical_candidate_rejection_override_permitted is False
    assert row.canonical_candidate_promotion_override_permitted is False
    assert row.canonical_feedback_permitted is False
    assert result.canonical_authority_changed is False


def test_projection_fails_closed_on_hypothesis_coverage_mismatch():
    decisions = _canonical_decisions().model_copy(deep=True)
    decisions.cards[0].hypothesis_id = "hypothesis:other"
    try:
        SERSCanonicalLifecycleProjector().project(decisions, _review_bundle())
    except ValueError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("expected coverage mismatch to fail closed")
