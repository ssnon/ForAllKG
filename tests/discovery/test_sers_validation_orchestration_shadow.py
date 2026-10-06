from __future__ import annotations

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)

from domains.sers.classical_em_readiness_contracts import (
    SERSClassicalEMReadinessAssessment,
    SERSClassicalEMReadinessBundle,
    SERSEMModelFormSnapshot,
)
from domains.sers.validation_orchestration import SERSValidationOrchestrator
from domains.sers.validation_orchestration_contracts import (
    SERSRouteEvidenceBundle,
    SERSRouteEvidenceRecord,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlan,
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


def _portfolio() -> HypothesisPortfolio:
    card = HypothesisCard(
        hypothesis_id="hypothesis:1",
        domain_profile_id="sers_au_ag",
        source_context_id="context:1",
        source_context_sha256="a" * 64,
        source_report_id="report:1",
        source_report_sha256="b" * 64,
        title="MB wavelength-dependent Au@Ag SERS optimum",
        hypothesis_statement=(
            "Methylene-blue wavelength dependence and Au@Ag plasmonic response "
            "jointly shift the integrated SERS optimum."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        inferential_bridge=(
            "The substrate electromagnetic response and analyte wavelength response "
            "both contribute to measured SERS intensity."
        ),
        predicted_observations=[PredictedObservation(
            observation_id="prediction:1",
            observable="excitation wavelength producing maximum SERS intensity",
            expected_direction="shift",
            rationale="The integrated optimum need not equal the substrate-only baseline.",
        )],
        falsification_criteria=[FalsificationCriterion(
            criterion_id="falsifier:1",
            observable="excitation wavelength producing maximum SERS intensity",
            falsifying_outcome="No resolvable shift from the predeclared baseline/control.",
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


def _composite_plans() -> SERSHypothesisValidationPlanBundle:
    routes = [
        SERSValidationRoute(
            route_id="route:em",
            route_kind="classical_em",
            statement="Evaluate substrate-side electromagnetic response.",
            rationale="EM mechanism branch.",
            source_fields=["hypothesis_statement"],
            target_observables=[
                "substrate electromagnetic spectral response",
                "electromagnetic near-field enhancement",
            ],
            validator_candidates=["fdtd"],
            readiness="ready",
            source_fdtd_subclaim_ids=["subclaim:em"],
            required_for_mechanism_assessment=True,
        ),
        SERSValidationRoute(
            route_id="route:molecular",
            route_kind="molecular_spectroscopy",
            statement="Evaluate methylene-blue wavelength-dependent response.",
            rationale="Molecular contribution branch.",
            source_fields=["hypothesis_statement", "inferential_bridge"],
            target_observables=["molecular wavelength-dependent Raman response"],
            validator_candidates=["literature", "spectroscopy"],
            readiness="deferred",
            required_for_mechanism_assessment=True,
        ),
        SERSValidationRoute(
            route_id="route:experiment",
            route_kind="integrated_sers_experiment",
            statement="Measure the integrated SERS excitation optimum.",
            rationale="Whole measured outcome branch.",
            source_fields=["predicted_observations[0].observable"],
            target_observables=[
                "excitation wavelength producing maximum SERS intensity"
            ],
            validator_candidates=["experiment"],
            readiness="deferred",
            required_for_outcome_assessment=True,
            required_for_mechanism_assessment=True,
        ),
    ]
    plan = SERSHypothesisValidationPlan(
        plan_id="plan:1",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_report_id="fdtd-app:1",
        hypothesis_id="hypothesis:1",
        fdtd_applicability="partial",
        routes=routes,
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


def _em_readiness() -> SERSClassicalEMReadinessBundle:
    model = SERSEMModelFormSnapshot(
        validation_design_id="design:1",
        source_label="operator_screen",
        operator_note="Screening geometry only.",
        geometry_candidate_id="geometry:1",
        geometry_candidate_label="outer90x30_shell7p5",
        geometry_candidate_rationale="operator assumption",
        nanorod_length_nm=75.0,
        nanorod_diameter_nm=15.0,
        shell_thickness_nm=7.5,
        surrounding_medium="water",
        source_baseline_wavelength_nm=785.0,
        source_reported_resonance_wavelengths_nm=[512.0, 772.0],
    )
    row = SERSClassicalEMReadinessAssessment(
        assessment_id="readiness:1",
        hypothesis_id="hypothesis:1",
        source_evidence_id="em-evidence:1",
        source_validation_plan_id="plan:1",
        source_validation_route_id="route:em",
        source_validation_design_id="design:1",
        source_simulation_spec_id="spec:1",
        route_target_observables=[
            "substrate electromagnetic spectral response",
            "electromagnetic near-field enhancement",
        ],
        observed_route_target_observables=[
            "substrate electromagnetic spectral response"
        ],
        unobserved_route_target_observables=[
            "electromagnetic near-field enhancement"
        ],
        evidence_scope_status="route_targets_partially_observed",
        numerical_evidence_state="underresolved_observation",
        numerical_blockers=["underresolved_geometry", "no_convergence_threshold"],
        numerical_readiness="repair_required",
        model_form=model,
        readiness_blockers=[
            "numerical_evidence_not_ready",
            "operator_concretized_model_unvalidated",
            "positive_control_not_assessed",
            "route_target_observable_unobserved",
        ],
        immediate_numerical_repairs=["establish_convergence_acceptance_policy"],
        deferred_high_cost_actions=["defer_high_cost_spatial_fidelity_escalation"],
        next_research_actions=[
            "establish_convergence_acceptance_policy",
            "establish_positive_control_model_validation",
            "review_operator_concretized_model",
            "expand_classical_em_observables",
            "defer_high_cost_spatial_fidelity_escalation",
        ],
        overall_readiness="numerical_repair_required",
    )
    return SERSClassicalEMReadinessBundle(
        bundle_id="readiness-bundle:1",
        source_evidence_bundle_id="em-evidence-bundle:1",
        source_validation_plan_bundle_id="plans:1",
        source_validation_design_bundle_id="designs:1",
        assessments=[row],
        assessment_count=1,
    )


def _evidence(*records: SERSRouteEvidenceRecord) -> SERSRouteEvidenceBundle:
    return SERSRouteEvidenceBundle(
        bundle_id="route-evidence:1",
        records=list(records),
        record_count=len(records),
    )


def test_composite_state_prefers_empty_molecular_route_over_deferred_em_escalation():
    result = SERSValidationOrchestrator().build(
        _portfolio(),
        _composite_plans(),
        classical_em_readiness=_em_readiness(),
    )
    state = result.states[0]
    by_kind = {row.route_kind: row for row in state.route_states}

    assert by_kind["classical_em"].state == "evidence_available_not_reviewable"
    assert by_kind["classical_em"].high_cost_escalation_deferred is True
    assert by_kind["molecular_spectroscopy"].state == "no_evidence"
    assert by_kind["integrated_sers_experiment"].state == "no_evidence"

    assert state.mechanism_evidence_state == "incomplete"
    assert state.outcome_evidence_state == "incomplete"
    assert state.overall_research_state == "mechanism_evidence_incomplete"
    assert state.recommended_next_route_kind == "molecular_spectroscopy"
    assert state.recommended_next_action == (
        "collect_independent_molecular_spectroscopy_evidence"
    )

    assert result.experiment_requirement_count == 1
    requirement = result.experiment_requirements[0]
    assert requirement.source_prediction_ids == ["prediction:1"]
    assert requirement.source_falsification_criterion_ids == ["falsifier:1"]
    assert requirement.protocol_generation_permitted is False
    assert requirement.experiment_execution_authority is False
    assert result.scientific_verdict_authority_created is False


def test_independent_review_ready_molecular_evidence_moves_focus_back_to_em_readiness():
    molecular = SERSRouteEvidenceRecord(
        evidence_id="molecular:1",
        hypothesis_id="hypothesis:1",
        route_id="route:molecular",
        route_kind="molecular_spectroscopy",
        evidence_role="independent_validation",
        review_readiness="review_ready",
        relation_to_claim="supportive",
        source_paper_ids=["paper:independent"],
        source_overlap_with_generation=False,
    )
    result = SERSValidationOrchestrator().build(
        _portfolio(),
        _composite_plans(),
        classical_em_readiness=_em_readiness(),
        route_evidence=_evidence(molecular),
    )
    state = result.states[0]
    by_kind = {row.route_kind: row for row in state.route_states}

    assert by_kind["molecular_spectroscopy"].state == "review_ready"
    assert state.recommended_next_route_kind == "classical_em"
    assert state.recommended_next_action == "establish_convergence_acceptance_policy"
    assert state.scientific_verdict_permitted is False


def test_generation_evidence_cannot_self_validate_route_even_when_marked_review_ready():
    molecular = SERSRouteEvidenceRecord(
        evidence_id="molecular:generation",
        hypothesis_id="hypothesis:1",
        route_id="route:molecular",
        route_kind="molecular_spectroscopy",
        evidence_role="generation_evidence",
        review_readiness="review_ready",
        relation_to_claim="supportive",
        source_paper_ids=["paper:generation"],
        source_overlap_with_generation=True,
    )
    result = SERSValidationOrchestrator().build(
        _portfolio(),
        _composite_plans(),
        classical_em_readiness=_em_readiness(),
        route_evidence=_evidence(molecular),
    )
    route = next(
        row
        for row in result.states[0].route_states
        if row.route_kind == "molecular_spectroscopy"
    )
    assert route.state == "evidence_available_not_reviewable"
    assert "generation_evidence_not_independent_validation" in route.blockers
    assert "source_overlap_with_generation" in route.blockers


def test_paper_id_overlap_fails_closed_even_if_caller_marks_overlap_false():
    molecular = SERSRouteEvidenceRecord(
        evidence_id="molecular:bad-overlap",
        hypothesis_id="hypothesis:1",
        route_id="route:molecular",
        route_kind="molecular_spectroscopy",
        evidence_role="independent_validation",
        review_readiness="review_ready",
        source_paper_ids=["paper:generation"],
        source_overlap_with_generation=False,
    )
    result = SERSValidationOrchestrator().build(
        _portfolio(),
        _composite_plans(),
        classical_em_readiness=_em_readiness(),
        route_evidence=_evidence(molecular),
    )
    route = next(
        row
        for row in result.states[0].route_states
        if row.route_kind == "molecular_spectroscopy"
    )
    assert route.state == "evidence_available_not_reviewable"
    assert "source_overlap_with_generation" in route.blockers


def test_non_em_routes_can_be_review_ready_without_creating_hypothesis_verdict_authority():
    portfolio = _portfolio()
    routes = [
        SERSValidationRoute(
            route_id="route:surface",
            route_kind="surface_chemistry",
            statement="Assess the surface-chemistry premise.",
            rationale="fixture",
            source_fields=["hypothesis_statement"],
            target_observables=["surface association"],
            validator_candidates=["literature", "experiment"],
            readiness="deferred",
            required_for_mechanism_assessment=True,
        ),
        SERSValidationRoute(
            route_id="route:experiment-only",
            route_kind="integrated_sers_experiment",
            statement="Measure integrated SERS response.",
            rationale="fixture",
            source_fields=["predicted_observations[0].observable"],
            target_observables=[
                "excitation wavelength producing maximum SERS intensity"
            ],
            validator_candidates=["experiment"],
            readiness="deferred",
            required_for_outcome_assessment=True,
        ),
    ]
    plan = SERSHypothesisValidationPlan(
        plan_id="plan:non-em",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_report_id="fdtd-app:none",
        hypothesis_id="hypothesis:1",
        fdtd_applicability="not_applicable",
        routes=routes,
        mechanism_claim_present=True,
        mechanism_assessment="deferred_pending_routed_evidence",
        integrated_experiment_required=True,
    )
    plans = SERSHypothesisValidationPlanBundle(
        bundle_id="plans:non-em",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_bundle_id="fdtd-apps:none",
        plans=[plan],
    )
    surface = SERSRouteEvidenceRecord(
        evidence_id="surface:1",
        hypothesis_id="hypothesis:1",
        route_id="route:surface",
        route_kind="surface_chemistry",
        evidence_role="independent_validation",
        review_readiness="review_ready",
        source_paper_ids=["paper:surface-independent"],
        source_overlap_with_generation=False,
    )
    experiment = SERSRouteEvidenceRecord(
        evidence_id="experiment:1",
        hypothesis_id="hypothesis:1",
        route_id="route:experiment-only",
        route_kind="integrated_sers_experiment",
        evidence_role="experimental_result",
        review_readiness="review_ready",
        source_ids=["experiment-run:1"],
        source_overlap_with_generation=False,
    )

    result = SERSValidationOrchestrator().build(
        portfolio,
        plans,
        route_evidence=_evidence(surface, experiment),
    )
    state = result.states[0]
    assert state.mechanism_evidence_state == "review_ready"
    assert state.outcome_evidence_state == "review_ready"
    assert state.overall_research_state == "routed_evidence_review_ready"
    assert state.recommended_next_route_kind is None
    assert state.recommended_next_action == "perform_multi_route_scientific_review"
    assert state.scientific_verdict_permitted is False
    assert result.hypothesis_rejection_authority is False
    assert result.experiment_promotion_authority is False


def test_unknown_route_evidence_lineage_fails_closed():
    bad = SERSRouteEvidenceRecord(
        evidence_id="molecular:unknown",
        hypothesis_id="hypothesis:1",
        route_id="route:missing",
        route_kind="molecular_spectroscopy",
        evidence_role="independent_validation",
        review_readiness="review_ready",
        source_paper_ids=["paper:independent"],
        source_overlap_with_generation=False,
    )
    with pytest.raises(ValueError, match="unknown validation route"):
        SERSValidationOrchestrator().build(
            _portfolio(),
            _composite_plans(),
            classical_em_readiness=_em_readiness(),
            route_evidence=_evidence(bad),
        )
