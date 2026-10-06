from __future__ import annotations

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from domains.sers.validation_evidence_intake import SERSRouteEvidenceIntakeCompiler
from domains.sers.validation_evidence_intake_contracts import (
    SERSRouteEvidenceSubmission,
    SERSRouteEvidenceSubmissionBundle,
)
from domains.sers.validation_orchestration_contracts import (
    SERSHypothesisValidationState,
    SERSRouteEvidenceBundle,
    SERSRouteEvidenceRecord,
    SERSRouteValidationState,
    SERSValidationOrchestrationBundle,
)
from domains.sers.validation_review import SERSMultiValidatorScientificReviewer
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
        title="candidate",
        hypothesis_statement=(
            "Molecular wavelength dependence and plasmonic response jointly shift "
            "the measured SERS optimum."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        inferential_bridge="Molecular and electromagnetic wavelength responses interact.",
        predicted_observations=[PredictedObservation(
            observation_id="prediction:1",
            observable="excitation wavelength producing maximum SERS intensity",
            expected_direction="shift",
            rationale="test",
        )],
        falsification_criteria=[FalsificationCriterion(
            criterion_id="falsifier:1",
            observable="excitation wavelength producing maximum SERS intensity",
            falsifying_outcome="No shift is observed under the controlled comparison.",
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


def _plans(*, include_em=True):
    routes = []
    if include_em:
        routes.append(SERSValidationRoute(
            route_id="route:em",
            route_kind="classical_em",
            statement="Evaluate substrate electromagnetic response.",
            rationale="fixture",
            source_fields=["hypothesis_statement"],
            target_observables=["substrate electromagnetic spectral response"],
            validator_candidates=["fdtd"],
            readiness="ready",
            source_fdtd_subclaim_ids=["subclaim:em"],
            required_for_mechanism_assessment=True,
        ))
    routes.append(SERSValidationRoute(
        route_id="route:mol",
        route_kind="molecular_spectroscopy",
        statement="Evaluate molecular wavelength dependence.",
        rationale="fixture",
        source_fields=["hypothesis_statement"],
        target_observables=["molecular Raman wavelength dependence"],
        validator_candidates=["literature"],
        readiness="ready",
        required_for_mechanism_assessment=True,
    ))
    routes.append(SERSValidationRoute(
        route_id="route:exp",
        route_kind="integrated_sers_experiment",
        statement="Measure integrated SERS optimum.",
        rationale="fixture",
        source_fields=["predicted_observations[0].observable"],
        target_observables=["measured SERS performance under controlled comparison"],
        validator_candidates=["experiment"],
        readiness="deferred",
        required_for_mechanism_assessment=True,
        required_for_outcome_assessment=True,
    ))
    plan = SERSHypothesisValidationPlan(
        plan_id="plan:1",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_report_id="app:1",
        hypothesis_id="hypothesis:1",
        fdtd_applicability="partial" if include_em else "not_applicable",
        routes=routes,
        mechanism_claim_present=True,
        mechanism_assessment="deferred_pending_routed_evidence",
        integrated_experiment_required=True,
    )
    return SERSHypothesisValidationPlanBundle(
        bundle_id="plans:1",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_bundle_id="apps:1",
        plans=[plan],
    )


def _orchestration(*, include_em=True, molecular_state="no_evidence", experimental_state="no_evidence"):
    route_states = []
    if include_em:
        route_states.append(SERSRouteValidationState(
            route_id="route:em",
            route_kind="classical_em",
            required_for_mechanism_assessment=True,
            state="evidence_available_not_reviewable",
            evidence_ids=["em:evidence"],
            blockers=["underresolved_geometry"],
            next_actions=["defer_classical_em_high_cost_escalation"],
            high_cost_escalation_deferred=True,
        ))
    route_states.append(SERSRouteValidationState(
        route_id="route:mol",
        route_kind="molecular_spectroscopy",
        required_for_mechanism_assessment=True,
        state=molecular_state,
        evidence_ids=[] if molecular_state == "no_evidence" else ["mol:evidence"],
        blockers=["route_evidence_missing"] if molecular_state == "no_evidence" else [],
        next_actions=[
            "collect_independent_molecular_spectroscopy_evidence"
            if molecular_state == "no_evidence"
            else "perform_component_scientific_review"
        ],
    ))
    route_states.append(SERSRouteValidationState(
        route_id="route:exp",
        route_kind="integrated_sers_experiment",
        required_for_mechanism_assessment=True,
        required_for_outcome_assessment=True,
        state=experimental_state,
        evidence_ids=[] if experimental_state == "no_evidence" else ["exp:evidence"],
        blockers=["route_evidence_missing"] if experimental_state == "no_evidence" else [],
        next_actions=[
            "review_integrated_sers_experiment_requirement_for_execution"
            if experimental_state == "no_evidence"
            else "perform_component_scientific_review"
        ],
        experiment_requirement_id="requirement:1",
    ))
    state = SERSHypothesisValidationState(
        state_id="state:1",
        hypothesis_id="hypothesis:1",
        source_validation_plan_id="plan:1",
        route_states=route_states,
        mechanism_evidence_state="incomplete",
        outcome_evidence_state="incomplete",
        overall_research_state="mechanism_evidence_incomplete",
        recommended_next_route_kind="molecular_spectroscopy",
        recommended_next_route_id="route:mol",
        recommended_next_action="collect_independent_molecular_spectroscopy_evidence",
        rationale="fixture",
        experiment_requirement_ids=["requirement:1"],
    )
    return SERSValidationOrchestrationBundle(
        bundle_id="orchestration:1",
        source_portfolio_id="portfolio:1",
        source_validation_plan_bundle_id="plans:1",
        source_classical_em_readiness_bundle_id="readiness:1" if include_em else None,
        source_route_evidence_bundle_id="evidence:1" if molecular_state != "no_evidence" or experimental_state != "no_evidence" else None,
        states=[state],
        state_count=1,
        experiment_requirements=[],
        experiment_requirement_count=0,
    )


def _record(
    *,
    evidence_id,
    route_id,
    route_kind,
    relation,
    role="independent_validation",
    paper="paper:independent",
):
    return SERSRouteEvidenceRecord(
        evidence_id=evidence_id,
        hypothesis_id="hypothesis:1",
        route_id=route_id,
        route_kind=route_kind,
        evidence_role=role,
        review_readiness="review_ready",
        relation_to_claim=relation,
        source_paper_ids=[paper] if paper else [],
        source_overlap_with_generation=False,
    )


def _bundle(*records):
    return SERSRouteEvidenceBundle(
        bundle_id="evidence:1",
        records=list(records),
        record_count=len(records),
    )


def test_evidence_intake_resolves_route_and_rejects_generation_overlap():
    portfolio = _portfolio()
    plans = _plans()
    submissions = SERSRouteEvidenceSubmissionBundle(
        bundle_id="submissions:1",
        submissions=[SERSRouteEvidenceSubmission(
            submission_id="submission:1",
            hypothesis_id="hypothesis:1",
            route_kind="molecular_spectroscopy",
            evidence_role="independent_validation",
            review_readiness="review_ready",
            relation_to_claim="supportive",
            source_paper_ids=["paper:independent"],
            source_overlap_with_generation=False,
        )],
        submission_count=1,
    )
    compiled = SERSRouteEvidenceIntakeCompiler().compile(portfolio, plans, submissions)
    assert compiled.records[0].route_id == "route:mol"

    bad = submissions.model_copy(deep=True)
    bad.submissions[0].source_paper_ids = ["paper:generation"]
    with pytest.raises(ValueError, match="overlap"):
        SERSRouteEvidenceIntakeCompiler().compile(portfolio, plans, bad)


def test_current_state_still_collects_missing_molecular_evidence():
    result = SERSMultiValidatorScientificReviewer().review(
        _portfolio(), _plans(), _orchestration()
    )
    review = result.reviews[0]
    assert review.research_decision == "collect_missing_route_evidence"
    assert review.recommended_next_route_kind == "molecular_spectroscopy"
    assert review.experiment_review_state == "not_candidate"
    assert review.pre_experiment_mechanism_verdict == "incomplete"


def test_supportive_molecular_evidence_escapes_deferred_em_tunnel_to_experiment_review():
    evidence = _bundle(_record(
        evidence_id="mol:evidence",
        route_id="route:mol",
        route_kind="molecular_spectroscopy",
        relation="supportive",
    ))
    result = SERSMultiValidatorScientificReviewer().review(
        _portfolio(),
        _plans(),
        _orchestration(molecular_state="review_ready"),
        route_evidence=evidence,
    )
    review = result.reviews[0]
    by_kind = {row.route_kind: row for row in review.route_assessments}
    assert by_kind["molecular_spectroscopy"].verdict == "supportive"
    assert by_kind["classical_em"].verdict == "not_reviewable"
    assert review.research_decision == "consider_integrated_experiment_review"
    assert review.experiment_review_state == "direct_test_review_candidate"
    assert review.recommended_next_route_kind == "integrated_sers_experiment"


def test_inconsistent_molecular_evidence_revises_only_that_component():
    evidence = _bundle(_record(
        evidence_id="mol:evidence",
        route_id="route:mol",
        route_kind="molecular_spectroscopy",
        relation="inconsistent",
    ))
    result = SERSMultiValidatorScientificReviewer().review(
        _portfolio(),
        _plans(),
        _orchestration(molecular_state="review_ready"),
        route_evidence=evidence,
    )
    review = result.reviews[0]
    assert review.research_decision == "revise_component_claim"
    assert review.feedback[0].route_kind == "molecular_spectroscopy"
    assert review.feedback[0].applies_to_whole_hypothesis is False
    assert review.feedback[0].automatic_hypothesis_rejection_permitted is False


def test_conflicting_independent_molecular_evidence_is_mixed():
    evidence = _bundle(
        _record(
            evidence_id="mol:support",
            route_id="route:mol",
            route_kind="molecular_spectroscopy",
            relation="supportive",
            paper="paper:a",
        ),
        _record(
            evidence_id="mol:against",
            route_id="route:mol",
            route_kind="molecular_spectroscopy",
            relation="inconsistent",
            paper="paper:b",
        ),
    )
    result = SERSMultiValidatorScientificReviewer().review(
        _portfolio(),
        _plans(),
        _orchestration(molecular_state="review_ready"),
        route_evidence=evidence,
    )
    review = result.reviews[0]
    mol = next(row for row in review.route_assessments if row.route_kind == "molecular_spectroscopy")
    assert mol.verdict == "mixed"
    assert mol.independent_source_count == 2
    assert review.research_decision == "resolve_conflicting_component_evidence"


def test_without_em_supportive_molecular_route_becomes_experiment_review_candidate():
    evidence = _bundle(_record(
        evidence_id="mol:evidence",
        route_id="route:mol",
        route_kind="molecular_spectroscopy",
        relation="supportive",
    ))
    result = SERSMultiValidatorScientificReviewer().review(
        _portfolio(),
        _plans(include_em=False),
        _orchestration(include_em=False, molecular_state="review_ready"),
        route_evidence=evidence,
    )
    review = result.reviews[0]
    assert review.pre_experiment_mechanism_verdict == "supportive"
    assert review.research_decision == "candidate_for_integrated_experiment_review"
    assert review.experiment_review_state == "review_candidate"


def test_inconsistent_integrated_experiment_triggers_failure_analysis_not_auto_rejection():
    evidence = _bundle(
        _record(
            evidence_id="mol:evidence",
            route_id="route:mol",
            route_kind="molecular_spectroscopy",
            relation="supportive",
        ),
        _record(
            evidence_id="exp:evidence",
            route_id="route:exp",
            route_kind="integrated_sers_experiment",
            relation="inconsistent",
            role="experimental_result",
            paper=None,
        ),
    )
    result = SERSMultiValidatorScientificReviewer().review(
        _portfolio(),
        _plans(include_em=False),
        _orchestration(
            include_em=False,
            molecular_state="review_ready",
            experimental_state="review_ready",
        ),
        route_evidence=evidence,
    )
    review = result.reviews[0]
    assert review.outcome_verdict == "inconsistent"
    assert review.research_decision == "outcome_inconsistent_reassess_mechanism"
    assert review.hypothesis_rejection_authority is False
    assert review.feedback[0].automatic_hypothesis_rejection_permitted is False
