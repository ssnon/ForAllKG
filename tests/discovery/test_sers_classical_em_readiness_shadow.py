from __future__ import annotations

import pytest

from domains.sers.classical_em_evidence_contracts import (
    SERSClassicalEMPhysicsEvidence,
    SERSClassicalEMPhysicsEvidenceBundle,
    SERSEMSpectrumSnapshot,
)
from domains.sers.classical_em_readiness import SERSClassicalEMReadinessGate
from domains.sers.validation_design_contracts import (
    SERSGeometryCandidate,
    SERSSourceSpectralConstraint,
    SERSValidationDesign,
    SERSValidationDesignBundle,
)
from domains.sers.validation_routing_contracts import (
    SERSHypothesisValidationPlan,
    SERSHypothesisValidationPlanBundle,
    SERSValidationRoute,
)


def _plans(*, targets=None):
    route = SERSValidationRoute(
        route_id="route:em",
        route_kind="classical_em",
        statement="Evaluate the routed classical EM mechanism.",
        rationale="fixture",
        source_fields=["hypothesis_statement"],
        target_observables=targets or [
            "electromagnetic near-field enhancement",
            "substrate electromagnetic spectral response",
        ],
        validator_candidates=["fdtd"],
        readiness="ready",
        source_fdtd_subclaim_ids=["subclaim:em"],
        required_for_mechanism_assessment=True,
    )
    plan = SERSHypothesisValidationPlan(
        plan_id="plan:1",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_report_id="app:1",
        hypothesis_id="hypothesis:1",
        fdtd_applicability="partial",
        routes=[route],
        mechanism_claim_present=True,
        mechanism_assessment="deferred_pending_routed_evidence",
        integrated_experiment_required=False,
    )
    return SERSHypothesisValidationPlanBundle(
        bundle_id="plans:1",
        source_portfolio_id="portfolio:1",
        source_fdtd_applicability_bundle_id="apps:1",
        plans=[plan],
    )


def _designs():
    design = SERSValidationDesign(
        design_id="design:1",
        request_id="request:1",
        hypothesis_id="hypothesis:1",
        source_spec_id="spec:1",
        source_label="operator_screen",
        operator_note="Dimensions and water are screening assumptions, not source facts.",
        geometry_family="core_shell_nanorod",
        core_material="Au",
        shell_material="Ag",
        source_constraints=SERSSourceSpectralConstraint(
            baseline_wavelength_nm=785.0,
            reported_resonance_wavelengths_nm=[512.0, 772.0],
        ),
        geometry_candidates=[SERSGeometryCandidate(
            candidate_id="geometry:1",
            label="outer90x30_shell7p5",
            nanorod_length_nm=75.0,
            nanorod_diameter_nm=15.0,
            shell_thickness_nm=7.5,
            rationale="operator screening assumption",
        )],
        wavelength_sweep_nm=[450.0, 512.0, 772.0, 785.0, 850.0],
        polarizations=[
            "parallel_to_nanorod_long_axis",
            "perpendicular_to_nanorod_long_axis",
        ],
        surrounding_media=["water"],
    )
    return SERSValidationDesignBundle(
        bundle_id="designs:1",
        source_request_bundle_id="requests:1",
        designs=[design],
    )


def _evidence(*, blockers=None, actions=None):
    blockers = blockers if blockers is not None else [
        "spectral_boundary_censoring",
        "underresolved_geometry",
    ]
    actions = actions if actions is not None else [
        "resolve_spectral_boundary",
        "improve_spatial_resolution_or_solver_fidelity",
    ]
    row = SERSClassicalEMPhysicsEvidence(
        evidence_id="evidence:1",
        hypothesis_id="hypothesis:1",
        source_classical_em_handoff_id="handoff:1",
        source_validation_plan_id="plan:1",
        source_validation_route_id="route:em",
        source_fdtd_subclaim_ids=["subclaim:em"],
        source_simulation_spec_id="spec:1",
        source_solver_job_ids=["job:1"],
        source_calibration_observation_ids=["cal:1"],
        geometry_candidate_id="geometry:1",
        surrounding_medium="water",
        required_for_mechanism_assessment=True,
        spectrum_snapshots=[SERSEMSpectrumSnapshot(
            source_calibration_observation_id="cal:1",
            request_id="execution:1",
            source_solver_job_id="job:1",
            polarization="parallel_to_nanorod_long_axis",
            field_component="Ex",
            resolution_px_per_um=150.0,
            numerical_resolution_status="underresolved",
            evidence_eligibility="screening_only",
            peak_wavelength_nm=683.0,
            peak_scattering_cross_section_um2=0.02,
            peak_location_status="interior_peak",
            spectrum_wavelength_min_nm=450.0,
            spectrum_wavelength_max_nm=850.0,
            parent_quality_gate="observation_only_underresolved",
        )],
        numerical_evidence_state=(
            "boundary_limited_observation" if blockers else "calibration_candidate_review"
        ),
        numerical_blockers=blockers,
        next_required_actions=actions,
    )
    return SERSClassicalEMPhysicsEvidenceBundle(
        bundle_id="evidence_bundle:1",
        source_handoff_bundle_id="handoffs:1",
        source_solver_job_bundle_id="jobs:1",
        source_calibration_bundle_ids=["cals:1"],
        evidence=[row],
        evidence_count=1,
    )


def test_current_candidate_is_numerical_repair_and_model_validation_blocked():
    result = SERSClassicalEMReadinessGate().assess(
        _evidence(), _plans(), _designs()
    )
    row = result.assessments[0]
    assert row.overall_readiness == "numerical_repair_required"
    assert row.numerical_readiness == "repair_required"
    assert row.evidence_scope_status == "route_targets_partially_observed"
    assert row.observed_route_target_observables == [
        "substrate electromagnetic spectral response"
    ]
    assert row.unobserved_route_target_observables == [
        "electromagnetic near-field enhancement"
    ]
    assert row.model_form.model_form_status == "operator_concretized_unvalidated"
    assert row.model_form.positive_control_status == "not_assessed"
    assert "resolve_spectral_boundary" in row.immediate_numerical_repairs
    assert row.deferred_high_cost_actions == [
        "defer_high_cost_spatial_fidelity_escalation"
    ]
    assert row.high_cost_numerical_escalation_permitted is False
    assert row.mechanism_support_verdict_permitted is False


def test_numerically_clean_operator_model_still_requires_model_validation():
    result = SERSClassicalEMReadinessGate().assess(
        _evidence(blockers=[], actions=[]),
        _plans(targets=["substrate electromagnetic spectral response"]),
        _designs(),
    )
    row = result.assessments[0]
    assert row.numerical_readiness == "review_candidate"
    assert row.evidence_scope_status == "route_targets_observed"
    assert row.overall_readiness == "model_validation_required"
    assert "operator_concretized_model_unvalidated" in row.readiness_blockers
    assert "positive_control_not_assessed" in row.readiness_blockers


def test_unknown_route_observable_fails_closed_as_unobserved():
    result = SERSClassicalEMReadinessGate().assess(
        _evidence(blockers=[], actions=[]),
        _plans(targets=["custom electromagnetic quantity"]),
        _designs(),
    )
    row = result.assessments[0]
    assert row.observed_route_target_observables == []
    assert row.unobserved_route_target_observables == [
        "custom electromagnetic quantity"
    ]
    assert "route_target_observable_unobserved" in row.readiness_blockers
    assert "expand_classical_em_observables" in row.next_research_actions


def test_missing_geometry_lineage_fails_closed():
    evidence = _evidence().model_copy(deep=True)
    evidence.evidence[0].geometry_candidate_id = "geometry:missing"
    with pytest.raises(ValueError, match="geometry_candidate_id absent"):
        SERSClassicalEMReadinessGate().assess(evidence, _plans(), _designs())
