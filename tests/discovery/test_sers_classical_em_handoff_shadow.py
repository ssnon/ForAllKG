from __future__ import annotations

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from domains.sers.classical_em_handoff import SERSClassicalEMHandoffBuilder
from domains.sers.validation_routing import SERSHypothesisValidationRouter


def _card(
    statement: str,
    *,
    bridge: str,
    observable: str,
    expected_direction: str = "increase",
    assumptions: list[str] | None = None,
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="hypothesis:test",
        domain_profile_id="sers_au_ag",
        source_context_id="context:test",
        source_context_sha256="c" * 64,
        source_report_id="report:test",
        source_report_sha256="r" * 64,
        title="context title must not feed routed FDTD compilation",
        hypothesis_statement=statement,
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=["statement:test"],
        inferential_bridge=bridge,
        predicted_observations=[PredictedObservation(
            observation_id="prediction:test",
            observable=observable,
            expected_direction=expected_direction,
            rationale="The proposed mechanism should alter the stated observable.",
        )],
        falsification_criteria=[FalsificationCriterion(
            criterion_id="falsifier:test",
            observable=observable,
            falsifying_outcome="No directional change under a matched comparison.",
        )],
        assumptions=assumptions or [],
        source_paper_ids=["paper:test"],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )


def _portfolio(card: HypothesisCard) -> HypothesisPortfolio:
    return HypothesisPortfolio(
        portfolio_id="portfolio:test",
        domain_profile_id="sers_au_ag",
        source_context_id="context:test",
        source_context_sha256="c" * 64,
        source_report_id="report:test",
        source_report_sha256="r" * 64,
        hypotheses=[card],
    )


def _route(portfolio: HypothesisPortfolio):
    return SERSHypothesisValidationRouter().plan_portfolio(portfolio)


def test_context_only_radius_does_not_leak_into_fdtd_spec():
    portfolio = _portfolio(_card(
        (
            "For an Au nanosphere dimer, reducing the 20 nm interparticle gap "
            "increases electromagnetic near-field enhancement through plasmonic coupling."
        ),
        bridge="Gap-dependent plasmonic coupling controls local field enhancement.",
        observable="electromagnetic near-field enhancement in the gap",
        assumptions=[
            "A 40 nm particle radius would be a convenient simulation starting point."
        ],
    ))
    plans = _route(portfolio)
    handoffs, compilation, validation, _ = SERSClassicalEMHandoffBuilder().build(
        portfolio, plans
    )

    assert handoffs.routed_hypothesis_count == 1
    assert handoffs.not_routed_hypothesis_ids == []
    spec = compilation.specs[0]
    assert spec.gap_nm == pytest.approx(20.0)
    assert spec.particle_radius_nm is None
    assert validation.reports[0].disposition == "requires_concretization"
    assert all("assumptions[" not in row.source_field for row in spec.parameter_provenance)
    handoff = handoffs.handoffs[0]
    assert handoff.backend_role == "em_evidence_generator"
    assert handoff.whole_hypothesis_verdict_permitted is False
    assert handoff.integrated_sers_outcome_verdict_permitted is False


def test_context_only_shell_thickness_does_not_become_core_shell_parameter():
    portfolio = _portfolio(_card(
        (
            "For methylene blue on an Au@Ag core-shell nanorod substrate, the "
            "excitation wavelength that maximizes SERS intensity shifts relative "
            "to the substrate-only 785 nm resonance-matching choice because "
            "analyte-specific wavelength dependence modifies the plasmonic response."
        ),
        bridge=(
            "The Au@Ag nanorod resonances at 512 and 772 nm contribute through "
            "the LSPR response."
        ),
        observable="excitation wavelength producing maximum methylene-blue SERS intensity",
        expected_direction="shift",
        assumptions=["A 5 nm shell thickness could be used as an initial model choice."],
    ))
    plans = _route(portfolio)
    handoffs, compilation, _, _ = SERSClassicalEMHandoffBuilder().build(portfolio, plans)

    spec = compilation.specs[0]
    assert spec.geometry_family == "core_shell_nanorod"
    assert spec.shell_thickness_nm is None
    assert spec.baseline_wavelength_nm == pytest.approx(785.0)
    assert spec.reported_resonance_wavelengths_nm == [512.0, 772.0]
    assert handoffs.handoffs[0].required_for_mechanism_assessment is True
    assert handoffs.handoffs[0].required_for_outcome_assessment is False


def test_surface_chemistry_only_hypothesis_is_not_sent_to_fdtd():
    portfolio = _portfolio(_card(
        (
            "A reporter with stronger chemical association to the Ag-containing "
            "surface yields greater SERS intensity than a weakly associating reporter."
        ),
        bridge="Surface association and chemical enhancement control reporter response.",
        observable="relative SERS intensity of strongly versus weakly associating reporter",
    ))
    plans = _route(portfolio)
    handoffs, compilation, validation, _ = SERSClassicalEMHandoffBuilder().build(
        portfolio, plans
    )

    assert handoffs.routed_hypothesis_count == 0
    assert handoffs.not_routed_hypothesis_ids == ["hypothesis:test"]
    assert compilation.specs == []
    assert validation.reports == []


def test_fdtd_subclaim_lineage_mismatch_fails_closed():
    portfolio = _portfolio(_card(
        "Reducing a nanosphere-dimer gap increases electromagnetic near-field enhancement.",
        bridge="Plasmonic coupling localizes the field in the gap.",
        observable="electromagnetic near-field enhancement",
    ))
    plans = _route(portfolio)
    plan = plans.plans[0]
    routes = [
        row.model_copy(update={"source_fdtd_subclaim_ids": ["sers_fdtd_subclaim:wrong"]})
        if row.route_kind == "classical_em" else row
        for row in plan.routes
    ]
    bad_plan = plan.model_copy(update={"routes": routes})
    bad_bundle = plans.model_copy(update={"plans": [bad_plan]})

    with pytest.raises(ValueError, match="subclaim absent"):
        SERSClassicalEMHandoffBuilder().build(portfolio, bad_bundle)
