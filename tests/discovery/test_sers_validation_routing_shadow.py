from __future__ import annotations

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)
from domains.sers.validation_routing import SERSHypothesisValidationRouter


def _portfolio(
    statement: str,
    bridge: str,
    observable: str,
    rationale: str = "test",
    expected_direction: str = "qualitative_change",
    assumptions: list[str] | None = None,
):
    source_context_id = "context:test"
    source_context_sha256 = "c" * 64
    source_report_id = "report:test"
    source_report_sha256 = "d" * 64
    card = HypothesisCard(
        hypothesis_id="hypothesis:test",
        domain_profile_id="sers_au_ag",
        source_context_id=source_context_id,
        source_context_sha256=source_context_sha256,
        source_report_id=source_report_id,
        source_report_sha256=source_report_sha256,
        title="test",
        hypothesis_statement=statement,
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:test"],
        inferential_bridge=bridge,
        predicted_observations=[PredictedObservation(
            observation_id="prediction:test",
            observable=observable,
            expected_direction=expected_direction,
            rationale=rationale,
        )],
        falsification_criteria=[FalsificationCriterion(
            criterion_id="falsifier:test",
            observable=observable,
            falsifying_outcome="The predicted directional or qualitative effect is absent.",
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
    return HypothesisPortfolio(
        portfolio_id="portfolio:test",
        domain_profile_id="sers_au_ag",
        source_context_id=source_context_id,
        source_context_sha256=source_context_sha256,
        source_report_id=source_report_id,
        source_report_sha256=source_report_sha256,
        hypotheses=[card],
    )


def test_composite_sers_hypothesis_routes_em_molecular_and_experiment():
    portfolio = _portfolio(
        statement=(
            "For methylene blue on an Au@Ag nanorod, the excitation wavelength "
            "that maximizes SERS intensity shifts because analyte-specific "
            "wavelength dependence modifies the plasmonic substrate response."
        ),
        bridge="The substrate has an LSPR response and methylene blue is wavelength-dependent.",
        observable="excitation wavelength producing maximum methylene-blue SERS intensity",
        expected_direction="shift",
        assumptions=[
            "Methylene-blue wavelength dependence is transferable enough to motivate the test.",
            "Synthesis and temperature treatment do not completely dominate the wavelength response.",
        ],
    )
    plan = SERSHypothesisValidationRouter().plan_portfolio(portfolio).plans[0]
    kinds = {row.route_kind for row in plan.routes}
    assert kinds == {
        "classical_em",
        "molecular_spectroscopy",
        "integrated_sers_experiment",
    }
    assert plan.fdtd_applicability == "partial"
    assert plan.integrated_experiment_required is True
    assert plan.mechanism_claim_present is True
    assert plan.mechanism_assessment == "deferred_pending_routed_evidence"
    assert plan.whole_hypothesis_assessment == "deferred_pending_routed_evidence"

    by_kind = {row.route_kind: row for row in plan.routes}
    assert by_kind["classical_em"].required_for_outcome_assessment is False
    assert by_kind["classical_em"].required_for_mechanism_assessment is True
    assert by_kind["molecular_spectroscopy"].required_for_outcome_assessment is False
    assert by_kind["molecular_spectroscopy"].required_for_mechanism_assessment is True
    assert by_kind["integrated_sers_experiment"].required_for_outcome_assessment is True
    assert by_kind["integrated_sers_experiment"].required_for_mechanism_assessment is True
    assert all(
        not field.startswith("assumptions[") and field != "title"
        for route in plan.routes
        for field in route.source_fields
    )
    assert {note.source_field for note in plan.context_notes} == {
        "assumptions[0]",
        "assumptions[1]",
    }
    process_note = next(
        note for note in plan.context_notes if note.source_field == "assumptions[1]"
    )
    assert "fabrication_process" in process_note.matched_topics
    assert process_note.routing_effect == "context_only"


def test_pure_em_claim_does_not_force_integrated_sers_experiment():
    portfolio = _portfolio(
        statement=(
            "Reducing the interparticle gap increases electromagnetic near-field "
            "enhancement through plasmonic coupling."
        ),
        bridge="Gap-dependent plasmonic coupling controls local field enhancement.",
        observable="electromagnetic near-field enhancement in the gap",
        expected_direction="increase",
    )
    plan = SERSHypothesisValidationRouter().plan_portfolio(portfolio).plans[0]
    kinds = {row.route_kind for row in plan.routes}
    assert kinds == {"classical_em"}
    assert plan.integrated_experiment_required is False
    route = plan.routes[0]
    assert route.required_for_outcome_assessment is True
    assert route.required_for_mechanism_assessment is True


def test_surface_chemistry_claim_is_not_forced_into_fdtd():
    portfolio = _portfolio(
        statement=(
            "A reporter with stronger chemical association to the Ag-containing "
            "surface yields greater SERS intensity than a weakly associating reporter."
        ),
        bridge="Surface association changes reporter proximity and chemical enhancement.",
        observable="relative SERS intensity of strongly versus weakly associating reporter",
        expected_direction="increase",
    )
    plan = SERSHypothesisValidationRouter().plan_portfolio(portfolio).plans[0]
    kinds = {row.route_kind for row in plan.routes}
    assert "surface_chemistry" in kinds
    assert "integrated_sers_experiment" in kinds
    assert "classical_em" not in kinds
    by_kind = {row.route_kind: row for row in plan.routes}
    assert by_kind["surface_chemistry"].required_for_mechanism_assessment is True
    assert by_kind["surface_chemistry"].required_for_outcome_assessment is False
    assert by_kind["integrated_sers_experiment"].required_for_outcome_assessment is True


def test_ambiguous_hypothesis_fails_to_human_interpretation_route():
    portfolio = _portfolio(
        statement="Surface preparation changes the measured response.",
        bridge="The responsible physical mechanism is unresolved.",
        observable="measurement response",
    )
    plan = SERSHypothesisValidationRouter().plan_portfolio(portfolio).plans[0]
    assert [row.route_kind for row in plan.routes] == ["unresolved"]
    assert plan.routes[0].readiness == "requires_interpretation"


def test_assumption_only_process_signal_is_context_not_route():
    portfolio = _portfolio(
        statement="A plasmonic geometry increases electromagnetic near-field enhancement.",
        bridge="The proposed geometry concentrates the local electromagnetic field.",
        observable="electromagnetic near-field enhancement",
        expected_direction="increase",
        assumptions=[
            "Synthesis temperature and aging do not dominate the observed response."
        ],
    )
    plan = SERSHypothesisValidationRouter().plan_portfolio(portfolio).plans[0]
    assert {row.route_kind for row in plan.routes} == {"classical_em"}
    assert all(row.route_kind != "fabrication_process" for row in plan.routes)
    assert len(plan.context_notes) == 1
    assert plan.context_notes[0].source_field == "assumptions[0]"
    assert "fabrication_process" in plan.context_notes[0].matched_topics
