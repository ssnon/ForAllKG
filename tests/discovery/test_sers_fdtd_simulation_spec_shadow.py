from __future__ import annotations

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)

from domains.sers.fdtd_applicability import (
    SERSFDTDApplicabilityAnalyzer,
)
from domains.sers.simulation_compiler import (
    SERSHypothesisSimulationCompiler,
)
from domains.sers.simulation_validator import (
    SERSDeterministicSimulationValidator,
)


def _card(
    statement: str,
    *,
    hypothesis_id: str = "hypothesis:sers:test",
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        source_report_id="report",
        source_report_sha256="r" * 64,
        title="SERS FDTD fixture",
        hypothesis_statement=statement,
        hypothesis_type="design_lever_interaction",
        premise_statement_ids=["statement:1"],
        inferential_bridge=(
            "The proposed geometry changes electromagnetic coupling."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable="junction-localized electromagnetic enhancement",
                expected_direction="increase",
                rationale="plasmonic coupling is the proposed mediator",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable="junction-localized electromagnetic enhancement",
                falsifying_outcome="no enhancement relative to comparator",
            )
        ],
        source_paper_ids=["paper:1"],
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
        portfolio_id="portfolio:sers:test",
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        source_report_id="report",
        source_report_sha256="r" * 64,
        hypotheses=[card],
    )


def test_explicit_sphere_dimer_compiles_to_runnable_shadow_spec():
    card = _card(
        "At 785 nm excitation in water, two 40 nm-radius Au "
        "nanospheres forming a dimer with a 20 nm gap and polarization "
        "parallel to the dimer axis may increase junction enhancement."
    )

    portfolio = _portfolio(card)
    applicability = SERSFDTDApplicabilityAnalyzer().analyze_portfolio(
        portfolio
    )
    assert applicability.reports[0].applicability == "direct"

    bundle = SERSHypothesisSimulationCompiler().compile_portfolio(
        portfolio,
        applicability_bundle=applicability,
    )
    assert len(bundle.specs) == 1
    spec = bundle.specs[0]

    assert spec.fdtd_applicability == "direct"
    assert spec.geometry_family == "sphere_dimer"
    assert spec.left_material == "Au"
    assert spec.right_material == "Au"
    assert spec.core_material is None
    assert spec.shell_material is None
    assert spec.particle_radius_nm == 40.0
    assert spec.gap_nm == 20.0
    assert spec.excitation_wavelength_nm == 785.0
    assert spec.polarization == "parallel_to_dimer_axis"
    assert spec.surrounding_medium == "water"

    report = SERSDeterministicSimulationValidator().validate(spec)
    assert report.disposition == "runnable_shadow"
    assert report.issues == []

    assert spec.shadow_only is True
    assert spec.physics_authority_created is False
    assert spec.hypothesis_rejection_authority is False
    assert spec.feedback_generation_authority is False
    assert spec.canonical_graph_mutated is False


def test_missing_conditions_are_preserved_as_requires_concretization():
    card = _card(
        "Reducing the interparticle gap in an Au nanosphere dimer "
        "may increase junction-localized electromagnetic enhancement."
    )

    spec = (
        SERSHypothesisSimulationCompiler()
        .compile_portfolio(_portfolio(card))
        .specs[0]
    )

    assert spec.fdtd_applicability == "direct"
    assert spec.study_kind == "parameter_sweep"
    assert spec.sweep is not None
    assert spec.sweep.parameter == "gap_nm"
    assert spec.sweep.values == []
    assert spec.excitation_wavelength_nm is None
    assert spec.particle_radius_nm is None
    assert spec.surrounding_medium is None
    assert spec.polarization is None

    report = SERSDeterministicSimulationValidator().validate(spec)
    assert report.disposition == "requires_concretization"

    codes = {row.code for row in report.issues}
    assert "MISSING_SWEEP_VALUES" in codes
    assert "MISSING_NUMERIC_PARAMETER" in codes
    assert "MISSING_ENVIRONMENT" in codes
    assert "MISSING_EXCITATION_CONDITION" in codes


def test_diameter_is_derived_without_inventing_radius():
    card = _card(
        "At 785 nm excitation in air, an Au nanosphere dimer made "
        "from 80 nm-diameter particles with a 20 nm gap under "
        "longitudinal polarization parallel to the dimer axis may "
        "enhance the junction electromagnetic field."
    )

    spec = (
        SERSHypothesisSimulationCompiler()
        .compile_portfolio(_portfolio(card))
        .specs[0]
    )

    assert spec.particle_radius_nm == 40.0
    provenance = [
        row
        for row in spec.parameter_provenance
        if row.parameter == "particle_radius_nm"
    ]
    assert len(provenance) == 1
    assert provenance[0].source_kind == "derived_hypothesis_text"
    assert provenance[0].derivation == (
        "particle_radius_nm = diameter_nm / 2"
    )


def test_core_shell_sphere_dimer_is_not_silently_approximated():
    card = _card(
        "At 785 nm excitation in water, an Au@Ag core-shell "
        "nanosphere dimer with 40 nm particle radius, 20 nm gap, "
        "and polarization parallel to the dimer axis may enhance "
        "the electromagnetic near-field."
    )

    spec = (
        SERSHypothesisSimulationCompiler()
        .compile_portfolio(_portfolio(card))
        .specs[0]
    )

    assert "core_shell_sphere_dimer" in spec.unsupported_features
    assert spec.geometry_family is None
    assert spec.left_material is None
    assert spec.right_material is None

    report = SERSDeterministicSimulationValidator().validate(spec)
    assert report.disposition == "unsupported"
    assert any(
        row.code == "UNSUPPORTED_FEATURE"
        for row in report.issues
    )


def test_actual_analyte_wavelength_hypothesis_is_partial_and_decomposed():
    card = _card(
        "For methylene blue measured with an Au@Ag core-shell nanorod "
        "substrate, the excitation wavelength that maximizes SERS intensity "
        "will be shifted relative to the substrate-only 785 nm "
        "resonance-matching choice because the analyte-specific wavelength "
        "dependence contributes to the overall spectral response."
    ).model_copy(
        update={
            "title": (
                "Analyte-dependent spectral response modifies the "
                "substrate-resonance wavelength choice"
            ),
            "hypothesis_type": "context_dependency",
            "inferential_bridge": (
                "785 nm excitation is the substrate-only resonance "
                "matching reference, while methylene blue contributes "
                "an analyte-specific wavelength dependence."
            ),
            "predicted_observations": [
                PredictedObservation(
                    observation_id="prediction:mb",
                    observable=(
                        "The excitation wavelength producing the maximum "
                        "methylene-blue SERS intensity on the Au@Ag nanorod "
                        "substrate"
                    ),
                    expected_direction="shift",
                    rationale=(
                        "Analyte-specific wavelength dependence is coupled "
                        "to the Au@Ag LSPR response."
                    ),
                )
            ],
            "assumptions": [
                "Methylene-blue wavelength dependence transfers enough to "
                "motivate the substrate comparison."
            ],
        }
    )

    portfolio = _portfolio(card)
    applicability_bundle = (
        SERSFDTDApplicabilityAnalyzer().analyze_portfolio(portfolio)
    )
    applicability = applicability_bundle.reports[0]

    assert applicability.applicability == "partial"
    assert applicability.study_axes == ["wavelength"]
    assert applicability.baseline_wavelength_nm == 785.0
    assert applicability.detected_analytes == ["methylene blue"]
    assert "core_shell" in applicability.detected_geometry_terms
    assert "nanorod" in applicability.detected_geometry_terms
    assert len(applicability.fdtd_subclaims) == 1
    assert len(applicability.non_fdtd_subclaims) == 1

    spec = (
        SERSHypothesisSimulationCompiler()
        .compile_portfolio(
            portfolio,
            applicability_bundle=applicability_bundle,
        )
        .specs[0]
    )

    assert spec.fdtd_applicability == "partial"
    assert spec.geometry_family == "core_shell_nanorod"
    assert spec.left_material is None
    assert spec.right_material is None
    assert spec.core_material == "Au"
    assert spec.shell_material == "Ag"
    assert spec.study_kind == "wavelength_sweep"
    assert spec.baseline_wavelength_nm == 785.0
    assert spec.excitation_wavelength_nm is None
    assert spec.gap_nm is None
    assert spec.sweep is not None
    assert spec.sweep.parameter == "excitation_wavelength_nm"
    assert spec.sweep.values == []

    report = SERSDeterministicSimulationValidator().validate(spec)
    assert report.disposition == "requires_concretization"
    fields = {row.field for row in report.issues}
    codes = {row.code for row in report.issues}

    assert "gap_nm" not in fields
    assert "excitation_wavelength_nm" not in fields
    assert "nanorod_length_nm" in fields
    assert "nanorod_diameter_nm" in fields
    assert "shell_thickness_nm" in fields
    assert "MISSING_SWEEP_VALUES" in codes
    assert "MISSING_EXCITATION_CONDITION" in codes
    assert "MISSING_ENVIRONMENT" in codes


def test_non_fdtd_chemical_claim_routes_not_applicable_without_geometry_errors():
    card = _card(
        "Au@Ag charge transfer to methylene blue may dominate chemical "
        "enhancement of the measured Raman response."
    ).model_copy(
        update={
            "inferential_bridge": (
                "The proposed mechanism is charge-transfer-mediated chemical "
                "enhancement of the reporter response."
            ),
            "predicted_observations": [
                PredictedObservation(
                    observation_id="prediction:chemical",
                    observable="methylene-blue Raman response",
                    expected_direction="increase",
                    rationale="charge transfer changes the reporter response",
                )
            ],
        }
    )

    portfolio = _portfolio(card)
    applicability = (
        SERSFDTDApplicabilityAnalyzer()
        .analyze_portfolio(portfolio)
        .reports[0]
    )
    assert applicability.applicability == "not_applicable"
    assert not applicability.fdtd_subclaims
    assert applicability.non_fdtd_subclaims

    spec = (
        SERSHypothesisSimulationCompiler()
        .compile_portfolio(portfolio)
        .specs[0]
    )
    report = SERSDeterministicSimulationValidator().validate(spec)

    assert report.disposition == "not_applicable"
    assert {row.code for row in report.issues} == {"FDTD_NOT_APPLICABLE"}


def test_ambiguous_claim_routes_requires_interpretation():
    card = _card(
        "Surface preparation changes the SERS response under otherwise "
        "matched conditions."
    ).model_copy(
        update={
            "inferential_bridge": "The causal physical channel is unresolved.",
            "predicted_observations": [
                PredictedObservation(
                    observation_id="prediction:ambiguous",
                    observable="SERS response",
                    expected_direction="qualitative_change",
                    rationale="The response changes with preparation.",
                )
            ],
        }
    )

    portfolio = _portfolio(card)
    applicability = (
        SERSFDTDApplicabilityAnalyzer()
        .analyze_portfolio(portfolio)
        .reports[0]
    )
    assert applicability.applicability == "requires_interpretation"

    spec = (
        SERSHypothesisSimulationCompiler()
        .compile_portfolio(portfolio)
        .specs[0]
    )
    report = SERSDeterministicSimulationValidator().validate(spec)
    assert report.disposition == "requires_interpretation"


def test_wrong_domain_fails_closed():
    card = _card(
        "At 785 nm excitation in water, two 40 nm-radius Au "
        "nanospheres forming a dimer with a 20 nm gap and polarization "
        "parallel to the dimer axis may increase electromagnetic enhancement."
    )
    portfolio = _portfolio(card).model_copy(
        update={"domain_profile_id": "dac_her"}
    )

    try:
        SERSFDTDApplicabilityAnalyzer().analyze_portfolio(portfolio)
    except ValueError as exc:
        assert "domain_profile_id='sers_au_ag'" in str(exc)
    else:
        raise AssertionError("wrong-domain portfolio must fail closed")
