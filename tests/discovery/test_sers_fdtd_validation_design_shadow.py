from __future__ import annotations

import pytest

from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisEvidenceProfile,
    HypothesisPortfolio,
    PredictedObservation,
)

from domains.sers.simulation_compiler import (
    SERSHypothesisSimulationCompiler,
)
from domains.sers.simulation_contracts import (
    SERSSimulationCompilationBundle,
    SERSSimulationSpec,
    SERSSweepDefinition,
)
from domains.sers.simulation_validator import (
    SERSDeterministicSimulationValidator,
)
from domains.sers.validation_design import (
    SERSValidationDesignBuilder,
    SERSValidationDesignPlanner,
)
from domains.sers.validation_design_contracts import (
    SERSGeometryCandidate,
    SERSValidationDesignOverride,
    SERSValidationDesignOverrideBundle,
)


def _missing_spec() -> SERSSimulationSpec:
    return SERSSimulationSpec(
        spec_id="sers_simulation_spec:s13_fixture",
        source_portfolio_id="portfolio:sers:test",
        source_applicability_report_id="sers_fdtd_applicability:fixture",
        hypothesis_id="hypothesis:s13",
        domain_profile_id="sers_au_ag",
        source_hypothesis_text_sha256="a" * 64,
        fdtd_applicability="partial",
        fdtd_subclaim_ids=["sers_fdtd_subclaim:em"],
        non_fdtd_subclaim_ids=["sers_fdtd_subclaim:analyte"],
        study_axes=["wavelength"],
        study_kind="wavelength_sweep",
        geometry_family="core_shell_nanorod",
        core_material="Au",
        shell_material="Ag",
        baseline_wavelength_nm=785.0,
        reported_resonance_wavelengths_nm=[512.0, 772.0],
        sweep=SERSSweepDefinition(
            parameter="excitation_wavelength_nm",
            values=[],
        ),
    )


def _compilation(spec: SERSSimulationSpec) -> SERSSimulationCompilationBundle:
    return SERSSimulationCompilationBundle(
        bundle_id="sers_simulation_bundle:s13_fixture",
        source_portfolio_id=spec.source_portfolio_id,
        source_applicability_bundle_id="sers_fdtd_applicability_bundle:fixture",
        domain_profile_id="sers_au_ag",
        specs=[spec],
    )


def _plan(spec: SERSSimulationSpec):
    compilation = _compilation(spec)
    validation = SERSDeterministicSimulationValidator().validate_bundle(
        compilation
    )
    requests = SERSValidationDesignPlanner().plan(compilation, validation)
    return compilation, validation, requests


def _override(spec: SERSSimulationSpec) -> SERSValidationDesignOverride:
    return SERSValidationDesignOverride(
        hypothesis_id=spec.hypothesis_id,
        source_spec_id=spec.spec_id,
        source_label="fixture_multi_case_screening_design",
        operator_note=(
            "Representative screening geometries; not source-derived facts."
        ),
        geometry_candidates=[
            SERSGeometryCandidate(
                candidate_id="geometry:a",
                label="screening geometry A",
                nanorod_length_nm=90.0,
                nanorod_diameter_nm=30.0,
                shell_thickness_nm=5.0,
                rationale="explicit operator screening assumption",
            ),
            SERSGeometryCandidate(
                candidate_id="geometry:b",
                label="screening geometry B",
                nanorod_length_nm=120.0,
                nanorod_diameter_nm=35.0,
                shell_thickness_nm=8.0,
                rationale="explicit operator screening assumption",
            ),
        ],
        wavelength_sweep_nm=[
            500.0,
            512.0,
            600.0,
            700.0,
            772.0,
            785.0,
            850.0,
        ],
        polarizations=[
            "parallel_to_nanorod_long_axis",
            "perpendicular_to_nanorod_long_axis",
        ],
        surrounding_media=["air", "water"],
    )


def _source_card() -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="hypothesis:source_resonances",
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        source_report_id="report",
        source_report_sha256="r" * 64,
        title="Analyte-dependent spectral response modifies wavelength choice",
        hypothesis_statement=(
            "For methylene blue measured with an Au@Ag core-shell nanorod "
            "substrate, the excitation wavelength that maximizes SERS "
            "intensity will be shifted relative to the substrate-only 785 nm "
            "resonance-matching choice because analyte-specific wavelength "
            "dependence contributes to the overall response."
        ),
        hypothesis_type="context_dependency",
        premise_statement_ids=["statement:1"],
        inferential_bridge=(
            "The Au@Ag nanorod resonances at 512 and 772 nm are reported to "
            "match 785 nm excitation, and the Au@Ag response has an LSPR "
            "contribution."
        ),
        predicted_observations=[
            PredictedObservation(
                observation_id="prediction:1",
                observable=(
                    "excitation wavelength producing maximum methylene-blue "
                    "SERS intensity"
                ),
                expected_direction="shift",
                rationale="analyte-specific wavelength dependence",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="falsifier:1",
                observable="spectral optimum",
                falsifying_outcome="no shift under matched comparison",
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


def test_compiler_preserves_reported_resonances_as_source_constraints():
    card = _source_card()
    portfolio = HypothesisPortfolio(
        portfolio_id="portfolio:source_resonances",
        domain_profile_id="sers_au_ag",
        source_context_id="ctx",
        source_context_sha256="c" * 64,
        source_report_id="report",
        source_report_sha256="r" * 64,
        hypotheses=[card],
    )

    spec = SERSHypothesisSimulationCompiler().compile_portfolio(portfolio).specs[0]

    assert spec.baseline_wavelength_nm == 785.0
    assert spec.reported_resonance_wavelengths_nm == [512.0, 772.0]
    assert 785.0 not in spec.reported_resonance_wavelengths_nm
    provenance = [
        row
        for row in spec.parameter_provenance
        if row.parameter == "reported_resonance_wavelengths_nm"
    ]
    assert len(provenance) == 2
    assert all(row.source_kind == "hypothesis_text" for row in provenance)


def test_planner_lifts_six_missing_fields_into_four_design_requirements():
    spec = _missing_spec()
    _, validation, requests = _plan(spec)

    assert validation.reports[0].disposition == "requires_concretization"
    request = requests.requests[0]
    assert request.status == "ready_for_design_input"
    assert request.focal_axes == ["wavelength"]
    assert request.uncertainty_axes == [
        "geometry_candidate",
        "polarization",
        "surrounding_medium",
    ]
    assert {row.field for row in request.requirements} == {
        "geometry_candidates",
        "wavelength_sweep_nm",
        "polarizations",
        "surrounding_media",
    }
    assert request.source_constraints.baseline_wavelength_nm == 785.0
    assert request.source_constraints.reported_resonance_wavelengths_nm == [
        512.0,
        772.0,
    ]


def test_validation_design_expands_one_hypothesis_into_many_cases():
    spec = _missing_spec()
    compilation, _, requests = _plan(spec)

    designs, cases = SERSValidationDesignBuilder().build(
        compilation,
        requests,
        SERSValidationDesignOverrideBundle(overrides=[_override(spec)]),
    )

    assert len(designs.designs) == 1
    design = designs.designs[0]
    assert len(design.geometry_candidates) == 2
    assert len(design.wavelength_sweep_nm) == 7
    assert len(design.polarizations) == 2
    assert len(design.surrounding_media) == 2

    # 2 geometry candidates x 7 wavelengths x 2 polarizations x 2 media.
    assert len(cases.cases) == 56
    assert len({row.case_id for row in cases.cases}) == 56


def test_every_case_preserves_source_constraints_and_operator_design_axes():
    spec = _missing_spec()
    compilation, _, requests = _plan(spec)
    _, cases = SERSValidationDesignBuilder().build(
        compilation,
        requests,
        SERSValidationDesignOverrideBundle(overrides=[_override(spec)]),
    )

    for case in cases.cases:
        assert case.core_material == "Au"
        assert case.shell_material == "Ag"
        assert case.baseline_wavelength_nm == 785.0
        assert case.reported_resonance_wavelengths_nm == [512.0, 772.0]
        assert case.geometry_candidate_id in {"geometry:a", "geometry:b"}
        assert case.shadow_only is True
        assert case.physics_authority_created is False
        assert case.hypothesis_rejection_authority is False
        assert case.feedback_generation_authority is False
        assert case.canonical_graph_mutated is False


def test_source_spectral_landmarks_cannot_be_dropped_from_design_sweep():
    spec = _missing_spec()
    compilation, _, requests = _plan(spec)
    override = _override(spec).model_copy(
        update={
            "wavelength_sweep_nm": [500.0, 600.0, 700.0, 785.0, 850.0]
        }
    )

    with pytest.raises(ValueError, match="source-derived spectral landmarks"):
        SERSValidationDesignBuilder().build(
            compilation,
            requests,
            SERSValidationDesignOverrideBundle(overrides=[override]),
        )


def test_design_sweep_must_bracket_source_baseline():
    spec = _missing_spec()
    compilation, _, requests = _plan(spec)
    override = _override(spec).model_copy(
        update={
            "wavelength_sweep_nm": [512.0, 700.0, 772.0, 785.0]
        }
    )

    with pytest.raises(ValueError, match="bracket"):
        SERSValidationDesignBuilder().build(
            compilation,
            requests,
            SERSValidationDesignOverrideBundle(overrides=[override]),
        )


def test_geometry_candidates_are_correlated_tuples_not_independent_axes():
    spec = _missing_spec()
    compilation, _, requests = _plan(spec)
    designs, _ = SERSValidationDesignBuilder().build(
        compilation,
        requests,
        SERSValidationDesignOverrideBundle(overrides=[_override(spec)]),
    )

    design = designs.designs[0]
    assert [
        (
            row.nanorod_length_nm,
            row.nanorod_diameter_nm,
            row.shell_thickness_nm,
        )
        for row in design.geometry_candidates
    ] == [(90.0, 30.0, 5.0), (120.0, 35.0, 8.0)]


def test_incomplete_design_does_not_generate_cases():
    spec = _missing_spec()
    compilation, _, requests = _plan(spec)
    override = _override(spec).model_copy(update={"geometry_candidates": []})

    with pytest.raises(ValueError, match="incomplete"):
        SERSValidationDesignBuilder().build(
            compilation,
            requests,
            SERSValidationDesignOverrideBundle(overrides=[override]),
        )
