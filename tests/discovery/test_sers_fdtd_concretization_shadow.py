from __future__ import annotations

import pytest

from domains.sers.simulation_concretization import (
    SERSConcretizationPlanner,
    SERSOperatorConcretizer,
)
from domains.sers.simulation_concretization_contracts import (
    SERSConcretizationOverride,
    SERSConcretizationOverrideBundle,
)
from domains.sers.simulation_contracts import (
    SERSSimulationCompilationBundle,
    SERSSimulationSpec,
    SERSSweepDefinition,
)
from domains.sers.simulation_validator import (
    SERSDeterministicSimulationValidator,
)


def _missing_core_shell_nanorod_spec() -> SERSSimulationSpec:
    return SERSSimulationSpec(
        spec_id="sers_simulation_spec:fixture_missing",
        source_portfolio_id="portfolio:sers:test",
        source_applicability_report_id="sers_fdtd_applicability:fixture",
        hypothesis_id="hypothesis:b6a113075001443ad5f2",
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
        sweep=SERSSweepDefinition(
            parameter="excitation_wavelength_nm",
            values=[],
        ),
    )


def _compilation(spec: SERSSimulationSpec) -> SERSSimulationCompilationBundle:
    return SERSSimulationCompilationBundle(
        bundle_id="sers_simulation_bundle:fixture",
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
    requests = SERSConcretizationPlanner().plan(
        compilation,
        validation,
    )
    return compilation, validation, requests


def _valid_override(spec: SERSSimulationSpec) -> SERSConcretizationOverride:
    return SERSConcretizationOverride(
        hypothesis_id=spec.hypothesis_id,
        source_spec_id=spec.spec_id,
        source_label="fixture_operator_design",
        operator_note="Explicit screening conditions for shadow FDTD.",
        nanorod_length_nm=100.0,
        nanorod_diameter_nm=30.0,
        shell_thickness_nm=5.0,
        polarization="parallel_to_nanorod_long_axis",
        surrounding_medium="water",
        sweep_values=[650.0, 700.0, 750.0, 785.0, 800.0, 850.0, 900.0],
    )


def test_planner_exposes_only_explicitly_fillable_missing_fields():
    spec = _missing_core_shell_nanorod_spec()
    _, validation, requests = _plan(spec)

    assert validation.reports[0].disposition == "requires_concretization"
    request = requests.requests[0]
    assert request.status == "ready_for_operator_input"
    assert request.geometry_family == "core_shell_nanorod"
    assert request.baseline_wavelength_nm == 785.0

    assert {row.field for row in request.requirements} == {
        "nanorod_length_nm",
        "nanorod_diameter_nm",
        "shell_thickness_nm",
        "sweep.values",
        "polarization",
        "surrounding_medium",
    }


def test_operator_concretization_can_make_partial_hypothesis_runnable_shadow():
    spec = _missing_core_shell_nanorod_spec()
    compilation, _, requests = _plan(spec)
    override = _valid_override(spec)

    concretized = SERSOperatorConcretizer().apply(
        compilation,
        requests,
        SERSConcretizationOverrideBundle(overrides=[override]),
    )

    assert len(concretized.records) == 1
    assert len(concretized.specs) == 1
    output = concretized.specs[0]

    assert output.spec_id != spec.spec_id
    assert output.nanorod_length_nm == 100.0
    assert output.nanorod_diameter_nm == 30.0
    assert output.shell_thickness_nm == 5.0
    assert output.polarization == "parallel_to_nanorod_long_axis"
    assert output.surrounding_medium == "water"
    assert output.sweep is not None
    assert output.sweep.values == [
        650.0,
        700.0,
        750.0,
        785.0,
        800.0,
        850.0,
        900.0,
    ]

    report = SERSDeterministicSimulationValidator().validate(output)
    assert report.disposition == "runnable_shadow"
    assert report.issues == []

    operator_provenance = [
        row
        for row in output.parameter_provenance
        if row.source_kind == "operator_override"
    ]
    assert {row.parameter for row in operator_provenance} == {
        "nanorod_length_nm",
        "nanorod_diameter_nm",
        "shell_thickness_nm",
        "sweep.values",
        "polarization",
        "surrounding_medium",
    }

    record = concretized.records[0]
    assert record.source_spec_id == spec.spec_id
    assert record.output_spec_id == output.spec_id
    assert record.source_label == "fixture_operator_design"


def test_concretization_preserves_partial_applicability_and_authority_boundary():
    spec = _missing_core_shell_nanorod_spec()
    compilation, _, requests = _plan(spec)
    output = SERSOperatorConcretizer().apply(
        compilation,
        requests,
        SERSConcretizationOverrideBundle(
            overrides=[_valid_override(spec)]
        ),
    ).specs[0]

    assert output.fdtd_applicability == "partial"
    assert output.fdtd_subclaim_ids == ["sers_fdtd_subclaim:em"]
    assert output.non_fdtd_subclaim_ids == ["sers_fdtd_subclaim:analyte"]
    assert output.baseline_wavelength_nm == 785.0
    assert output.shadow_only is True
    assert output.physics_authority_created is False
    assert output.hypothesis_rejection_authority is False
    assert output.feedback_generation_authority is False
    assert output.canonical_graph_mutated is False


def test_wavelength_sweep_must_include_and_bracket_baseline():
    spec = _missing_core_shell_nanorod_spec()
    compilation, _, requests = _plan(spec)
    override = _valid_override(spec).model_copy(
        update={"sweep_values": [700.0, 750.0, 780.0]}
    )

    output = SERSOperatorConcretizer().apply(
        compilation,
        requests,
        SERSConcretizationOverrideBundle(overrides=[override]),
    ).specs[0]
    report = SERSDeterministicSimulationValidator().validate(output)

    assert report.disposition == "requires_concretization"
    codes = {row.code for row in report.issues}
    assert "MISSING_BASELINE_SWEEP_POINT" in codes
    assert "MISSING_BASELINE_BRACKETING" in codes


def test_core_shell_nanorod_core_length_must_exceed_core_diameter():
    spec = _missing_core_shell_nanorod_spec()
    compilation, _, requests = _plan(spec)
    override = _valid_override(spec).model_copy(
        update={
            "nanorod_length_nm": 30.0,
            "nanorod_diameter_nm": 40.0,
        }
    )

    output = SERSOperatorConcretizer().apply(
        compilation,
        requests,
        SERSConcretizationOverrideBundle(overrides=[override]),
    ).specs[0]
    report = SERSDeterministicSimulationValidator().validate(output)

    assert report.disposition == "invalid"
    assert "INVALID_NANOROD_ASPECT" in {
        row.code for row in report.issues
    }


def test_wrong_geometry_polarization_is_invalid():
    spec = _missing_core_shell_nanorod_spec()
    compilation, _, requests = _plan(spec)
    override = _valid_override(spec).model_copy(
        update={"polarization": "parallel_to_dimer_axis"}
    )

    output = SERSOperatorConcretizer().apply(
        compilation,
        requests,
        SERSConcretizationOverrideBundle(overrides=[override]),
    ).specs[0]
    report = SERSDeterministicSimulationValidator().validate(output)

    assert report.disposition == "invalid"
    assert "INVALID_POLARIZATION_GEOMETRY" in {
        row.code for row in report.issues
    }


def test_operator_cannot_override_field_not_requested_by_validation():
    spec = _missing_core_shell_nanorod_spec().model_copy(
        update={"nanorod_length_nm": 100.0}
    )
    compilation, _, requests = _plan(spec)
    override = _valid_override(spec)

    with pytest.raises(ValueError, match="was not requested"):
        SERSOperatorConcretizer().apply(
            compilation,
            requests,
            SERSConcretizationOverrideBundle(overrides=[override]),
        )


def test_unedited_empty_sweep_template_is_not_a_concretization():
    spec = _missing_core_shell_nanorod_spec()
    compilation, _, requests = _plan(spec)
    override = _valid_override(spec).model_copy(
        update={"sweep_values": []}
    )

    with pytest.raises(ValueError, match="empty template"):
        SERSOperatorConcretizer().apply(
            compilation,
            requests,
            SERSConcretizationOverrideBundle(overrides=[override]),
        )
