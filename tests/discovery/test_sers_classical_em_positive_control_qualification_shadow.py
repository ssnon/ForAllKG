from domains.sers.classical_em_positive_control import (
    SERSClassicalEMPositiveControlQualifier,
)
from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlReference,
    SERSEMPositiveControlReferenceBundle,
)


def _ref(**overrides):
    data = dict(
        control_id="control:1",
        source_label="independent_single_particle_control",
        source_locator="doi:10.example/control",
        source_note="source-extracted facts only",
        core_material="Au",
        shell_material="Ag",
        particle_shape="spherocylinder_like",
        shell_geometry="uniform_thickness",
        core_length_nm=80.0,
        core_diameter_nm=20.0,
        uniform_shell_thickness_nm=5.0,
        environment="homogeneous_water",
        illumination="normal_incidence_plane_wave",
        polarization="parallel_to_nanorod_long_axis",
        observable="single_particle_scattering_spectrum",
        reference_peak_wavelength_nm=650.0,
        reference_peak_label="longitudinal LSPR scattering maximum",
        independent_reference=True,
    )
    data.update(overrides)
    return SERSEMPositiveControlReference(**data)


def _qualify(ref):
    return SERSClassicalEMPositiveControlQualifier().qualify(
        SERSEMPositiveControlReferenceBundle(references=[ref])
    ).qualifications[0]


def test_exact_backend_compatible_reference_is_execution_candidate_only():
    row = _qualify(_ref())
    assert row.status == "qualified_for_current_backend"
    assert row.blockers == []
    assert row.positive_control_execution_candidate is True
    assert row.model_validation_permitted is False
    assert row.target_hypothesis_verdict_permitted is False


def test_faceted_anisotropic_shell_fails_closed():
    row = _qualify(_ref(
        particle_shape="cuboid_or_faceted",
        shell_geometry="anisotropic_or_faceted",
        uniform_shell_thickness_nm=None,
    ))
    assert row.status == "requires_model_extension"
    assert "particle_shape_mismatch" in row.blockers
    assert "shell_geometry_mismatch" in row.blockers
    assert row.positive_control_execution_candidate is False


def test_dark_field_substrate_reference_is_not_silently_equated_to_plane_wave_water():
    row = _qualify(_ref(
        environment="substrate_with_water",
        illumination="dark_field_high_na",
        polarization="unpolarized",
    ))
    assert row.status == "requires_model_extension"
    assert set(row.blockers) >= {
        "environment_mismatch",
        "illumination_mismatch",
        "polarization_mismatch",
    }


def test_extinction_reference_does_not_validate_scattering_backend():
    row = _qualify(_ref(observable="ensemble_extinction_spectrum"))
    assert row.status == "requires_model_extension"
    assert row.blockers == ["observable_mismatch"]


def test_missing_peak_is_insufficient_reference_detail_not_model_mismatch():
    row = _qualify(_ref(reference_peak_wavelength_nm=None))
    assert row.status == "insufficient_reference_detail"
    assert row.blockers == ["reference_peak_missing"]
    assert row.backend_geometry_representable is True
    assert row.backend_observable_representable is True
