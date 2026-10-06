from domains.sers.classical_em_backend_gap import SERSClassicalEMBackendGapAssessor
from domains.sers.classical_em_positive_control import SERSClassicalEMPositiveControlQualifier
from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlReference,
    SERSEMPositiveControlReferenceBundle,
)


def _reference(
    control_id: str,
    source_locator: str,
    *,
    environment="substrate_with_water",
    particle_shape="cuboid_or_faceted",
    shell_geometry="anisotropic_or_faceted",
    illumination="dark_field_high_na",
    polarization="unpolarized",
    observable="single_particle_scattering_spectrum",
    peak=700.0,
):
    return SERSEMPositiveControlReference(
        control_id=control_id,
        source_label=control_id,
        source_locator=source_locator,
        core_material="Au",
        shell_material="Ag",
        particle_shape=particle_shape,
        shell_geometry=shell_geometry,
        core_length_nm=73.0,
        core_diameter_nm=18.0,
        environment=environment,
        illumination=illumination,
        polarization=polarization,
        observable=observable,
        reference_peak_wavelength_nm=peak,
    )


def _assess(*refs):
    q = SERSClassicalEMPositiveControlQualifier().qualify(
        SERSEMPositiveControlReferenceBundle(references=list(refs))
    )
    return SERSClassicalEMBackendGapAssessor().assess(q)


def test_same_source_repetition_does_not_trigger_extension_priority():
    result = _assess(
        _reference("control:a", "paper:1"),
        _reference("control:b", "paper:1"),
    )
    assert result.control_count == 2
    assert result.independent_source_count == 1
    assert result.extension_decision == "collect_more_independent_controls"
    assert set(result.shared_blockers_across_model_extension_controls) == {
        "particle_shape_mismatch",
        "shell_geometry_mismatch",
        "environment_mismatch",
        "illumination_mismatch",
        "polarization_mismatch",
    }
    assert all(
        row.priority_status == "insufficient_independent_source_support"
        for row in result.gap_assessments
    )
    assert all(row.single_axis_unlock_count == 0 for row in result.gap_assessments)
    assert result.backend_extension_authorized is False


def test_two_independent_sources_can_create_targeted_extension_candidate():
    kwargs = dict(
        particle_shape="spherocylinder_like",
        shell_geometry="uniform_thickness",
        illumination="normal_incidence_plane_wave",
        polarization="parallel_to_nanorod_long_axis",
    )
    refs = []
    for control_id, source in (("control:a", "paper:1"), ("control:b", "paper:2")):
        row = _reference(control_id, source, environment="substrate_with_water", **kwargs)
        row = row.model_copy(update={"uniform_shell_thickness_nm": 5.0})
        refs.append(row)
    result = _assess(*refs)
    assert result.extension_decision == "targeted_extension_candidate_available"
    gap = next(row for row in result.gap_assessments if row.blocker == "environment_mismatch")
    assert gap.independent_source_count == 2
    assert gap.single_axis_unlock_count == 2
    assert gap.priority_status == "cross_source_gap_candidate"


def test_cross_source_multi_axis_gaps_require_joint_design():
    result = _assess(
        _reference("control:a", "paper:1"),
        _reference("control:b", "paper:2"),
    )
    assert result.extension_decision == "cross_source_gaps_require_joint_design"
    assert all(
        row.priority_status == "cross_source_gap_candidate"
        for row in result.gap_assessments
    )
    assert all(row.single_axis_unlock_count == 0 for row in result.gap_assessments)


def test_missing_peak_is_reference_curation_not_backend_extension():
    ref = _reference(
        "control:a",
        "paper:1",
        particle_shape="spherocylinder_like",
        shell_geometry="uniform_thickness",
        environment="homogeneous_water",
        illumination="normal_incidence_plane_wave",
        polarization="parallel_to_nanorod_long_axis",
        peak=None,
    ).model_copy(update={"uniform_shell_thickness_nm": 5.0})
    result = _assess(ref)
    assert result.extension_decision == "no_backend_extension_indicated"
    assert result.insufficient_detail_control_count == 1
    assert len(result.gap_assessments) == 1
    gap = result.gap_assessments[0]
    assert gap.blocker == "reference_peak_missing"
    assert gap.gap_kind == "reference_detail_gap"
    assert gap.priority_status == "reference_curation_required"
    assert result.recommended_next_actions == [
        "curate_missing_positive_control_reference_details"
    ]


def test_qualified_reference_points_to_execution_planning_without_extension():
    ref = _reference(
        "control:a",
        "paper:1",
        particle_shape="spherocylinder_like",
        shell_geometry="uniform_thickness",
        environment="homogeneous_water",
        illumination="normal_incidence_plane_wave",
        polarization="parallel_to_nanorod_long_axis",
    ).model_copy(update={"uniform_shell_thickness_nm": 5.0})
    result = _assess(ref)
    assert result.qualified_control_count == 1
    assert result.gap_assessments == []
    assert result.extension_decision == "no_backend_extension_indicated"
    assert result.recommended_next_actions == [
        "plan_qualified_positive_control_execution"
    ]
    assert result.model_validation_permitted is False
