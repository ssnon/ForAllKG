import pytest

from domains.sers.classical_em_backend_gap import SERSClassicalEMBackendGapAssessor
from domains.sers.classical_em_extension_design import (
    SERSClassicalEMJointExtensionDesigner,
)
from domains.sers.classical_em_positive_control import (
    SERSClassicalEMPositiveControlQualifier,
)
from domains.sers.classical_em_positive_control_contracts import (
    SERSEMPositiveControlReference,
    SERSEMPositiveControlReferenceBundle,
)


def _reference(
    control_id: str,
    source_locator: str,
    *,
    particle_shape="cuboid_or_faceted",
    shell_geometry="anisotropic_or_faceted",
    environment="substrate_with_water",
    illumination="dark_field_high_na",
    polarization="unpolarized",
    observable="single_particle_scattering_spectrum",
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
        reference_peak_wavelength_nm=700.0,
    )


def _design(*refs):
    qualifications = SERSClassicalEMPositiveControlQualifier().qualify(
        SERSEMPositiveControlReferenceBundle(references=list(refs))
    )
    gaps = SERSClassicalEMBackendGapAssessor().assess(qualifications)
    return qualifications, gaps, SERSClassicalEMJointExtensionDesigner().design(
        qualifications, gaps
    )


def test_cross_source_frontier_preserves_cost_vs_coverage_tradeoff():
    ryu = _reference("control:ryu", "paper:ryu")
    nima = _reference(
        "control:nima",
        "paper:nima",
        shell_geometry="uniform_thickness",
        environment="homogeneous_water",
        illumination="unpolarized_ensemble",
        observable="other_or_uncertain",
    ).model_copy(update={"uniform_shell_thickness_nm": 4.0})
    nano = _reference(
        "control:nano",
        "paper:nano",
        observable="ensemble_extinction_spectrum",
    )

    _, _, result = _design(ryu, nima, nano)

    assert result.design_decision == "review_joint_extension_frontier"
    assert result.minimum_axis_count_for_any_unlock == 4
    assert result.minimum_axis_count_for_cross_source_unlock == 6
    assert result.minimum_axis_count_for_full_coverage == 6

    summary = [
        (
            row.axis_count,
            row.unlocked_control_count,
            row.unlocked_independent_source_count,
        )
        for row in result.candidates
    ]
    assert summary == [(4, 1, 1), (5, 1, 1), (6, 3, 3)]
    assert all(row.backend_extension_authorized is False for row in result.candidates)
    assert result.backend_extension_authorized is False


def test_same_source_variants_do_not_count_as_cross_source_unlock():
    refs = [
        _reference("control:a", "paper:1"),
        _reference("control:b", "paper:1"),
        _reference(
            "control:c",
            "paper:2",
            shell_geometry="uniform_thickness",
            environment="homogeneous_water",
            illumination="unpolarized_ensemble",
            observable="other_or_uncertain",
        ).model_copy(update={"uniform_shell_thickness_nm": 4.0}),
    ]
    _, _, result = _design(*refs)
    smallest = result.candidates[0]
    assert smallest.axis_count == 4
    assert smallest.unlocked_independent_source_count == 1
    cross = [row for row in result.candidates if row.unlocks_cross_source_controls]
    assert cross
    assert min(row.axis_count for row in cross) == 6


def test_designer_requires_joint_design_decision():
    ref = _reference(
        "control:a",
        "paper:1",
        particle_shape="spherocylinder_like",
        shell_geometry="uniform_thickness",
        environment="substrate_with_water",
        illumination="normal_incidence_plane_wave",
        polarization="parallel_to_nanorod_long_axis",
    ).model_copy(update={"uniform_shell_thickness_nm": 5.0})
    qualifications = SERSClassicalEMPositiveControlQualifier().qualify(
        SERSEMPositiveControlReferenceBundle(references=[ref])
    )
    gaps = SERSClassicalEMBackendGapAssessor().assess(qualifications)
    assert gaps.extension_decision == "collect_more_independent_controls"
    with pytest.raises(ValueError, match="joint extension design requires"):
        SERSClassicalEMJointExtensionDesigner().design(qualifications, gaps)


def test_lineage_mismatch_fails_closed():
    refs = [
        _reference("control:a", "paper:1"),
        _reference("control:b", "paper:2"),
    ]
    qualifications = SERSClassicalEMPositiveControlQualifier().qualify(
        SERSEMPositiveControlReferenceBundle(references=refs)
    )
    gaps = SERSClassicalEMBackendGapAssessor().assess(qualifications)
    bad = gaps.model_copy(update={"source_qualification_bundle_id": "wrong"})
    with pytest.raises(ValueError, match="lineage mismatch"):
        SERSClassicalEMJointExtensionDesigner().design(qualifications, bad)


def test_joint_design_never_authorizes_execution_or_validation():
    _, _, result = _design(
        _reference("control:a", "paper:1"),
        _reference("control:b", "paper:2"),
    )
    assert result.backend_extension_authorized is False
    assert result.positive_control_execution_authorized is False
    assert result.model_validation_permitted is False
    assert result.target_hypothesis_verdict_permitted is False
    assert result.feedback_generation_authority is False
