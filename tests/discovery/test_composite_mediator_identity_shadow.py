from __future__ import annotations

from pipeline_core.discovery.composite_mediator_identity_shadow import (
    CompositeMediatorIdentityShadowCritic,
)


critic = (
    CompositeMediatorIdentityShadowCritic()
)


def test_shape_size_shared_object_role_is_lexically_supported():

    review = critic.review(
        mediator_tokens=(
            "shape",
            "size",
        ),

        source_subject=(
            "superlattice structural order"
        ),

        source_relation=(
            "VARIES_WITH"
        ),

        source_object=(
            "size and shape uniformity of "
            "nanoparticle building blocks"
        ),

        target_subject=(
            "plasmonic properties"
        ),

        target_relation=(
            "VARIES_WITH"
        ),

        target_object=(
            "nanostructure size, shape, "
            "composition, and arrangement"
        ),
    )

    assert (
        review.status
        ==
        "LEXICALLY_SUPPORTED"
    )

    assert (
        review.common_mediator_fields
        ==
        ("object",)
    )

    assert (
        review.multi_token_mediator
        is True
    )


def test_e_field_enhancement_vs_sers_enhancement_is_ambiguous():

    review = critic.review(
        mediator_tokens=(
            "enhancement",
        ),

        source_subject=(
            "maximum local E-field enhancement"
        ),

        source_relation=(
            "VARIES_WITH"
        ),

        source_object=(
            "array architecture"
        ),

        target_subject=(
            "SERS enhancement"
        ),

        target_relation=(
            "VARIES_WITH"
        ),

        target_object=(
            "interparticle spacing"
        ),
    )

    assert (
        review.status
        ==
        "IDENTITY_AMBIGUOUS"
    )

    assert (
        review.common_mediator_fields
        ==
        ("subject",)
    )

    assert (
        review.context_overlap_tokens
        ==
        ()
    )


def test_identical_sers_enhancement_context_is_supported():

    review = critic.review(
        mediator_tokens=(
            "enhancement",
        ),

        source_subject=(
            "SERS enhancement"
        ),

        source_relation=(
            "VARIES_WITH"
        ),

        source_object=(
            "interparticle spacing"
        ),

        target_subject=(
            "SERS enhancement"
        ),

        target_relation=(
            "VARIES_WITH"
        ),

        target_object=(
            "anode material"
        ),
    )

    assert (
        review.status
        ==
        "LEXICALLY_SUPPORTED"
    )

    assert (
        review.context_overlap_tokens
        ==
        ("sers",)
    )


def test_same_token_in_different_sro_roles_is_incompatible():

    review = critic.review(
        mediator_tokens=(
            "assembly",
        ),

        source_subject=(
            "nanoparticle assembly"
        ),

        source_relation=(
            "PROMOTES"
        ),

        source_object=(
            "interparticle hot spots"
        ),

        target_subject=(
            "SERS performance"
        ),

        target_relation=(
            "VARIES_WITH"
        ),

        target_object=(
            "Ag assembly method"
        ),
    )

    assert (
        review.status
        ==
        "ROLE_INCOMPATIBLE"
    )

    assert (
        review.common_mediator_fields
        ==
        ()
    )


def test_laser_wavelength_vs_laser_power_is_ambiguous():

    review = critic.review(
        mediator_tokens=(
            "laser",
        ),

        source_subject=(
            "nanoparticle size"
        ),

        source_relation=(
            "VARIES_WITH"
        ),

        source_object=(
            "laser wavelength"
        ),

        target_subject=(
            "SERS spectra"
        ),

        target_relation=(
            "VARIES_WITH"
        ),

        target_object=(
            "laser power"
        ),
    )

    assert (
        review.status
        ==
        "IDENTITY_AMBIGUOUS"
    )


def test_single_token_with_role_context_overlap_is_supported():

    review = critic.review(
        mediator_tokens=(
            "nanoparticle",
        ),

        source_subject=(
            "Co3O4 nanowire gaps"
        ),

        source_relation=(
            "PROMOTES"
        ),

        source_object=(
            "Ag nanoparticle growth"
        ),

        target_subject=(
            "plasmonic properties"
        ),

        target_relation=(
            "VARIES_WITH"
        ),

        target_object=(
            "Au-Ag nanoparticle proportion"
        ),
    )

    assert (
        review.status
        ==
        "LEXICALLY_SUPPORTED"
    )

    assert (
        "ag"
        in
        review.context_overlap_tokens
    )
