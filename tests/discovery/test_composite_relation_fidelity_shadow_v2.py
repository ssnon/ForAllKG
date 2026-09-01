from __future__ import annotations

from types import SimpleNamespace

from pipeline_core.discovery.composite_relation_fidelity_shadow import (
    CompositeRelationFidelityShadowCritic,
)

from pipeline_core.discovery.composite_relation_fidelity_shadow_v2 import (
    CompositeRelationFidelityShadowCriticV2,
)


def _q38_axis():
    return SimpleNamespace(
        axis_id="axis:q38",
        source_mode=(
            "task_conditioned_composite_bridge_projection"
        ),
        proposed_subject=(
            "structural motif Au nanorod@TiO2 composite "
            "architecture within Au NRs@TiO2 substrate"
        ),
        proposed_relation=(
            "MAY_RELATE_TO_VIA_COMPOSED_CANDIDATE_BRIDGE"
        ),
        proposed_object=(
            "measured or mechanistically interpreted "
            "SERS/plasmonic behavior"
        ),
        rendered_path=(
            "structural motif Au nanorod@TiO2 composite "
            "architecture within Au NRs@TiO2 substrate "
            "-> [UNVERIFIED SOURCE RELATION: "
            "superlattice structural order | VARIES_WITH | "
            "size and shape uniformity of nanoparticle "
            "building blocks] "
            "-> [SHARED MEDIATOR: shape, size] "
            "-> [UNVERIFIED TARGET RELATION: "
            "plasmonic properties | VARIES_WITH | "
            "nanostructure size, shape, composition, "
            "and arrangement] "
            "-> measured or mechanistically interpreted "
            "SERS/plasmonic behavior"
        ),
    )


def _q40_axis():
    return SimpleNamespace(
        axis_id="axis:q40",
        source_mode=(
            "task_conditioned_composite_bridge_projection"
        ),
        proposed_subject=(
            "architecture of p-AuAg-NWs array, including "
            "Porous structure on nanowire tip"
        ),
        proposed_relation=(
            "MAY_RELATE_TO_VIA_COMPOSED_CANDIDATE_BRIDGE"
        ),
        proposed_object=(
            "measured or mechanistically interpreted "
            "SERS/plasmonic behavior"
        ),
        rendered_path=(
            "architecture of p-AuAg-NWs array, including "
            "Porous structure on nanowire tip "
            "-> [UNVERIFIED SOURCE RELATION: "
            "maximum local E-field enhancement | VARIES_WITH | "
            "array architecture] "
            "-> [SHARED MEDIATOR: enhancement] "
            "-> [UNVERIFIED TARGET RELATION: "
            "SERS enhancement | VARIES_WITH | "
            "interparticle spacing] "
            "-> measured or mechanistically interpreted "
            "SERS/plasmonic behavior"
        ),
    )


def _card(hypothesis_id, *observables):
    return SimpleNamespace(
        hypothesis_id=hypothesis_id,
        predicted_observations=[
            SimpleNamespace(
                observable=text
            )
            for text in observables
        ],
    )


def test_relationship_variability_form_is_recovered():
    axis = _q38_axis()

    card = _card(
        "h:q38-r1",
        (
            "A systematic relationship between Au nanorod "
            "size and shape uniformity and the spatial "
            "variability of local SERS enhancement across "
            "otherwise comparable Au NRs@TiO2 regions"
        ),
    )

    base = (
        CompositeRelationFidelityShadowCritic()
        .review(
            axis,
            card,
        )
    )

    v2 = (
        CompositeRelationFidelityShadowCriticV2()
        .review(
            axis,
            card,
        )
    )

    assert base.status == "fail"
    assert v2.status == "pass"

    row = v2.observations[0]

    assert row.matched_source_state is True
    assert row.mediator_contrast is True
    assert row.outcome_contrast is True
    assert (
        row.complete_conditional_consequence
        is True
    )


def test_variation_in_shape_and_sers_form_is_recovered():
    axis = _q38_axis()

    card = _card(
        "h:q38-r2",
        (
            "Among otherwise comparable Au NRs@TiO2 "
            "substrates, variation in rod shape and size "
            "together with their packing is associated "
            "with variation in the spatial distribution "
            "or relative intensity of SERS enhancement "
            "environments."
        ),
    )

    base = (
        CompositeRelationFidelityShadowCritic()
        .review(
            axis,
            card,
        )
    )

    v2 = (
        CompositeRelationFidelityShadowCriticV2()
        .review(
            axis,
            card,
        )
    )

    assert base.status == "fail"
    assert v2.status == "pass"


def test_relational_language_without_source_state_still_fails():
    axis = _q38_axis()

    card = _card(
        "h:q38-r3",
        (
            "The association between nanorod shape-size "
            "uniformity, spatial arrangement, and spatial "
            "variability of local SERS enhancement."
        ),
        (
            "The dependence of the spatial distribution "
            "of SERS intensity on combined nanorod "
            "shape-size uniformity and spatial arrangement."
        ),
    )

    v2 = (
        CompositeRelationFidelityShadowCriticV2()
        .review(
            axis,
            card,
        )
    )

    assert v2.status == "fail"

    assert not any(
        row.complete_conditional_consequence
        for row in v2.observations
    )


def test_q40_no_explicit_mediator_comparison_remains_fail():
    axis = _q40_axis()

    card = _card(
        "h:q40-r1",
        (
            "SERS or plasmonic observables differ between "
            "porous nanowire-tip regions and comparatively "
            "less porous regions of the same p-AuAg-NWs "
            "architecture."
        ),
        (
            "Changing the architecture or degree of "
            "porosity at the nanowire tips is accompanied "
            "by a corresponding change in the SERS or "
            "plasmonic response."
        ),
    )

    v2 = (
        CompositeRelationFidelityShadowCriticV2()
        .review(
            axis,
            card,
        )
    )

    assert v2.status == "fail"


def test_missing_requested_outcome_still_fails():
    axis = _q38_axis()

    card = _card(
        "h:no-outcome",
        (
            "Among otherwise comparable Au NRs@TiO2 "
            "substrates, variation in nanorod shape and "
            "size is observed across regions."
        ),
    )

    v2 = (
        CompositeRelationFidelityShadowCriticV2()
        .review(
            axis,
            card,
        )
    )

    assert v2.status == "fail"
