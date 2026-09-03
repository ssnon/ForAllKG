from pipeline_core.discovery.novelty_closure_planner import (
    _remove_moderator_structure_terms,
)


def test_moderator_base_drops_conditional_dependence():
    result = _remove_moderator_structure_terms(
        (
            "M H iCOHP",
            "hydrogen adsorption free energy",
            "conditional dependence",
        )
    )

    assert result == (
        "M H iCOHP",
        "hydrogen adsorption free energy",
    )


def test_moderator_base_drops_conditional_association():
    result = _remove_moderator_structure_terms(
        (
            "M H iCOHP",
            "HER activity",
            "conditional association",
        )
    )

    assert result == (
        "M H iCOHP",
        "HER activity",
    )


def test_scientific_terms_are_not_removed():
    result = _remove_moderator_structure_terms(
        (
            "interparticle spacing",
            "SERS intensity",
            "laser power",
        )
    )

    assert result == (
        "interparticle spacing",
        "SERS intensity",
        "laser power",
    )
